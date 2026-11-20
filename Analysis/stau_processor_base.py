"""Shared functionality for the stau processors.

Anything that is used by both stau_processor_signal.py and
stau_processor_wjets.py lives here. Edit a function once here and both
processors pick up the change (unless a processor overrides it).
"""
import pepper
import awkward as ak
from functools import partial
import numpy as np
import mt2
import numba as nb
import coffea
import logging
import os
from copy import copy

from coffea.nanoevents import NanoAODSchema

logger = logging.getLogger(__name__)

class BaseProcessor(pepper.ProcessorBasicPhysics):
    config_class = pepper.ConfigSTau
    
    def zero_handler(func):
        def _function(self, data, *args, **kwargs):
            if len(data) > 0: return func(self, data, *args, **kwargs)
            else: return ak.Array([])
        return _function
    

    def __init__(self, config, eventdir):
        config["histogram_format"] = "root"
        # Need to call parent init to make histograms and such ready
        super().__init__(config, eventdir)

    # ------------------------------------------------------------------
    # Shared selection blocks (call these from process_selection)
    # ------------------------------------------------------------------
    def add_trigger(self, selector, dsname, is_mc):
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname,
            is_mc,
            self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"]
        )

        if self.config.get("apply_trigger", True):
            selector.add_cut(
                "Trigger",
                partial(self.passing_trigger, pos_triggers, neg_triggers)
            )
            logger.info(f"Positive triggers: {pos_triggers}")
            logger.info(f"Negative triggers: {neg_triggers}")

        else:
            logger.info("Trigger requirement disabled for offline efficiency denominator")

    def add_quality_cuts(self, selector, is_mc):
        selector.add_cut("MET filters", partial(self.met_filters, is_mc))
        selector.add_cut("PV", lambda data: data["PV"].npvsGood > 0)

    # ------------------------------------------------------------------
    # Weights / reweighting
    # ------------------------------------------------------------------
    @zero_handler
    def HEM_veto(self, data, is_mc):
        weight = np.ones(len(data), dtype=np.float32)
        jets = data["Jet"]
        elctron = data["Electron"]
        electron_in15or16_hem = ((elctron.pt > 20) & (elctron.eta > -3.0) & (elctron.eta < -1.3) & (elctron.phi > -1.57) & (elctron.phi < -0.87))
        jet_in15or16_hem = ((jets.pt > 20) & (jets.eta > -3.2) & (jets.eta < -1.3) & (jets.phi > -1.77) & (jets.phi < -0.67))
        in_hem = (ak.any(electron_in15or16_hem, axis=-1) | ak.any(jet_in15or16_hem, axis=-1))
        if is_mc:
            weight[in_hem] = (1 - 0.66)
        else:
            issue_period = (data.run >= 319077)
            weight[in_hem & issue_period] = 0.0
        return weight

    @zero_handler
    def do_dy_jet_reweighting(self, data):
        njet = data["LHE"]["Njets"]
        weights = self.config["DY_jet_reweight"][njet]
        return weights

    @zero_handler
    def do_w_jet_reweighting(self, data):
        njet = data["LHE"]["Njets"]
        weights = self.config["W_jet_reweight"][njet]
        return weights

    @zero_handler
    def sum_ll_gen(self, data):
        part = data["GenPart"]
        part = part[part.hasFlags("isLastCopy")
                    & (part.hasFlags("fromHardProcess")
                       & ((abs(part["pdgId"]) == 11)
                          | (abs(part["pdgId"]) == 13)
                          | (abs(part["pdgId"]) == 12)
                          | (abs(part["pdgId"]) == 14)
                          | (abs(part["pdgId"]) == 15)
                          | (abs(part["pdgId"]) == 16)))
                    | (part.hasFlags("isDirectHardProcessTauDecayProduct"))
                    ]
        sum_p4 = part.sum(axis=1)  # sum over all particles in event
        return sum_p4

    @zero_handler
    def get_dy_gen_sfs(self, data, is_mc, dsname):
        weight = np.ones(len(data))
        # The weights are taken from the following repository:
        # https://github.com/cms-tau-pog/TauFW/tree/master/PicoProducer/data/zpt
        if is_mc and (
                dsname.startswith("DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8")
                or dsname.startswith("DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8")
                or dsname.startswith("DY1JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")
                or dsname.startswith("DY2JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")
                or dsname.startswith("DY3JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")
                or dsname.startswith("DY4JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")):
            z_boson = data["sum_ll_gen"]
            dy_gen_sfs = self.config["DY_ZptLO_weights"](mass=z_boson.mass, pt=z_boson.pt)
            weight *= ak.to_numpy(dy_gen_sfs)
            return weight
        else:
            return weight

    @zero_handler
    def MET_cut(self, data):
        return data["PFMET"].pt > self.config["MET_cut"]

    # ------------------------------------------------------------------
    # Gen-level helpers
    # ------------------------------------------------------------------
    def gen_lep(self, data):
        if len(data) == 0:
            return {"gen_mu": ak.Array([]),
                    "gen_ele": ak.Array([])}
        gen_mu = data.GenPart[
            (abs(data.GenPart.pdgId) == 13)
            & data.GenPart.hasFlags(["isHardProcess"])
            & data.GenPart.hasFlags(["isFirstCopy"])
        ]
        gen_ele = data.GenPart[
            (abs(data.GenPart.pdgId) == 11)
            & data.GenPart.hasFlags(["isHardProcess"])
            & data.GenPart.hasFlags(["isFirstCopy"])
        ]
        return {
            "gen_mu": gen_mu,
            "gen_ele": gen_ele
        }

    # ------------------------------------------------------------------
    # Lepton selections
    # ------------------------------------------------------------------
    @zero_handler
    def select_muons(self, data):
        muons = data["Muon"]
        is_good = (
            (muons.pt > self.config["muon_veto_pt_min"])
            & (muons.eta < self.config["muon_veto_eta_max"])
            & (muons.eta > self.config["muon_veto_eta_min"])
            & (muons[self.config["muon_veto_ID"]] == 1)
            & (muons.pfIsoId >= self.config["muon_veto_pfIsoId"])
        )
        return muons[is_good]

    @zero_handler
    def select_electrons(self, data):
        ele = data["Electron"]
        is_good = (
            (ele.pt > self.config["elec_veto_pt"])
            & (ele.eta < self.config["elec_veto_eta_max"])
            & (ele.eta > self.config["elec_veto_eta_min"])
            & (ele[self.config["elec_veto"]] == 1)
            & (ele[self.config["elec_ID"]] == 1)
        )
        return ele[is_good]

    @zero_handler
    def jet_updated_flavour(self, data):
        jets = data["Jet_select"]
        # add hadronic tau flavour:
        tau_vis = data.GenVisTau[((data.GenVisTau.pt > 30) & (abs(data.GenVisTau.eta) < 2.4)
                                  & (data.GenVisTau.parent.hasFlags(["fromHardProcess"])))
                                 ]
        matches_tauhad, _ = jets.nearest(tau_vis, return_metric=True, threshold=0.4)
        matches_mu, _ = jets.nearest(data["gen_mu"], return_metric=True, threshold=0.4)
        matches_ele, _ = jets.nearest(data["gen_ele"], return_metric=True, threshold=0.4)

        updFlavour = ak.where(~ak.is_none(matches_tauhad, axis=1), 15, jets.partonFlavour)
        updFlavour = ak.where(~ak.is_none(matches_mu, axis=1), 13, updFlavour)
        updFlavour = ak.where(~ak.is_none(matches_ele, axis=1), 11, updFlavour)
        return np.abs(updFlavour)

    # ------------------------------------------------------------------
    # Jets and PF candidates
    # ------------------------------------------------------------------
    @zero_handler
    def jet_selection(self, data, from_collection="JetCHS"):
        jets = data[from_collection]
        jets = jets[(
            (self.config["jet_eta_min"] < jets.eta)
            & (jets.eta < self.config["jet_eta_max"])
            & (self.config["jet_pt_min"] < jets.pt)
            # & (jets.jetId >= self.config["jet_jetId"] )
        )]
        return jets

    @zero_handler
    def b_tagged_jet_cut(self, data, name="Jet"):
        # Events with no b-tagged jets
        jets = data[name]
        return ak.num(jets[jets.btagged], axis=1) == 0

    @zero_handler
    def getloose_jets(self, data):
        jets = data["Jet_select"]
        jets = jets[(jets.disTauTag_score1 >= self.config["loose_thr"])]
        return jets

    @zero_handler
    def gettight_jets(self, data):
        jets = data["Jet_select"]
        jets = jets[(jets.disTauTag_score1 >= self.config["tight_thr"])]
        n_after = ak.num(jets, axis=1)
        logger.info(
            f"after tight cut: "
            f"exactly 2 = {ak.sum(n_after == 2)}, "
            f">=2 = {ak.sum(n_after >= 2)}"
        )
        return jets

    @zero_handler
    def pfcand_valid(self, data):
        pfCands = data["PFCand"]
        is_good = (
            (pfCands.pt > self.config["pfcand_pt"])
            & (pfCands.eta < self.config["pfcand_eta_max"])
            & (pfCands.eta > self.config["pfcand_eta_min"])
        )
        pfCands_selected = pfCands[is_good]
        sort_idx = ak.argsort(pfCands_selected.pt, axis=-1, ascending=False)
        return pfCands_selected[sort_idx]

    @zero_handler
    def match_jet_to_pfcand(self, data, jet_name=None, pf_name=None, dR=0.4):
        jets = data[jet_name]
        pfcands = data[pf_name]

        # shape: events x jets x pfcands
        pf_b = pfcands[:, None, :]  # broadcast over jets
        deta = jets.eta[:, :, None] - pf_b.eta
        dphi = (jets.phi[:, :, None] - pf_b.phi + np.pi) % (2 * np.pi) - np.pi
        dr = np.sqrt(deta**2 + dphi**2)

        # pfcands duplicated per jet, then masked by dR
        pf_per_jet = ak.broadcast_arrays(jets.eta[:, :, None], pf_b)[1]
        return pf_per_jet[dr < dR]

    @zero_handler
    def get_matched_pfCands(self, data, match_object, dR=0.4):
        pfCands = self.match_jet_to_pfcand(data, jet_name=match_object, pf_name="PfCands", dR=dR)
        pfCands_lead = ak.firsts(pfCands, axis=-1)
        return pfCands_lead
    # ------------------------------------------------------------------
    # Kinematic variables
    # ------------------------------------------------------------------
    def delta_phi(self, phi1_ak, phi2_ak):
        phi1 = np.array(phi1_ak)
        phi2 = np.array(phi2_ak)
        assert phi1.shape == phi2.shape
        d = phi1 - phi2
        indx_pos = d > np.pi
        d[indx_pos] -= np.pi * 2
        indx_neg = d <= -np.pi
        d[indx_neg] += np.pi * 2
        return d

    @zero_handler
    def sum_jj(self, data):
        jets = data['Jet_select']
        j1 = jets[:, 0]
        j2 = jets[:, 1]
        mt1 = np.sqrt(j1.mass**2 + j1.pt**2)
        mt2_ = np.sqrt(j2.mass**2 + j2.pt**2)
        mjj2 = (
            j1.mass**2 + j2.mass**2
            + 2 * (
                mt1 * mt2_ * np.cosh(j1.eta - j2.eta)
                - j1.pt * j2.pt * np.cos(j1.phi - j2.phi)
            )
        )
        return np.sqrt(np.maximum(mjj2, 0))

    @zero_handler
    def missing_energy(self, data):
        # Selected jets
        jets = data["Jet_select"]
        HT_valid = ak.sum(jets.pt, axis=-1)
        px = ak.sum(jets.pt * np.cos(jets.phi), axis=-1)
        py = ak.sum(jets.pt * np.sin(jets.phi), axis=-1)
        HT_miss_valid = np.sqrt(px**2 + py**2)

        # All jets
        jets = data["Jet"]
        HT = ak.sum(jets.pt, axis=-1)
        px = ak.sum(jets.pt * np.cos(jets.phi), axis=-1)
        py = ak.sum(jets.pt * np.sin(jets.phi), axis=-1)
        HT_miss = np.sqrt(px**2 + py**2)
        min_MET_MHT = np.minimum(data["PFMET"].pt, HT_miss)
        
        return {
            "HT_valid": HT_valid,
            "HT_miss_valid": HT_miss_valid,
            "HT": HT,
            "HT_miss": HT_miss,
            "min_MET_MHT": min_MET_MHT
        }

    @zero_handler
    def mt_jets(self, data):
        jet1 = data["Jet_select"][:, 0]
        jet2 = data["Jet_select"][:, 1]
        met = data["PFMET"]

        delta_phi1 = self.delta_phi(jet1.phi, met.phi)
        mt_j1 = np.sqrt(2 * jet1.pt * met.pt * (1.0 - np.cos(delta_phi1)))

        delta_phi2 = self.delta_phi(jet2.phi, met.phi)
        mt_j2 = np.sqrt(2 * jet2.pt * met.pt * (1.0 - np.cos(delta_phi2)))

        return {
            "mt_jet1": mt_j1,
            "mt_jet2": mt_j2,
            "mt_sum": mt_j1 + mt_j2
        }

    @zero_handler
    def dphi_jet1_jet2(self, data):
        return self.delta_phi(data["Jet_select"][:, 0].phi,
                              data["Jet_select"][:, 1].phi)

    @zero_handler
    def dr_jet1_jet2(self, data):
        jets = data["Jet_select"]
        jet1 = jets[:, 0]
        jet2 = jets[:, 1]
        delta_eta = jet1.eta - jet2.eta
        delta_phi = self.delta_phi(jet1.phi, jet2.phi)
        return np.sqrt(delta_eta**2 + delta_phi**2)

    @zero_handler
    def dphi_min_cut(self, data):
        return abs(data["dphi_jet1_jet2"]) > self.config["dphi_j1j2"]

    @zero_handler
    def get_mt2(self, data):
        jet1 = data["Jet_select"][:, 0]
        jet2 = data["Jet_select"][:, 1]
        met = data["PFMET"]
        jet1_px = jet1.pt * np.cos(jet1.phi)
        jet1_py = jet1.pt * np.sin(jet1.phi)
        jet2_px = jet2.pt * np.cos(jet2.phi)
        jet2_py = jet2.pt * np.sin(jet2.phi)
        met_px = met.pt * np.cos(met.phi)
        met_py = met.pt * np.sin(met.phi)
        return mt2.mt2(
            jet1.mass, jet1_px, jet1_py,
            jet2.mass, jet2_px, jet2_py,
            met_px, met_py,
            0, 0
        )

    @zero_handler
    def binning_schema(self, data):
        jets = data["Jet_select"]
        met = data["PFMET"].pt
        jet2_pt = jets[:, 1].pt
        mt2_ = data["mt2_j1_j2_MET"]
        bins = np.full((len(met)), np.nan)
        B1 = (jet2_pt < 50) & (met < 250) & (mt2_ < 100)
        B2 = (jet2_pt < 50) & (met < 250) & (mt2_ >= 100)
        B3 = (jet2_pt < 50) & (met >= 250)
        B4 = (jet2_pt >= 50) & (jet2_pt < 100) & (met < 250) & (mt2_ < 100)
        B5 = (jet2_pt >= 50) & (jet2_pt < 100) & (met < 250) & (mt2_ >= 100)
        B6 = (jet2_pt >= 50) & (jet2_pt < 100) & (met >= 250)
        B7 = (jet2_pt >= 100) & (mt2_ < 100)
        B8 = (jet2_pt >= 100) & (mt2_ >= 100)
        bins[B1] = 1
        bins[B2] = 2
        bins[B3] = 3
        bins[B4] = 4
        bins[B5] = 5
        bins[B6] = 6
        bins[B7] = 7
        bins[B8] = 8
        return bins

    # ------------------------------------------------------------------
    # Tagger scores and yield prediction
    # ------------------------------------------------------------------
    @zero_handler
    def set_njets_pass(self, data):
        jets_score = data["Jet_select"].disTauTag_score1
        n_pass = []
        for score in self.config["score_pass"]:
            jets_pass = jets_score[(jets_score >= score)]
            passed = ak.num(jets_pass, axis=1)
            n_pass.append(passed)
        n_pass = ak.from_regular(
            np.stack(n_pass, axis=1), axis=-1)
        tight_wp = self.config["score_pass"].index(self.config["tight_thr"])
        tight_pass = n_pass[:, tight_wp]
        tight_bin0 = (n_pass[:, tight_wp] == 0)
        tight_bin1 = (n_pass[:, tight_wp] == 1)
        tight_bin2 = (n_pass[:, tight_wp] == 2)
        return {
            "n_pass": n_pass,
            "n_pass_score_bin": ak.local_index(n_pass, axis=1),
            "tight_pass": tight_pass,
            "tight_bin0": tight_bin0,
            "tight_bin1": tight_bin1,
            "tight_bin2": tight_bin2
        }

    def _fake_yields(self, jets, fake_func):
        """Predicted yields 0->1, 0->2, 1->2 tags for every score WP.

        fake_func() must return the per-jet fake rate.
        """
        weights_bin0to1 = []
        weights_bin0to2 = []
        weights_bin1to2 = []

        for score in self.config["score_pass"]:
            fake = fake_func()

            # from bin 0 to bin 1 and 2
            events_0tag = (ak.num(jets[(jets.disTauTag_score1 >= score)]) == 0)
            masked_jets = ak.broadcast_arrays(jets.disTauTag_score1, True)[1]
            masked_jets = events_0tag * masked_jets
            fake0 = ak.mask(fake, masked_jets)
            f_1, f_2 = fake0[:, 0], fake0[:, 1]
            from0to1 = (f_1 * (1 - f_2) + f_2 * (1 - f_1)) / ((1 - f_2) * (1 - f_1))
            from0to2 = (f_1 * f_2) / ((1 - f_2) * (1 - f_1))
            weights_bin0to1.append(ak.fill_none(from0to1, 0.0))
            weights_bin0to2.append(ak.fill_none(from0to2, 0.0))

            # from bin 1 to bin 2
            events_1tag = (ak.num(jets[(jets.disTauTag_score1 >= score)]) == 1)
            masked_jets = ak.broadcast_arrays(jets.disTauTag_score1, True)[1]
            masked_jets = events_1tag * masked_jets
            fake1 = ak.mask(fake, masked_jets)
            f_1, f_2 = fake1[:, 0], fake1[:, 1]
            from1to2 = (f_1 * f_2) / (f_1 * (1 - f_2) + f_2 * (1 - f_1))
            weights_bin1to2.append(ak.fill_none(from1to2, 0.0))

        return (
            ak.from_regular(np.stack(weights_bin0to1, axis=1), axis=-1),
            ak.from_regular(np.stack(weights_bin0to2, axis=1), axis=-1),
            ak.from_regular(np.stack(weights_bin1to2, axis=1), axis=-1),
        )

    @zero_handler
    def predict_yield(self, data, weight=None):
        jets = data["Jet_select"]
        fake_rate = self.config["jet_fake_rate"]

        yield_bin0to1, yield_bin0to2, yield_bin1to2 = self._fake_yields(
            jets, lambda: fake_rate(jet_pt=jets.pt, jet_dxy=jets.dxy))

        # now we need to each predicted yield assign cooresponding score bin
        score_bin = ak.local_index(yield_bin0to1, axis=1) + 1  # +1 because we skip first bin

        # One of the WP (called tight) is used for the final analysis
        # the weight for this will be saved separately to have one weight per event
        tight_wp = self.config["score_pass"].index(self.config["tight_thr"])

        return_cols = {
            "yield_bin0to1": weight * yield_bin0to1,
            "yield_bin1to2": weight * yield_bin1to2,
            "yield_bin0to2": weight * yield_bin0to2,
            "tight_yield_bin0to1": weight * yield_bin0to1[:, tight_wp],
            "tight_yield_bin1to2": weight * yield_bin1to2[:, tight_wp],
            "tight_yield_bin0to2": weight * yield_bin0to2[:, tight_wp],
            "score_bin": score_bin
        }

        if self.config["compute_systematics"]:
            variations = {
                "stat_up": lambda: fake_rate(jet_pt=jets.pt, jet_dxy=jets.dxy, variation="up"),
                "stat_down": lambda: fake_rate(jet_pt=jets.pt, jet_dxy=jets.dxy, variation="down"),
                # +/-10% systematic uncertainty for the fake rate
                "sys_up": lambda: fake_rate(jet_pt=jets.pt, jet_dxy=jets.dxy) * 1.1,
                "sys_down": lambda: fake_rate(jet_pt=jets.pt, jet_dxy=jets.dxy) * 0.9,
            }
            for sys, fake_func in variations.items():
                y01, y02, y12 = self._fake_yields(jets, fake_func)
                return_cols["tight_yield_bin0to1_" + sys] = weight * y01[:, tight_wp]
                return_cols["tight_yield_bin1to2_" + sys] = weight * y12[:, tight_wp]
                return_cols["tight_yield_bin0to2_" + sys] = weight * y02[:, tight_wp]

        return return_cols
