#   python3 -m pepper.runproc stau_processor_wjets.py configs/proc_2024/stau2024_wjets.json  --debug 
#   python3 -m pepper.runproc stau_processor_signal.py configs/proc_2024/stau2024_signal.json --debug           
import pepper
import logging
import awkward as ak
from functools import partial
import awkward as ak
import numpy as np
import pepper
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stau_processor_base import BaseProcessor  

logger = logging.getLogger(__name__)


class Processor(BaseProcessor):
    """WJets control-region processor.

    Everything that is shared with the signal processor comes from
    BaseProcessor. This file only contains:
      1. the WJets selection (process_selection / process_flav_study)
      2. WJets-only functions (muon tag, muon SFs, ...)
      3. OVERRIDES of base functions, kept because the WJets code still
         uses different collection names / config keys than the signal one.
         Delete an override to switch WJets to the shared base version.
    """
    def zero_handler(func):
        def _function(self, data, *args, **kwargs):
            if len(data) > 0: return func(self, data, *args, **kwargs)
            else: return ak.Array([])
        return _function

    def __init__(self, config, eventdir):
        super().__init__(config, eventdir)

        # if "pileup_reweighting" not in config:
        #     logger.warning("No pileup reweigthing specified")

        if "muon_sf" not in config or len(config["muon_sf"]) == 0:
            logger.warning("No muon scale factors specified")

        # It is not recommended to put anything as member variable into a
        # a Processor because the Processor instance is sent as raw bytes
        # between nodes when running on HTCondor.

    def process_selection(self, selector, dsname, is_mc, filler):

        if self.config["flavour_fake_rate_study"]:
            # Alternative processor which aims
            # calculating jet fake rate basing
            # on the gen level flavour matching
            # used only with WJet dataset at the moment
            selector.systematics["weight"] = \
                ak.full_like(selector.systematics["weight"], 1.0)
            self.process_flav_study(selector, dsname, is_mc, filler)
            return

        if is_mc and (dsname.startswith("WJetsToLNu")
                      or dsname.startswith("W1JetsToLNu")
                      or dsname.startswith("W2JetsToLNu")
                      or dsname.startswith("W3JetsToLNu")
                      or dsname.startswith("W4JetsToLNu")):
            selector.systematics["weight"] = \
                ak.full_like(selector.systematics["weight"], 1.0)
            selector.add_cut("reset_weight", lambda data: ak.Array(np.ones(len(data))))
            selector.add_cut("W jet reweighting",
                             partial(self.do_w_jet_reweighting))

        if is_mc and (dsname.startswith("DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8")
                      or dsname.startswith("DY1JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")
                      or dsname.startswith("DY2JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")
                      or dsname.startswith("DY3JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")
                      or dsname.startswith("DY4JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8")):
            selector.add_cut("DY jet reweighting",
                             partial(self.do_dy_jet_reweighting))

        # Triggers (shared with signal, see BaseProcessor)
        self.add_trigger(selector, dsname, is_mc)

        # if is_mc and "pileup_reweighting" in self.config:
        #     selector.add_cut("Pileup reweighting", partial(
        #         self.do_pileup_reweighting, dsname))

        # MET cut
        selector.add_cut("MET", self.MET_cut)

        # MET filters + PV cut (shared with signal, see BaseProcessor)
        self.add_quality_cuts(selector, is_mc)

        # one tight tag muon
        selector.set_column("Muon_tag", self.muons_tag)
        selector.add_cut("single_muon_tag", self.single_muon_tag_cut)
        selector.add_cut("Lead_pass_trigger", self.lead_pass_trigger)
        selector.add_cut("muon_sfs", partial(self.get_muon_sfs, is_mc=is_mc))

        # --- Further WJets steps -------------------
        selector.set_column("muon_veto", self.muons_veto)
        selector.add_cut("loose_muon_veto", self.loose_muon_veto_cut)
        selector.set_column("electron_veto", self.electron_veto)
        selector.add_cut("loose_electron_veto", self.loose_electron_veto_cut)
        # selector.set_column("mt_muon", partial(self.mt, name="Muon_tag"))
        # selector.add_cut("mt_muon", self.mt_muon_cut)
        # if is_mc:
        #     selector.set_column("sum_ll_gen", self.sum_ll_gen)
        # selector.add_cut("dy_gen_sfs", partial(self.get_dy_gen_sfs, is_mc=is_mc, dsname=dsname))
        # selector.set_column("Jet_select", self.jet_selection)
        # if self.config["b-veto"]:
        #     selector.add_cut("b-veto", partial(self.b_tagged_jet_cut, name="Jet_select"))
        # ---------------------- jet selections - to match baseline region ----------------------
        # selector.set_column("Jet_select", self.jet_veto_drop)
        # selector.add_cut("has_more_two_jets", self.has_more_two_jets)
        # selector.set_column("Jet_select", self.getloose_jets)
        # selector.set_column("PfCands", self.pfcand_valid)
        # selector.set_column("Jet_lead_pfcand", partial(self.get_matched_pfCands, match_object="Jet_select", dR=0.4))
        # selector.set_column("Jet_select", self.set_jet_dxy)
        # selector.add_cut("mt_muon2", self.mt_muon_cut)
        # selector.add_cut("has_more_two_loose-jets", self.has_more_two_jets)
        # selector.add_cut("two_loose_jets", self.has_two_jets)
        # selector.set_column("sum_jj", self.sum_jj)
        # selector.set_multiple_columns(self.mt_jets)
        # selector.set_column("dphi_jet1_jet2", self.dphi_jet1_jet2)
        # selector.add_cut("dphi_min_cut", self.dphi_min_cut)
        # selector.set_column("mt2_j1_j2_MET", self.get_mt2)
        # selector.set_column("binning_schema", self.binning_schema)
        # selector.set_multiple_columns(self.set_njets_pass)
        # if self.config["predict_yield"]:
        #     selector.set_multiple_columns(partial(self.predict_yield, weight=selector.systematics["weight"]))
        # selector.add_cut("two_loose_jets_final", self.has_two_jets)
        # selector.set_column("Jet_select", self.gettight_jets)
        # selector.add_cut("two_tight_jets", self.has_two_jets)

    # Gen Study -------------------------------------------------------------------------

    def process_flav_study(self, selector, dsname, is_mc, filler):
        selector.set_multiple_columns(self.gen_lep)
        # add cuts and selections on the jets
        selector.set_column("Jet_select", self.jet_selection_genveto)
        selector.set_column("Jet_select", self.getloose_jets)
        selector.set_column("PfCands", self.pfcand_valid)
        selector.set_column("Jet_lead_pfcand", partial(self.get_matched_pfCands, match_object="Jet_select", dR=0.4))
        selector.set_column("Jet_select", self.set_jet_dxy)
        selector.add_cut("two_loose_jets", self.has_two_jets)
        # Variables related to the two jets:
        selector.set_column("sum_jj", self.sum_jj)
        selector.set_multiple_columns(self.mt_jets)
        selector.set_column("dphi_jet1_jet2", self.dphi_jet1_jet2)
        selector.set_column("mt2_j1_j2_MET", self.get_mt2)
        selector.add_cut("two_loose_jets_final", self.has_two_jets)

    # ==================================================================
    # 2. WJets-only functions
    # ==================================================================
    @zero_handler
    def lead_pass_trigger(self, data):
        lead_muons = data["Muon_tag"][:, :1]
        trig_muons = data["TrigObj"]
        trig_muons = trig_muons[
            (abs(trig_muons.id) == 13)
            & (trig_muons.filterBits >= 8)
        ]
        matches, dRlist = lead_muons.nearest(trig_muons, return_metric=True, threshold=0.2)
        has_matched = ~ak.is_none(matches, axis=1)
        return has_matched[:, 0]  # since lead_muons has form: [ [muon1], [muon2], ....]

    @zero_handler
    def get_muon_sfs(self, data, is_mc):
        weight = np.ones(len(data))
        if is_mc:
            muons = data["Muon_tag"]
            id_iso_sfs, systematics = \
                self.muon_sfs(muons, sfs_name_config="muon_sf")
            weight *= ak.to_numpy(id_iso_sfs)
            muon_leading = muons[:, :1]  # trigger_sfs are applied only to leading muon
            mu_trigger_sfs, systematics_trig = \
                self.muon_sfs(muon_leading, sfs_name_config="muon_sf_trigger")
            weight *= ak.to_numpy(mu_trigger_sfs)
            systematics.update(systematics_trig)
            return weight, systematics
        else:
            return weight

    def muon_sfs(self, muons, sfs_name_config="muon_sf"):
        """Compute identification and isolation scale factors for
           leptons (electrons and muons). Also possible
           to use for muon trigger scale factors."""
        weight = np.ones(len(muons))
        systematics = {}
        # Muon identification and isolation efficiency
        for i, sffunc in enumerate(self.config[sfs_name_config]):
            params = {}
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(muons.eta)
                else:
                    params[dimlabel] = getattr(muons, dimlabel)
            central = ak.prod(sffunc(**params), axis=1)
            key = f"{sfs_name_config}{i}"
            if self.config["compute_systematics"]:
                if ("split_muon_uncertainty" not in self.config
                        or not self.config["split_muon_uncertainty"]):
                    unctypes = ("",)
                else:
                    unctypes = ("stat ", "syst ")
                for unctype in unctypes:
                    up = ak.prod(sffunc(
                        **params, variation=f"{unctype}up"), axis=1)
                    down = ak.prod(sffunc(
                        **params, variation=f"{unctype}down"), axis=1)
                    systematics[key + unctype.replace(" ", "")] = (
                        up / central, down / central)
            weight = weight * central
        return weight, systematics

    @zero_handler
    def single_muon_tag_cut(self, data):
        return ak.num(data["Muon_tag"]) == 1

    @zero_handler
    def loose_muon_veto_cut(self, data):
        return ak.num(data["muon_veto"]) == 0

    @zero_handler
    def loose_electron_veto_cut(self, data):
        return ak.num(data["electron_veto"]) == 0

    @zero_handler
    def mt_muon_cut(self, data):
        return ((data["mt_muon"] > self.config["mt_muon_min"]) & (data["mt_muon"] < self.config["mt_muon_max"]))

    @zero_handler
    def muons_tag(self, data):
        muons = data["Muon"]
        is_good = (
            (muons.pt > self.config["muon_pt_min"])
            & (muons.eta < self.config["muon_eta_max"])
            & (muons.eta > self.config["muon_eta_min"])
            & (muons[self.config["muon_ID"]] == 1)
            & (muons.pfIsoId >= self.config["muon_pfIsoId"])
            & (abs(muons.dxy) <= self.config["muon_absdxy"])
            & (abs(muons.dz) <= self.config["muon_absdz"])
        )
        return muons[is_good]

    @zero_handler
    def muons_veto(self, data):
        muons = data["Muon"]

        is_good = (
            (muons.pt > self.config["muon_veto_pt_min"])
            & (muons.eta < self.config["muon_veto_eta_max"])
            & (muons.eta > self.config["muon_veto_eta_min"])
            & (muons[self.config["muon_veto_ID"]] == 1)
            & (muons.pfIsoId >= self.config["muon_veto_pfIsoId"])
        )

        is_tag = (
            (muons.pt > self.config["muon_pt_min"])
            & (muons.eta < self.config["muon_eta_max"])
            & (muons.eta > self.config["muon_eta_min"])
            & (muons[self.config["muon_ID"]] == 1)
            & (muons.pfIsoId >= self.config["muon_pfIsoId"])
            & (abs(muons.dxy) <= self.config["muon_absdxy"])
            & (abs(muons.dz) <= self.config["muon_absdz"])
        )

        return muons[(is_good & (~is_tag))]

    @zero_handler
    def electron_veto(self, data):
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
    def jet_veto_drop(self, data):
        jets = data["Jet_select"]
        map_veto = self.config["jet_veto_map"]
        mask_per_jet = map_veto(eta=jets.eta, phi=jets.phi)
        jets = jets[mask_per_jet < 1]
        return jets

    @zero_handler
    def has_more_two_jets(self, data):
        jets = data["Jet_select"]
        return ak.num(jets) >= 2

    @zero_handler
    def n_jets(self, data):
        return ak.num(data["valid_jets"])

    @zero_handler
    def leading_jet(self, data, order=0):
        jets = data["Jet_select"]
        idx_leading = \
            ak.argsort(jets.pt, ascending=False)[:, order:order + 1]
        jets = jets[idx_leading]
        return jets

    @zero_handler
    def jets_available(self, data, n_available=1):
        jets = data["Jet_select"][~ak.is_none(data["Jet_select"].pt, axis=-1)]
        return ak.num(jets) >= n_available

    @zero_handler
    def mt(self, data, name):
        visible = ak.firsts(data[name])
        MET = data["MET"]
        one_min_cs = 1.0 - np.cos(self.delta_phi(visible.phi, MET.phi))
        prod = 2 * visible.pt * MET.pt
        return np.sqrt(prod * one_min_cs)

    @zero_handler
    def jet_selection_genveto(self, data):
        jets = data["Jet"]
        jets = jets[(
            (self.config["jet_eta_min"] < jets.eta)
            & (jets.eta < self.config["jet_eta_max"])
            & (self.config["jet_pt_min"] < jets.pt)
            & (jets.jetId >= self.config["jet_jetId"])
        )]

        # add hadronic tau flavour:
        tau_vis = data.GenVisTau[((data.GenVisTau.pt > 30) & (abs(data.GenVisTau.eta) < 2.4)
                                  & (data.GenVisTau.parent.hasFlags(["fromHardProcess"])))
                                 ]
        matches_tauhad, _ = jets.nearest(tau_vis, return_metric=True, threshold=self.config["tag_muon_veto_dR"])
        updFlavour = ak.where(~ak.is_none(matches_tauhad, axis=-1), 15, jets.partonFlavour)
        jets["partonFlavour_upd"] = updFlavour

        # mask the jets if they match to gen level but not hadron tau
        matches_tau, _ = jets.nearest(data["gen_tau"], return_metric=True, threshold=self.config["tag_muon_veto_dR"])
        matches_mu, _ = jets.nearest(data["gen_mu"], return_metric=True, threshold=self.config["tag_muon_veto_dR"])
        matches_ele, _ = jets.nearest(data["gen_ele"], return_metric=True, threshold=self.config["tag_muon_veto_dR"])

        jets = jets[
            ak.is_none(matches_ele, axis=1)
            & ak.is_none(matches_mu, axis=1)
            & (ak.is_none(matches_tau, axis=1) | (~ak.is_none(matches_tau, axis=1) & ~ak.is_none(matches_tauhad, axis=1)))
        ]
        return jets

    # ==================================================================
    # 3. OVERRIDES of BaseProcessor functions
    #    (WJets still uses the old collections / config keys.)
    # ==================================================================
    @zero_handler
    def MET_cut(self, data):  # base uses config["MET_cut"]
        return data["PFMET"].pt > self.config["MET_pt"]

    def gen_lep(self, data):  # base version has no gen_tau
        gen_tau = data.GenPart[
            (abs(data.GenPart.pdgId) == 15)
            & data.GenPart.hasFlags(["isHardProcess"])
            & data.GenPart.hasFlags(["isFirstCopy"])
        ]
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
            "gen_tau": gen_tau,
            "gen_mu": gen_mu,
            "gen_ele": gen_ele
        }

    @zero_handler
    def b_tagged_jet_cut(self, data, name="Jet"):  # base uses jets.btagged
        jets = data[name]
        # Jet_btagDeepFlavB satisfies the Medium (>0.2783) WP:
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
        year = self.config["year"]
        tagger = "deepjet"
        wp = "medium"
        wptuple = pepper.scale_factors.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise pepper.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        b_tagged_idx = jets["btagDeepFlavB"] > getattr(wptuple, wp)
        return ak.num(jets[b_tagged_idx], axis=1) == 0

    @zero_handler
    def jet_selection(self, data):  # base: JetCHS, no jetId, no muon isolation
        jets = data["Jet"]
        jets = jets[(
            (self.config["jet_eta_min"] < jets.eta)
            & (jets.eta < self.config["jet_eta_max"])
            & (self.config["jet_pt_min"] < jets.pt)
            & (jets.jetId >= self.config["jet_jetId"])
        )]
        matches_h, dRlist = jets.nearest(data["Muon_tag"], return_metric=True, threshold=self.config["tag_muon_veto_dR"])
        isoJets = jets[ak.is_none(matches_h, axis=-1)]
        return isoJets

    @zero_handler
    def pfcand_valid(self, data):  # base: "PFCand", no track requirement
        pfCands = data["PFCandidate"]
        is_good = (
            (pfCands.pt > self.config["pfcand_pt"])
            & (pfCands.eta < self.config["pfcand_eta_max"])
            & (pfCands.eta > self.config["pfcand_eta_min"])
            & (pfCands[self.config["track"]])
        )
        pfCands_selected = pfCands[is_good]
        sort_idx = ak.argsort(pfCands_selected.pt, axis=-1, ascending=False)
        return pfCands_selected[sort_idx]

    @zero_handler
    def match_jet_to_pfcand(self, data, jet_name=None, pf_name=None, dR=0.4):
        '''
        Match all particle-flow candidates to every jet in the [jet_name] collection
        (alternative to match_nearest, works for the multidim case, much slower)
        '''
        jets = data[jet_name]
        pfcands = data[pf_name]
        # return_combinations=True is needed to return _pfcands_unzipped,
        # which is broadcasted to duplicate the pfcands list per every jet in event
        (dr, (_, _pfcands_unzipped)) = jets.metric_table(
            pfcands,
            metric=coffea.nanoevents.methods.vector.LorentzVector.delta_r,
            return_combinations=True)
        pfcands_matched = _pfcands_unzipped[(dr < dR)]
        return pfcands_matched

    @zero_handler
    def get_matched_pfCands(self, data, match_object, dR=0.4):  # base: no extra variables
        pfCands = self.match_jet_to_pfcand(data, jet_name=match_object, pf_name="PfCands", dR=dR)
        pfCands_lead = ak.firsts(pfCands, axis=-1)
        pfCands_lead["dxysig"] = pfCands_lead.dxy / pfCands_lead.dxyError
        pfCands_lead["ip3d"] = np.sqrt(pfCands_lead.dxy**2 + pfCands_lead.dz**2)
        pfCands_lead["dxy_weight"] = ak.mean(pfCands.dxy, weight=pfCands.pt, axis=-1)
        pfCands_lead["dxysig_weight"] = ak.mean(pfCands.dxy / pfCands.dxyError, weight=pfCands.pt, axis=-1)
        return pfCands_lead

    @zero_handler
    def set_jet_dxy(self, data):  # base: jets already carry dxy
        jets = data["Jet_select"]
        # Mask jets with dxy nan (no selected pfcands matching)
        bad_jets = ak.is_none(data["Jet_lead_pfcand"].dxy, axis=-1)
        jets = ak.mask(jets, ~bad_jets)  # mask bad jets to keep correct shape
        jets["dz"] = np.abs(data["Jet_lead_pfcand"].dz)
        jets["dxy"] = np.abs(data["Jet_lead_pfcand"].dxy)
        jets["dxy_weight"] = np.abs(data["Jet_lead_pfcand"].dxy_weight)
        jets["dxysig"] = np.abs(data["Jet_lead_pfcand"].dxysig)
        jets["dxysig_weight"] = np.abs(data["Jet_lead_pfcand"].dxysig_weight)
        jets["ip3d"] = data["Jet_lead_pfcand"].ip3d
        jets = jets[~bad_jets]  # remove bad jets
        # add dxy cut
        jets = jets[jets.dxy >= self.config["jet_dxy_min"]]
        return jets

    @zero_handler
    def sum_jj(self, data):  # base: invariant mass of the two jets
        return data['Jet_select'][:, 0].add(data['Jet_select'][:, 1])

    @zero_handler
    def mt_jets(self, data):  # base: PFMET
        jet1 = data["Jet_select"][:, 0]
        jet2 = data["Jet_select"][:, 1]

        MET = data["MET"]
        one_min_cs = 1.0 - np.cos(self.delta_phi(jet1.phi, MET.phi))
        prod = 2 * jet1.pt * MET.pt
        mt_j1 = np.sqrt(prod * one_min_cs)

        one_min_cs = 1.0 - np.cos(self.delta_phi(jet2.phi, MET.phi))
        prod = 2 * jet2.pt * MET.pt
        mt_j2 = np.sqrt(prod * one_min_cs)

        return {
            "mt_jet1": mt_j1,
            "mt_jet2": mt_j2,
            "mt_sum": mt_j1 + mt_j2
        }

    @zero_handler
    def get_mt2(self, data):  # base: PFMET
        jet1 = data["Jet_select"][:, 0]
        jet2 = data["Jet_select"][:, 1]
        met = data["MET"]
        return mt2.mt2(
            jet1.mass, jet1.px, jet1.py,
            jet2.mass, jet2.px, jet2.py,
            met.px, met.py,
            0, 0
        )

    @zero_handler
    def binning_schema(self, data):  # base: 8-bin schema (currently unused by WJets)
        jets = data["Jet_select"]
        met = data["MET"].pt
        jet2_pt = jets[:, 1].pt
        mt2_ = data["mt2_j1_j2_MET"]
        bins = np.full((len(met)), np.nan)
        B1 = (jet2_pt < 50) & (met >= 250)
        B2 = (jet2_pt < 50) & (met < 250) & (mt2_ < 100)
        B3 = (jet2_pt < 50) & (met < 250) & (mt2_ >= 100)
        B4 = (jet2_pt >= 50) & (jet2_pt < 100)
        B5 = (jet2_pt >= 100)
        bins[B1] = 1
        bins[B2] = 2
        bins[B3] = 3
        bins[B4] = 4
        bins[B5] = 5
        return bins
