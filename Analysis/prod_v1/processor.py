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
    """Signal-region processor. Shared functions come from BaseProcessor."""
    def zero_handler(func):
        def _function(self, data, *args, **kwargs):
            if len(data) > 0: return func(self, data, *args, **kwargs)
            else: return ak.Array([])
        return _function

    def __init__(self, config, eventdir):
        # Base __init__ sets histogram_format and calls the pepper init
        super().__init__(config, eventdir)

        if "pileup_reweighting" not in config:
            logger.error("No pileup reweigthing specified")
        if "DY_ZptLO_weights" not in config:
            logger.error("No DY k-factor specified")
        if "MET_trigger_sfs" not in config:
            logger.error("No MET trigger scale factors specified")

        if "DY_jet_reweight" not in config or len(config["DY_jet_reweight"]) == 0:
            logger.error("No DY jet-binned sample stitching is specified")
        if "W_jet_reweight" not in config or len(config["W_jet_reweight"]) == 0:
            logger.error("No WjetsToLNu jet-binned sample stitching is specified")

        if "jet_fake_rate" not in config and len(config["jet_fake_rate"]) == 0:
            self.predict_jet_fakes = False
            logger.warning("No jet fake rate specified")
        else:
            self.predict_jet_fakes = config["predict_yield"]

        if "jet_veto_map" not in config:
            logger.error("No jet veto map is specified")

        if ("jet_uncertainty" not in config and config["compute_systematics"]):
            logger.warning("No jet uncertainty specified")

        if ("jet_resolution" not in config or "jet_ressf" not in config):
            logger.warning("No jet resolution or no jet resolution scale "
                           "factor specified. This is necessary for "
                           "smearing, even if not computing systematics")
        if "jet_correction_mc" not in config and (
                ("jet_resolution" in config and "jet_ressf" in config) or
                ("reapply_jec" in config and config["reapply_jec"])):
            raise pepper.config.ConfigError(
                "Need jet_correction_mc for propagating jet "
                "smearing/variation to MET or because reapply_jec is true")
        if ("jet_correction_data" not in config and "reapply_jec" in config
                and config["reapply_jec"]):
            raise pepper.config.ConfigError(
                "Need jet_correction_data because reapply_jec is true")

        if config["propagate_eff_factors"] and "signal_eff_factors" in config:
            self.propagate_eff_factors = True
        else:
            self.propagate_eff_factors = False

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)

        if self.config["stau_properties_study"]:
            assert dsname.startswith("SMS-TStauStau")
            self.signal_process_study(selector, dsname, is_mc)
            return

        # Triggers, MET filters, PV (shared with WJets, see BaseProcessor)
        self.add_trigger(selector, dsname, is_mc)
        self.add_quality_cuts(selector, is_mc)

        # lepton selection
        logger.debug("Running lepton veto")
        selector.set_column("Muon", self.select_muons)
        selector.set_column("Electron", self.select_electrons)
        selector.add_cut("muon_veto", self.muon_veto)
        selector.add_cut("elec_veto", self.elec_veto)

        selector.set_column("Lepton", lambda data: ak.Array([[]] * len(data)))
        selector.set_column("Jet_select", self.jet_basis)

        if self.config["dxy_cut_study"]:
            self.dxy_cut_study(selector, dsname, is_mc)
            return

        selector.set_column("Jet_select", self.getloose_jets)
        selector.set_column("PfCands", self.pfcand_valid)
        # we never use this matching, run3 jets are already matched to pf candidate
        selector.set_column("Jet_lead_pfcand", partial(self.get_matched_pfCands, match_object="Jet_select", dR=0.4))
        selector.set_column("Jet_select", self.set_jet_dxy)
        selector.add_cut("two_loose_jets", self.has_two_jets)

        selector.set_column("sum_jj", self.sum_jj)
        selector.set_multiple_columns(self.missing_energy) # sets HT_valid, HT_miss_valid, HT and HT_miss (all defined in base processor)
        selector.set_multiple_columns(self.mt_jets)
        selector.set_column("dphi_jet1_jet2", self.dphi_jet1_jet2)
        selector.set_column("dr_jet1_jet2", self.dr_jet1_jet2)

        selector.add_cut("dphi_min_cut", self.dphi_min_cut)
        selector.set_column("mt2_j1_j2_MET", self.get_mt2)
        selector.set_column("binning_schema", self.binning_schema)

        # Tagger part for calculating scale factors
        # Scale factors should be calculated -
        # before cuts on the number of the jets
        selector.set_multiple_columns(self.set_njets_pass)
        if self.config["compute_systematics"] and is_mc and self.propagate_eff_factors:
            selector.add_cut("signal_eff_sfs", partial(self.signal_eff_unc, dsname=dsname))
        if self.config["predict_yield"] and not is_mc:
            selector.set_multiple_columns(partial(self.predict_yield, weight=selector.systematics["weight"]))

        # Dummy basemark to have cut where regions were not categories yet
        selector.set_multiple_columns(partial(self.categories_bins))
        selector.set_cat("control_region", {"RT0", "RT1", "RT2"})  # setting categories, removed the INCLUSIVE CATEGORY: pepper error: needs an event to fit into exactly 1 category
        
        # selector.add_cut("two_loose_jets_final", lambda data: ak.ones_like(data.event, dtype=bool))
        # selector.add_cut("two_loose_jets_final", lambda data: ak.Array(np.ones(len(data))))
        # selector.add_cut("ht_cut", self.ht_cut)


    # ------------------------------------------------------------------
    # Signal-only selection pieces
    # ------------------------------------------------------------------
    def dxy_cut_study(self, selector, dsname, is_mc):
        selector.set_column("Jet_select", self.gettight_jets)
        selector.set_column("Jet_tau", partial(self.jet_tau, is_mc=is_mc))

        selector.add_cut("cut_separator1", lambda data: np.ones(len(data)))

        selector.add_cut("one_tight", lambda data: ak.num(data["Jet_select"]) >= 1)
        # Jet_select was not redefined yet
        selector.add_cut("cut_separato2", lambda data: np.ones(len(data)))

        selector.set_column("PfCands", self.pfcand_valid)
        selector.set_column("Jet_select", self.set_jet_dxy)
        selector.set_column("Jet_tau", partial(self.jet_tau, is_mc=is_mc))

        selector.add_cut("after_define_dxy", lambda data: np.ones(len(data)))
        selector.add_cut("two_tight", lambda data: ak.num(data["Jet_select"]) >= 2)

    ### Signal process study ---->

    def signal_process_study(self, selector, dsname, is_mc):
        # Define lifetime of GenVisTau:
        selector.set_column("GenVisTau", self.gen_vis_tau)
        selector.set_column("Jet_select", self.jet_selection)
        # match GenVisTau to Jet_select
        selector.set_column("Jet_select", partial(self.jet_tauvis_match, is_mc=is_mc))
        selector.set_column("Jet_select", self.set_jet_dxy)
        selector.add_cut("checkpoint", lambda data: np.ones(len(data)))
        selector.set_column("jet_score", self.jet_lead_score)

    def gen_vis_tau(self, data):
        tau = data.GenVisTau
        # define trevel distance and transverse travel distance of GenVisTau
        genpart = tau.parent
        assert ak.all(abs(genpart.pdgId) == 15)
        # calculate travel distance and transverse travel distance wrt. to the mother vertex
        tau["travel"] = np.sqrt(genpart.vx**2 + genpart.vy**2 + genpart.vz**2)
        tau["tr_travel"] = np.sqrt(genpart.vx**2 + genpart.vy**2)
        return tau

    @zero_handler
    def jet_tauvis_match(self, data, is_mc):
        jets = data["Jet_select"]
        if not is_mc:
            return jets

        gvt = data.GenVisTau
        tau_vis = gvt[
            (gvt.pt > 30)
            & (abs(gvt.eta) < 2.4)
            & (gvt.parent.hasFlags(["fromHardProcess"]))
        ]

        # All jet-tau pairs, shaped [event][jet][tau]
        pairs = ak.cartesian({"jet": jets, "tau": tau_vis}, axis=1, nested=True)

        deta = pairs.jet.eta - pairs.tau.eta
        dphi = pairs.jet.phi - pairs.tau.phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dr = np.sqrt(deta**2 + dphi**2)

        # Index of the closest tau for each jet (None if the event has no tau)
        idx = ak.argmin(dr, axis=-1, keepdims=True)
        closest_tau = ak.firsts(pairs.tau[idx], axis=-1)   # [event][jet], None if no tau
        min_dr = ak.firsts(dr[idx], axis=-1)

        matched = ak.fill_none(min_dr < 0.3, False)

        # Add travel distance of the matched tau to each jet
        jets = ak.with_field(jets, ak.fill_none(closest_tau.travel, np.nan), "travel")
        jets = ak.with_field(jets, ak.fill_none(closest_tau.tr_travel, np.nan), "tr_travel")

        return jets[matched]

    def categories_bins(self, data):
        if len(data) == 0:
            return {
                "RT0": ak.Array([]),
                "RT1": ak.Array([]),
                "RT2": ak.Array([]) 
                }
                # "INC": ak.Array([])}
        jets = data["Jet_select"]
        jets = jets[(jets.disTauTag_score1 >= self.config["tight_thr"])]
        n_tight = ak.num(jets)
        RT0 = (n_tight == 0)
        RT1 = (n_tight == 1)
        RT2 = (n_tight == 2)
        return {"RT0": RT0, "RT1": RT1, "RT2": RT2} # "INC": np.ones(len(data))

    ### Signal process study end <---

    # ------------------------------------------------------------------
    # Lepton vetoes (signal region vetoes leptons; WJets selects a muon)
    # ------------------------------------------------------------------
    @zero_handler
    def muon_veto(self, data):
        return ak.num(data["Muon"]) == 0

    @zero_handler
    def elec_veto(self, data):
        return ak.num(data["Electron"]) == 0

    # ------------------------------------------------------------------
    # Signal-only helpers
    # ------------------------------------------------------------------
    @zero_handler
    def jet_basis(self, data):
        jets = data["JetCHS"]
        n_before = ak.num(jets, axis=1)
        logger.info(
            f"Selecting CHS jets: "
            f"exactly 2 = {ak.sum(n_before == 2)}, "
            f">=2 = {ak.sum(n_before >= 2)}"
        )
        return jets

    @zero_handler
    def ht_cut(self, data):
        return data["HT_valid"] > 150

    @zero_handler
    def jet_veto(self, data, jet_name):
        # mask events with at least one jet in veto map (might be too tight)
        jets = data[jet_name]
        config = self.config["jet_veto_map"]
        mask_per_jet = config(eta=jets.eta, phi=jets.phi)
        mask = ak.any(mask_per_jet, axis=-1)
        return ~mask

    @zero_handler
    def has_jets(self, data):
        return ak.num(data["Jet_select"]) > 0

    @zero_handler
    def has_small_dxy(self, data):
        jets = data["Jet_select"]
        dxy1 = (jets.dxy[:, 0] < 0.4)
        dxy2 = (jets.dxy[:, 1] < 0.4)
        return (dxy1 | dxy2)

    @zero_handler
    def skim_jets(self, data):
        jets = data["Jet_select"]
        file_name = os.path.basename(data.metadata["filename"])
        dataset_name = data.metadata["dataset"]
        prefix = f"evn_{data.metadata['entrystart']}_{data.metadata['entrystop']}_"
        path = self.config["skim_path"]
        path = f"{path}/{dataset_name}"
        print(path)
        print(f"{path}/" + prefix + file_name + ".parquet")
        if not os.path.exists(path):
            os.makedirs(path)
        ak.to_parquet(jets, f"{path}/" + prefix + file_name + ".parquet")
        return np.ones(len(data))

    @zero_handler
    def MET_trigger_sfs(self, data, met_name="MET"):
        met_pt = data[met_name].pt
        scale_factors = self.config["MET_trigger_sfs"]
        met_thr = self.config["MET_trigger_plateau"]
        sfs = scale_factors(pt=met_pt)
        systematics = {}
        if self.config["compute_systematics"]:
            sfs_up = scale_factors(variation="up", pt=met_pt)
            sfs_down = scale_factors(variation="down", pt=met_pt)
            # ratio of the up/down to nominal and for 120-250 GeV - 2x stat. unc. of the MET_trigger_sfs is used, for > 250 - 1x stat. unc.
            systematics["MET_trigger_sfs_up"] = ak.where(met_pt > met_thr, sfs_up / sfs, 2 * sfs_up / sfs)
            systematics["MET_trigger_sfs_down"] = ak.where(met_pt > met_thr, sfs_down / sfs, 2 * sfs_down / sfs)
        return sfs, systematics

    @zero_handler
    def jet_lead(self, data):
        # select only first two pt leading jets
        return data["Jet_select"][:, :2]

    @zero_handler
    def jet_passed(self, data):
        jets = data["Jet_select"]
        jets = jets[jets.disTauTag_score1 >= self.config["tight_thr"]]
        return jets

    @zero_handler
    def jet_tau(self, data, is_mc):
        jets = data["Jet_select"]
        if not is_mc:
            return jets

        gvt = data.GenVisTau
        tau_vis = gvt[
            (gvt.pt > 30)
            & (abs(gvt.eta) < 2.4)
            & (gvt.parent.hasFlags(["fromHardProcess"]))
        ]

        # All jet-tau pairs per event, nested so the result is [event][jet][tau]
        pairs = ak.cartesian({"jet": jets, "tau": tau_vis}, axis=1, nested=True)

        deta = pairs.jet.eta - pairs.tau.eta
        dphi = pairs.jet.phi - pairs.tau.phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap into [-pi, pi)
        dr = np.sqrt(deta**2 + dphi**2)

        # Closest tau for each jet; events with no tau give None, so fill with a large value
        min_dr = ak.fill_none(ak.min(dr, axis=-1), 999.0)

        return jets[min_dr < 0.3]

    @zero_handler
    def jet_taupass(self, data, is_mc):
        jets = data["Jet_pass"]
        if not is_mc:
            return jets
        tau_vis = data.GenVisTau[((data.GenVisTau.pt > 30) & (abs(data.GenVisTau.eta) < 2.4)
                                  & (data.GenVisTau.parent.hasFlags(["fromHardProcess"])))
                                 ]
        matches_h, dRlist = jets.nearest(tau_vis, return_metric=True, threshold=0.3)
        tau_jet = jets[~ak.is_none(matches_h, axis=1)]
        return tau_jet

    @zero_handler
    def jet_taupass2(self, data, is_mc):
        jets = data["Jet_pass"]
        if not is_mc:
            return jets
        tau_vis = data.GenVisTau[((data.GenVisTau.pt > 30) & (abs(data.GenVisTau.eta) < 2.4)
                                  & (data.GenVisTau.parent.hasFlags(["fromHardProcess"])))
                                 ]
        matches_h, dRlist = jets.nearest(tau_vis, return_metric=True, threshold=0.3)
        two_jets_pass = (ak.num(jets) == 2)
        tau_jet = jets[(~ak.is_none(matches_h, axis=1)) & two_jets_pass]
        return tau_jet

    @zero_handler
    def jet_lead_score(self, data):
        # select only first two biggest score jets
        jets = data["Jet_select"]
        sort_idx = ak.argsort(jets.disTauTag_score1, axis=-1, ascending=False)
        jets = jets[sort_idx]
        return jets[:, :2]

    @zero_handler
    def set_njets_pass_finebin(self, data):
        jets_score = data["Jet_select"].disTauTag_score1
        n_pass = []
        for score in self.config["score_pass_finebin"]:
            jets_pass = jets_score[(jets_score >= score)]
            passed = ak.num(jets_pass, axis=1)
            n_pass.append(passed)
        n_pass = ak.from_regular(
            np.stack(n_pass, axis=1), axis=-1)
        return {
            "n_pass_finebin": n_pass,
            "n_pass_score_bin_finebin": ak.local_index(n_pass, axis=1),
        }

    @zero_handler
    def signal_eff_unc(self, data, dsname):
        weights = np.ones(len(data))
        if not dsname.startswith("SMS-TStauStau"):
            return weights
        else:
            factor = self.config["signal_eff_factors"]
            systematics = {}
            RT1 = data["tight_bin1"]
            RT2 = data["tight_bin2"]
            weights[RT1] = factor["central"]
            weights[RT2] = factor["central"]**2
            if self.config["compute_systematics"]:
                weights_up = np.ones(len(data))
                weights_up[RT1] = factor["up"]
                weights_up[RT2] = factor["up"]**2
                weights_down = np.ones(len(data))
                weights_down[RT1] = factor["down"]
                weights_down[RT2] = factor["down"]**2
                systematics["signal_eff_unc_up"] = weights_up / weights
                systematics["signal_eff_unc_down"] = weights_down / weights
            return weights, systematics
