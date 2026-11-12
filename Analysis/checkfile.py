
import ROOT
import numpy as np

# f = ROOT.TFile.Open("/pnfs/desy.de/cms/tier2/store/user/svashish/LLSTauProduction_RUN3/SUS-RunIIISummer2024_EXONANO_stau400_lsp1_ctau0p1mm_v1/SUS-Run3Summer2024-stau400_lsp1_ctau0p1mm/NANOAOD/261001_161008/0000/ExoNanoAOD_1.root")
f = ROOT.TFile.Open("/pnfs/desy.de/cms/tier2/store/user/svashish/LLSTauProduction_RUN3/SUS-RunIIISummer2024_EXONANO_stau500_lsp1_ctau1zmm_v1/SUS-Run3Summer2024-stau500_lsp1_ctau1mm/NANOAOD/260918_162855/0000/ExoNanoAOD_1.root")

tree = f.Get("Events")

pt_min = 30
eta_min = -2.1
eta_max = 2.1
dxy_min = 0.2
tag_threshold = 0.05

for i in range(min(10, tree.GetEntries())):

    tree.GetEntry(i)

    pt = np.array(tree.JetCHS_pt)
    eta = np.array(tree.JetCHS_eta)
    dxy = np.abs(np.array(tree.JetCHS_dxy))
    dxy_err = np.abs(np.array(tree.JetCHS_dxyerror))

    score1 = np.array(tree.JetCHS_disTauTag_score1)
    n_const = np.array(tree.JetCHS_nConstituents)

    npfcand = np.array(tree.nPFCand)
    PFCand_pdgId = np.array(tree.PFCand_pdgId)
    PFCand_mass = np.array(tree.PFCand_mass)
    PFCand_phi = np.array(tree.PFCand_phi)
    PFCand_pt = np.array(tree.PFCand_pt)
    # Selection
    mask_pt = pt > pt_min
    mask_eta = (eta > eta_min) & (eta < eta_max)
    mask_tag = score1 >= tag_threshold
    mask_dxy = dxy >= dxy_min

    mask_final = mask_pt & mask_eta & mask_tag & mask_dxy
    selected_indices = np.where(mask_final)[0]

    print(f"\n{'='*75}")
    print(f"EVENT {i}")
    print(f"Number of PFCandidates: {npfcand}")
    print(f"pgdID: {PFCand_pdgId}")
    print(f"mass: {PFCand_mass}")
    print(f"pT: {PFCand_pt}")
    print(f"Original CHS jets: {len(pt)}")
    print(f"After pT cut:       {np.sum(mask_pt)}")
    print(f"After eta cut:      {np.sum(mask_pt & mask_eta)}")
    print(f"After tagger cut:   {np.sum(mask_pt & mask_eta & mask_tag)}")
    print(f"After dxy cut:      {len(selected_indices)}")

    print("\nSelected jet properties:")
    print(
        f"{'Jet':>4} | {'pT':>8} | {'eta':>7} | {'dxy':>9} | "
        f"{'dxy significance':>16} | {'score1':>8} | {'nConst':>7}"
    )
    print("-" * 85)

    for j in selected_indices:
        dxysig = dxy[j] / dxy_err[j] if dxy_err[j] > 0 else np.nan

        print(
            f"{j:4d} | {pt[j]:8.2f} | {eta[j]:7.2f} | "
            f"{dxy[j]:9.4f} | {dxysig:16.2f} | "
            f"{score1[j]:8.4f} | {n_const[j]:7d}"
        )

    print("\nFinal selected jet indices:", selected_indices)
    print("Exactly two:", len(selected_indices) == 2)