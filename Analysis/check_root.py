import ROOT
import numpy as np

f = ROOT.TFile.Open("/pnfs/desy.de/cms/tier2/store/user/svashish/LLSTauProduction_RUN3/SUS-RunIIISummer2024_EXONANO_stau500_lsp1_ctau1mm_v1/SUS-Run3Summer2024-stau500_lsp1_ctau1mm/NANOAOD/260918_162855/0000/ExoNanoAOD_1.root")

tree = f.Get("Events")

tag_threshold = 0.05
dxy_min = 2.0  # Try 0.0, 0.2, and 2.0

n_exactly_2 = 0
n_at_least_2 = 0

for i in range(tree.GetEntries()):
    tree.GetEntry(i)

    score1 = np.asarray(tree.JetCHS_disTauTag_score1)
    dxy = np.abs(np.asarray(tree.JetCHS_dxy))

    # Match processor selection: tagger, then dxy
    mask_tag = score1 >= tag_threshold
    mask_dxy = dxy >= dxy_min
    mask_final = mask_tag & mask_dxy

    selected_indices = np.where(mask_final)[0]
    n_selected = len(selected_indices)

    if n_selected == 2:
        n_exactly_2 += 1

    if n_selected >= 2:
        n_at_least_2 += 1

    if i < 20:
        print(
            f"Event {i:4d}: "
            f"original jets = {len(score1):3d}, "
            f"after tagger = {np.sum(mask_tag):3d}, "
            f"after dxy = {n_selected:3d}, "
            f"exactly 2 = {n_selected == 2}"
        )

print("\nTotal events:", tree.GetEntries())
print("Exactly 2 selected jets:", n_exactly_2)
print("At least 2 selected jets:", n_at_least_2)