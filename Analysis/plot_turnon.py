
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.stats import beta

# --------------------------------------------------
# ROOT files
# --------------------------------------------------

D_DEN = "muon_trigger_study/hists/" # only isomu24 trigger
D_NUM = "muon_trigger_withMET/hists/" # with PFMET120 trigger

DEN = D_DEN + "Cut_012_atleast_two_loose_jets_final_PFMET_pt.root"
NUM = D_NUM + "Cut_013_atleast_two_loose_jets_final_PFMET_pt.root"

groups = {
    "Signal (all stau samples)": [
        f"SMS-TStauStau_MStau-{mass}_ctau-{ctau}mm_mLSP-1"
        for mass in (400, 500, 600)
        for ctau in (1, 10, 100)
    ],

    "W+jets": [
        f"WtoLNu-4Jets_Bin-{i}J"
        for i in (1, 2, 3, 4)
    ],
}


def get(path, datasets):

    f = uproot.open(path)

    vals = None
    var = None
    edges = None

    for ds in datasets:
        hist_path = f"{ds}/nominal/hist"
        h = f[hist_path]

        h_vals = h.values(flow=False)
        h_var = h.variances(flow=False)
        h_edges = h.axis().edges()

        if h_var is None:
            raise ValueError(f"No sumw2 variances found for {hist_path}")

        if vals is None:
            vals = h_vals.copy()
            var = h_var.copy()
            edges = h_edges.copy()
        else:
            if not np.allclose(edges, h_edges):
                raise ValueError("Inconsistent histogram binning!")

            vals += h_vals
            var += h_var

    f.close()
    return vals, var, edges


# --------------------------------------------------
# Calculate efficiency for each group
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for label, datasets in groups.items():
    # n: denominator
    # k: numerator 
    
    n, n_var, edges = get(DEN, datasets)
    k, k_var, edges_num = get(NUM, datasets)

    if not np.allclose(edges, edges_num):
        raise ValueError("Numerator and denominator binning differ!")

    eff = np.divide(
        k, n,
        out=np.zeros_like(k, dtype=float),
        where=n > 0
    )

    n_eff = np.divide(
        n**2, n_var,
        out=np.zeros_like(n, dtype=float),
        where=n_var > 0
    )

    k_eff = eff * n_eff

    lo = np.zeros_like(eff)
    hi = np.ones_like(eff)

    valid = (n_eff > 0) & (k_eff >= 0) & (k_eff <= n_eff)

    lo[valid] = beta.ppf(
        0.16,
        k_eff[valid],
        n_eff[valid] - k_eff[valid] + 1
    )

    hi[valid] = beta.ppf(
        0.84,
        k_eff[valid] + 1,
        n_eff[valid] - k_eff[valid]
    )

    lo[valid & (k_eff == 0)] = 0
    hi[valid & (k_eff == n_eff)] = 1

    x = 0.5 * (edges[1:] + edges[:-1])

    plt.errorbar(
        x,
        eff,
        yerr=[
            np.clip(eff - lo, 0, None),
            np.clip(hi - eff, 0, None)
        ],
        fmt="o-",
        capsize=3,
        label=label
    )

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.axhline(1, linestyle="--", color="gray")
plt.axhline(0.85, linestyle="--", color="red", label="85% efficiency")

plt.xlabel("PF MET [GeV]")
plt.ylabel("Trigger efficiency")
plt.title("PFMET120_PFMHT120_IDTight Trigger Turn-on")

plt.ylim(0, 1.1)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("turnon.png", dpi=300)
plt.show()
