
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-K robustness of wet/dry scores to weight choices.

Inputs
------
--csv   : long per-WU file with columns: year,pct_wet,pct_wet_plus,pct_dry
--wu    : name for titles and file names (e.g., 01AL000)
--K     : Top-K size (default 3)
--n     : # of random weight draws (default 500)
--outdir: output directory (default .)

Outputs
-------
- PNG: weights_topk_sensitivity_<WU>.png   (4-panel figure)
- CSV: weights_topk_sensitivity_<WU>.csv   (trial metrics)

Notes
-----
- Uses the same score definitions as the manuscript:
    S_wet = w_wet * %wet + w_wet_plus * %wet_plus - w_dry * %dry
    S_dry = %dry - alpha * %wet_plus - beta * %wet
- Constrained random simplex:
    w_wet_plus >= w_wet   and   alpha >= beta
- Top-K sets are computed deterministically (stable sort).
- For “persistence heatmap”, we show, for each year, the fraction of random
  draws where that year appears in the Top-K set (wet/dry separately).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASELINE = dict(w_wet=1.0, w_wet_plus=1.5, w_dry=1.0, alpha=2.0, beta=1.0)
RANGES   = dict(w_wet=(0.5, 1.5), w_wet_plus=(1.0, 2.5), w_dry=(0.5, 1.5),
                alpha=(1.0, 3.0), beta=(0.5, 1.5))

def load_series(fn):
    df = pd.read_csv(fn).sort_values("year")
    for c in ["pct_wet","pct_wet_plus","pct_dry"]:
        df[c] = df[c].astype(float)
    return df.set_index("year")

def compute_scores(df, w):
    sw = (w["w_wet"]*df["pct_wet"] + w["w_wet_plus"]*df["pct_wet_plus"]
          - w["w_dry"]*df["pct_dry"])
    sd = (df["pct_dry"] - w["alpha"]*df["pct_wet_plus"] - w["beta"]*df["pct_wet"])
    return sw, sd

def topk_years(series, K):
    # stable: highest score first; ties broken by earlier year
    s = series.sort_values(ascending=False, kind="mergesort")
    return list(s.index[:K])

def spearman(a, b):
    from scipy.stats import spearmanr
    # Align on same index
    X = pd.DataFrame({"a": a, "b": b}).dropna()
    if X["a"].nunique() <= 1 or X["b"].nunique() <= 1:
        return 1.0
    rho, _ = spearmanr(X["a"], X["b"])
    return float(rho)

def jaccard(A, B):
    A, B = set(A), set(B)
    if not A and not B:  # not expected for Top-K, but keep safe
        return 1.0
    return len(A & B) / len(A | B)

def sample_weights(n, ranges=RANGES, baseline=BASELINE):
    rng = np.random.default_rng(42)
    keys = ["w_wet","w_wet_plus","w_dry","alpha","beta"]
    draws = []
    while len(draws) < n:
        w = {k: float(rng.uniform(*ranges[k])) for k in keys}
        if w["w_wet_plus"] + 1e-9 >= w["w_wet"] and w["alpha"] + 1e-9 >= w["beta"]:
            draws.append(w)
    return [baseline.copy()] + draws

def one_at_a_time_levels(baseline=BASELINE, ranges=RANGES, levels=(0.5,0.75,1.25,1.5)):
    outs = []
    for k, v0 in baseline.items():
        lo, hi = ranges[k]
        for m in levels:
            w = dict(baseline)
            v = min(max(v0*m, lo), hi)
            w[k] = float(v)
            # enforce constraints minimally
            if w["w_wet_plus"] < w["w_wet"]:
                if k == "w_wet_plus": w["w_wet"] = w["w_wet_plus"]
                else:                 w["w_wet_plus"] = w["w_wet"]
            if w["alpha"] < w["beta"]:
                if k == "alpha": w["beta"] = w["alpha"]
                else:            w["alpha"] = w["beta"]
            outs.append((k, w))
    return outs

def analyze(fn, wu, K=3, n=500, outdir="."):
    df = load_series(fn)
    years = df.index.to_numpy()

    # Baseline
    sw0, sd0 = compute_scores(df, BASELINE)
    topk_wet0 = topk_years(sw0, K)
    topk_dry0 = topk_years(sd0, K)

    # Random draws
    trials = []
    weights = sample_weights(n, RANGES, BASELINE)
    # For persistence heatmap
    wet_hits = pd.Series(0.0, index=years)
    dry_hits = pd.Series(0.0, index=years)

    for i, w in enumerate(weights):
        sw, sd = compute_scores(df, w)
        rho_wet = spearman(sw0, sw)
        rho_dry = spearman(sd0, sd)

        tk_w = topk_years(sw, K)
        tk_d = topk_years(sd, K)
        jac_w = jaccard(topk_wet0, tk_w)
        jac_d = jaccard(topk_dry0, tk_d)

        # Update hit counters (exclude baseline from frequency if you prefer)
        if i > 0:
            wet_hits[tk_w] += 1
            dry_hits[tk_d] += 1

        trials.append({
            "trial": i, **{f"w_{k}": v for k,v in w.items()},
            "rho_wet": rho_wet, "rho_dry": rho_dry,
            "jaccard_wet_topK": jac_w, "jaccard_dry_topK": jac_d
        })

    df_trials = pd.DataFrame(trials)
    df_trials.to_csv(f"{outdir}/weights_topk_sensitivity_{wu}.csv", index=False)

    # OAT tornado
    oat = one_at_a_time_levels()
    rows = []
    for name, w in oat:
        sw, sd = compute_scores(df, w)
        rows.append({
            "param": name,
            "rho_wet": spearman(sw0, sw),
            "rho_dry": spearman(sd0, sd)
        })
    df_oat = pd.DataFrame(rows)
    agg = (df_oat.groupby("param")
           .agg(rho_wet_min=("rho_wet","min"), rho_wet_max=("rho_wet","max"),
                rho_dry_min=("rho_dry","min"), rho_dry_max=("rho_dry","max"))
           .reset_index())

    # Persistence frequency (% of random draws where each year is in Top-K)
    wet_freq = (wet_hits / (len(weights)-1)).reindex(years)
    dry_freq = (dry_hits / (len(weights)-1)).reindex(years)

    # --------- Plot ----------
    import matplotlib as mpl
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    # (A) Tornado: Spearman ρ ranges (OAT)
    axA = fig.add_subplot(gs[0,0])
    y = np.arange(len(agg))
    axA.hlines(y, agg["rho_wet_min"], agg["rho_wet_max"], color="#1f77b4", lw=6, label="Wet (ρ)")
    axA.hlines(y, agg["rho_dry_min"], agg["rho_dry_max"], color="#d62728", lw=6, label="Dry (ρ)")
    axA.set_yticks(y)
    axA.set_yticklabels(agg["param"])
    axA.set_xlim(0.8, 1.01)
    axA.set_xlabel("Spearman rank correlation vs. baseline")
    axA.set_title(f"(A) Tornado (OAT) — rank stability")
    axA.legend(loc="lower right")

    # (B) Boxplots: ρ across random draws
    axB = fig.add_subplot(gs[0,1])
    bp = axB.boxplot([df_trials["rho_wet"].values, df_trials["rho_dry"].values],
                     labels=["Wet ρ","Dry ρ"], patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, color in zip(axB.artists, ["#1f77b4","#d62728"]):
        patch.set_facecolor(color); patch.set_alpha(0.25)
    axB.set_ylim(0.8, 1.01)
    axB.set_title("(B) Rank stability across weight simplex")

    # (C) Boxplots: Top-K Jaccard vs baseline
    axC = fig.add_subplot(gs[1,0])
    bp2 = axC.boxplot([df_trials["jaccard_wet_topK"].values,
                       df_trials["jaccard_dry_topK"].values],
                      labels=[f"Wet Top-{K}", f"Dry Top-{K}"], patch_artist=True,
                      medianprops=dict(color="black"))
    for patch, color in zip(axC.artists, ["#1f77b4","#d62728"]):
        patch.set_facecolor(color); patch.set_alpha(0.25)
    axC.set_ylim(0.0, 1.01)
    axC.set_title(f"(C) Overlap with baseline Top-{K} sets (Jaccard)")

    # (D) Persistence heatmap: fraction of draws where each year appears in Top-K
    axD = fig.add_subplot(gs[1,1])
    Y = pd.DataFrame({"Wet": wet_freq.values, "Dry": dry_freq.values}, index=years)
    im = axD.imshow(Y.values.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axD.set_yticks([0,1]); axD.set_yticklabels(["Wet","Dry"])
    axD.set_xticks(np.arange(len(years))); axD.set_xticklabels(years, rotation=90, fontsize=8)
    axD.set_title(f"(D) Top-{K} membership persistence by year")
    cbar = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of random draws")

    fig.suptitle(f"Weights sensitivity (Top-{K}) — {wu}", y=0.99, fontsize=13)
    fig.savefig(f"{outdir}/weights_topk_sensitivity_{wu}.png", dpi=220, bbox_inches="tight")
    print(f"[OK] Figure: {outdir}/weights_topk_sensitivity_{wu}.png")
    print(f"[OK] Trials : {outdir}/weights_topk_sensitivity_{wu}.csv")

def main():
    """ 
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="year,pct_wet,pct_wet_plus,pct_dry")
    ap.add_argument("--wu", required=True, help="WU code for titles/filenames")
    ap.add_argument("--K", type=int, default=3, help="Top-K size (default 3)")
    ap.add_argument("--n", type=int, default=500, help="# random draws (default 500)")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()
    """

    incsv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\gisdata\\data\\01AL000-long-series.csv" 
    wu =  "01AL000" 
    outdir =  "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\sensitivity" 
    q = 0.917 
    n = 500

    analyze(incsv, wu, K=3, n=n, outdir=outdir)

if __name__ == "__main__":
    main()
