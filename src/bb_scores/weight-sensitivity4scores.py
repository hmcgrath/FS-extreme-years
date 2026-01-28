
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weights-sensitivity analysis for wet/dry scores:
- Computes robustness of year rankings and GEV-selected extremes to score weights.
- Produces a tornado-style OAT panel and boxplots for rank/Jaccard stability.

INPUT CSV schema (one WU):
    year,pct_wet,pct_wet_plus,pct_dry

Author: Heather McGrath (prepared with M365 Copilot)
Date: 2026-01-20
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    from scipy.stats import genextreme as gev
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    # Fallback: use empirical quantile if SciPy not available.


# ----------------------------
# Configuration (defaults)
# ----------------------------

BASELINE_WEIGHTS = dict(
    w_wet=1.0,
    w_wet_plus=1.5,
    w_dry=1.0,
    alpha=2.0,
    beta=1.0
)

# Reasonable ranges around baseline for the simplex sampling (uniform).
WEIGHT_RANGES = dict(
    w_wet=(0.5, 1.5),
    w_wet_plus=(1.0, 2.5),  # must be >= w_wet (enforced)
    w_dry=(0.5, 1.5),
    alpha=(1.0, 3.0),       # must be >= beta (enforced)
    beta=(0.5, 1.5)
)

GEV_Q = 0.917  # quantile used as RL cutoff (consistent with manuscript)
N_RANDOM = 500  # number of random weight draws for boxplots


# ----------------------------
# Utilities
# ----------------------------

def _ensure_frac(x: pd.Series) -> pd.Series:
    """Normalize to fraction [0,1] if values look like percentages."""
    x = x.astype(float)
    if x.max() > 1.5:  # likely 0–100
        return x / 100.0
    return x

def compute_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Compute wet/dry scores given weights.
    S_wet  = w_wet * %wet + w_wet_plus * %wet+  - w_dry * %dry
    S_dry  = %dry - alpha * %wet+ - beta * %wet
    """
    w = weights
    out = df.copy()
    out["S_wet"] = (
        w["w_wet"] * out["pct_wet"] +
        w["w_wet_plus"] * out["pct_wet_plus"] -
        w["w_dry"] * out["pct_dry"]
    )
    out["S_dry"] = (
        out["pct_dry"] -
        w["alpha"] * out["pct_wet_plus"] -
        w["beta"] * out["pct_wet"]
    )
    return out

def fit_rl(series: pd.Series, q: float) -> float:
    """
    Fit a stationary GEV to the input series and return the quantile at q.
    Fallback to empirical quantile if SciPy is unavailable or fit fails.
    """
    y = np.asarray(series.values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return float(np.quantile(y, q))

    if SCIPY_OK:
        try:
            # SciPy's shape parameter c == -xi (paper uses xi)
            c_hat, loc_hat, scale_hat = gev.fit(y)
            rl = float(gev.ppf(q, c=c_hat, loc=loc_hat, scale=scale_hat))
            if np.isfinite(rl):
                return rl
        except Exception:
            pass
    # Empirical fallback
    return float(np.quantile(y, q))

def select_extreme_years(series: pd.Series, rl: float) -> List[int]:
    """Years where series >= return level."""
    mask = series >= rl
    return list(series.index[mask])

def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    """
    Spearman ρ between two series indexed by year.
    If constant series, return 1.0 (identical order).
    """
    from scipy.stats import spearmanr
    merged = pd.DataFrame({"a": a, "b": b}).dropna()
    if merged["a"].nunique() <= 1 or merged["b"].nunique() <= 1:
        return 1.0
    rho, _ = spearmanr(merged["a"], merged["b"])
    return float(rho)

def decile_stability(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """
    % of years staying in the same decile and within ±1 decile.
    """
    def deciles(x):
        # ranks ascending; ties get average rank
        r = x.rank(method="average")
        d = pd.qcut(r, 10, labels=False, duplicates="drop")
        return pd.Series(d, index=x.index)

    da = deciles(a)
    db = deciles(b).reindex_like(da)
    same = (da == db).mean()
    within1 = ((da - db).abs() <= 1).mean()
    return float(same), float(within1)

def jaccard(a: List[int], b: List[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def sample_weights(n: int,
                   ranges: Dict[str, Tuple[float, float]],
                   baseline: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Random uniform sampling with constraints:
      w_wet_plus >= w_wet
      alpha >= beta
    Rejects until n valid draws are collected.
    """
    rng = np.random.default_rng(42)
    keys = ["w_wet", "w_wet_plus", "w_dry", "alpha", "beta"]
    out = []
    while len(out) < n:
        w = {}
        for k in keys:
            lo, hi = ranges[k]
            w[k] = float(rng.uniform(lo, hi))
        if w["w_wet_plus"] + 1e-9 >= w["w_wet"] and w["alpha"] + 1e-9 >= w["beta"]:
            out.append(w)
    # Include baseline first for reference (unique)
    return [baseline.copy()] + out

def one_at_a_time_levels(baseline: Dict[str, float],
                         ranges: Dict[str, Tuple[float, float]],
                         levels=(0.5, 0.75, 1.25, 1.5)) -> List[Tuple[str, Dict[str, float]]]:
    """
    Generate OAT perturbations: multiply one weight by each level, clip to allowed range,
    and enforce constraints afterward by minimal adjustments.
    """
    outs = []
    for k, v0 in baseline.items():
        lo, hi = ranges[k]
        for m in levels:
            v = np.clip(v0 * m, lo, hi)
            w = dict(baseline)
            w[k] = float(v)
            # enforce constraints
            if w["w_wet_plus"] < w["w_wet"]:
                if k == "w_wet_plus":
                    w["w_wet"] = w["w_wet_plus"]
                else:
                    w["w_wet_plus"] = w["w_wet"]
            if w["alpha"] < w["beta"]:
                if k == "alpha":
                    w["beta"] = w["alpha"]
                else:
                    w["alpha"] = w["beta"]
            outs.append((k, w))
    return outs


# ----------------------------
# Core analysis
# ----------------------------

def analyze(df_in: pd.DataFrame,
            out_png: str,
            out_csv: str,
            wu_name: str,
            q: float = GEV_Q,
            n_random: int = N_RANDOM):

    # Normalize and index by year
    df = df_in.copy()
    for col in ["pct_wet", "pct_wet_plus", "pct_dry"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = _ensure_frac(df[col])
    df = df.sort_values("year").set_index("year")

    # Baseline scores and selections
    base = compute_scores(df, BASELINE_WEIGHTS)
    rl_wet = fit_rl(base["S_wet"], q)
    rl_dry = fit_rl(base["S_dry"], q)
    base_wet_years = select_extreme_years(base["S_wet"], rl_wet)
    base_dry_years = select_extreme_years(base["S_dry"], rl_dry)

    # Prepare storage for random samples
    trials = []

    # Random simplex sampling (includes baseline as first element)
    for i, w in enumerate(sample_weights(n_random, WEIGHT_RANGES, BASELINE_WEIGHTS)):
        sc = compute_scores(df, w)
        rho_wet = spearman_rank_corr(base["S_wet"], sc["S_wet"])
        rho_dry = spearman_rank_corr(base["S_dry"], sc["S_dry"])
        same_wet, within1_wet = decile_stability(base["S_wet"], sc["S_wet"])
        same_dry, within1_dry = decile_stability(base["S_dry"], sc["S_dry"])

        rl_w = fit_rl(sc["S_wet"], q)
        rl_d = fit_rl(sc["S_dry"], q)
        wet_years = select_extreme_years(sc["S_wet"], rl_w)
        dry_years = select_extreme_years(sc["S_dry"], rl_d)
        j_wet = jaccard(base_wet_years, wet_years)
        j_dry = jaccard(base_dry_years, dry_years)

        trials.append({
            "trial": i,
            **{f"w_{k}": v for k, v in w.items()},
            "rho_wet": rho_wet,
            "rho_dry": rho_dry,
            "dec_same_wet": same_wet,
            "dec_within1_wet": within1_wet,
            "dec_same_dry": same_dry,
            "dec_within1_dry": within1_dry,
            "jaccard_wet": j_wet,
            "jaccard_dry": j_dry,
            "n_wet_years": len(wet_years),
            "n_dry_years": len(dry_years)
        })

    df_trials = pd.DataFrame(trials)

    # OAT tornado data
    oat_rows = []
    for name, w in one_at_a_time_levels(BASELINE_WEIGHTS, WEIGHT_RANGES):
        sc = compute_scores(df, w)
        rho_wet = spearman_rank_corr(base["S_wet"], sc["S_wet"])
        rho_dry = spearman_rank_corr(base["S_dry"], sc["S_dry"])
        rl_w = fit_rl(sc["S_wet"], q)
        rl_d = fit_rl(sc["S_dry"], q)
        wet_years = select_extreme_years(sc["S_wet"], rl_w)
        dry_years = select_extreme_years(sc["S_dry"], rl_d)
        j_w = jaccard(base_wet_years, wet_years)
        j_d = jaccard(base_dry_years, dry_years)
        oat_rows.append({
            "param": name,
            "rho_wet": rho_wet,
            "rho_dry": rho_dry,
            "jaccard_wet": j_w,
            "jaccard_dry": j_d
        })
    df_oat = pd.DataFrame(oat_rows)
    # Aggregate OAT by parameter (min/max or 5th/95th)
    agg = (df_oat.groupby("param")
           .agg(rho_wet_min=("rho_wet", "min"),
                rho_wet_max=("rho_wet", "max"),
                rho_dry_min=("rho_dry", "min"),
                rho_dry_max=("rho_dry", "max"),
                j_wet_min=("jaccard_wet", "min"),
                j_wet_max=("jaccard_wet", "max"),
                j_dry_min=("jaccard_dry", "min"),
                j_dry_max=("jaccard_dry", "max"))
           .reset_index())

    # ----------------------------
    # Plot
    # ----------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1, 1], hspace=0.25, wspace=0.22)

    # (A) Tornado: show spread of Spearman rho (1-at-a-time)
    axA = fig.add_subplot(gs[0, 0])
    params = list(agg["param"])
    y = np.arange(len(params))
    # plot bars for wet and dry using the range min..max
    axA.hlines(y, agg["rho_wet_min"], agg["rho_wet_max"], color="#1f77b4", linewidth=6, label="Wet (ρ)")
    axA.hlines(y, agg["rho_dry_min"], agg["rho_dry_max"], color="#d62728", linewidth=6, label="Dry (ρ)")
    axA.set_yticks(y)
    axA.set_yticklabels(params)
    axA.set_xlim(0, 1.0)
    axA.set_xlabel("Spearman rank correlation vs. baseline")
    axA.set_title(f"(A) Tornado (OAT perturbations): rank stability — {wu_name}")
    axA.legend(loc="lower right", frameon=True)

    # (B) Boxplots: Spearman rho across random samples
    axB = fig.add_subplot(gs[0, 1])
    box_data = [df_trials["rho_wet"].values, df_trials["rho_dry"].values]
    axB.boxplot(box_data, labels=["Wet ρ", "Dry ρ"], patch_artist=True,
                boxprops=dict(facecolor="#1f77b4", alpha=0.3),
                medianprops=dict(color="black"))
    # overlay dry with another color
    for patch, color in zip(axB.artists, ["#1f77b4", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25 if color == "#1f77b4" else 0.25)
    axB.set_ylim(0, 1.0)
    axB.set_title("(B) Rank stability across random weight simplex draws")
    axB.grid(True, axis="y", alpha=0.3)

    # (C) Boxplots: Jaccard overlap for extreme-year sets across random samples
    axC = fig.add_subplot(gs[1, 0])
    box_data2 = [df_trials["jaccard_wet"].values, df_trials["jaccard_dry"].values]
    bp = axC.boxplot(box_data2, labels=["Wet Jaccard", "Dry Jaccard"], patch_artist=True,
                     boxprops=dict(facecolor="#1f77b4", alpha=0.3),
                     medianprops=dict(color="black"))
    for patch, color in zip(axC.artists, ["#1f77b4", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    axC.set_ylim(0, 1.0)
    axC.set_title("(C) Extreme-year set overlap vs. baseline (Jaccard)")
    axC.grid(True, axis="y", alpha=0.3)

    # (D) Decile stability bar(s)
    axD = fig.add_subplot(gs[1, 1])
    same_wet = df_trials["dec_same_wet"].median()
    same_dry = df_trials["dec_same_dry"].median()
    within_wet = df_trials["dec_within1_wet"].median()
    within_dry = df_trials["dec_within1_dry"].median()
    labels = ["Wet: same decile", "Wet: ±1 decile", "Dry: same decile", "Dry: ±1 decile"]
    vals = [same_wet, within_wet, same_dry, within_dry]
    colors = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
    axD.bar(labels, vals, color=colors, alpha=0.6)
    axD.yaxis.set_major_formatter(PercentFormatter(1.0))
    axD.set_ylim(0, 1.0)
    axD.set_title("(D) Median decile stability across random draws")
    for i, v in enumerate(vals):
        axD.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Weights sensitivity — {wu_name}  (q={GEV_Q})", fontsize=13, y=0.98)
    for ax in [axA, axB, axC, axD]:
        ax.tick_params(axis='x', rotation=15)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"[OK] Figure saved: {out_png}")

    # Write trials to CSV
    df_trials.to_csv(out_csv, index=False)
    print(f"[OK] Metrics saved: {out_csv}")

    # Console summary (for manuscript text)
    summary = {
        "rho_wet_med": df_trials["rho_wet"].median(),
        "rho_wet_p05": df_trials["rho_wet"].quantile(0.05),
        "rho_dry_med": df_trials["rho_dry"].median(),
        "rho_dry_p05": df_trials["rho_dry"].quantile(0.05),
        "jac_wet_med": df_trials["jaccard_wet"].median(),
        "jac_wet_p05": df_trials["jaccard_wet"].quantile(0.05),
        "jac_dry_med": df_trials["jaccard_dry"].median(),
        "jac_dry_p05": df_trials["jaccard_dry"].quantile(0.05),
        "dec_same_wet_med": same_wet,
        "dec_same_dry_med": same_dry,
        "dec_within1_wet_med": within_wet,
        "dec_within1_dry_med": within_dry
    }
    print("\n=== Suggested text-ready metrics ===")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")


def main():
    """ parser = argparse.ArgumentParser(description="Weights sensitivity for wet/dry scores with GEV selection.")
    parser.add_argument("--csv", required=True, help="Input CSV with columns: year,pct_wet,pct_wet_plus,pct_dry")
    parser.add_argument("--wu",  required=True, help="Work Unit name for titling, e.g., 01AL000")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--q", type=float, default=GEV_Q, help="Quantile for GEV RL (default 0.917)")
    parser.add_argument("--n", type=int, default=N_RANDOM, help="Number of random weight draws (default 500)")
    args = parser.parse_args()
 """
    incsv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\gisdata\\data\\05OG000-long-series.csv" 
    wu =  "05OG000" 
    outdir =  "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\sensitivity" 
    q = 0.917 
    n = 500

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(incsv)

    out_png = os.path.join(outdir, f"weights_sensitivity_{wu}.png")
    out_csv = os.path.join(outdir, f"weights_sensitivity_{wu}.csv")
    analyze(df, out_png, out_csv, wu, q=q, n_random=n)


if __name__ == "__main__":
    main()


#######
""" 
python weights-sensitivity4scores.py \
    --csv 01AL000_fractions.csv \
    --wu 01AL000 \
    --outdir results \
    --q 0.917 \
    --n 500 """
##############

 