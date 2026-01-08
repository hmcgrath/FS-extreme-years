
# neighbor_expansion_sensitivity.py
# Purpose: Quantify and visualize neighbor expansion rationale:
#          persistence benefits vs over-inclusion, with sensitivity over margins/windows.
# Author: M365 Copilot (for Heather McGrath)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set(style="whitegrid", context="talk")

def _empirical_quantiles(x):
    """Return empirical CDF quantiles q_t for a 1D array x (ties averaged)."""
    x = np.asarray(x)
    ranks = pd.Series(x).rank(method="average")
    q = ranks / (len(x) + 1.0)
    return q.values

def _contiguous_blocks(years_selected):
    """Compute contiguous block lengths (run-lengths) on an integer year array."""
    if len(years_selected) == 0:
        return []
    years = np.sort(np.array(years_selected, dtype=int))
    runs = []
    run_len = 1
    for i in range(1, len(years)):
        if years[i] == years[i-1] + 1:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return runs

def _jaccard(a, b):
    """Jaccard similarity of two year sets."""
    A, B = set(a), set(b)
    if len(A) == 0 and len(B) == 0:
        return 1.0
    u = len(A | B)
    i = len(A & B)
    return i / u if u else 0.0

def neighbor_expansion_sensitivity(
    df_tile,
    score_col="wet_score",
    fraction_col="wet_fraction",
    year_col="year",
    q_base=0.91,
    margins=(0.0, 0.01, 0.02, 0.03),
    windows=(1, 2),
    tile_label="tile",
    savefig_path=None
):
    """
    Sensitivity test for neighbor expansion.

    Parameters
    ----------
    df_tile : pd.DataFrame
        Must contain columns [year_col, score_col]; fraction_col optional.
    score_col : str
        Name of the score series used to rank extremes (wet_score or dry_score).
    fraction_col : str or None
        Pixel exceedance fraction column (wet_fraction or dry_fraction). If None, fraction metrics are skipped.
    year_col : str
        Year column (integers).
    q_base : float
        Baseline recommended quantile threshold (e.g., recommended_q_wet).
    margins : tuple of float
        Margins to subtract from q_base for neighbor inclusion (e.g., 0, 0.01, 0.02).
    windows : tuple of int
        Neighbor window sizes k: include t±k if q_{t±k} >= q_base - margin.
    tile_label : str
        For titles/labels in the figure.
    savefig_path : str or None
        If provided, save the figure here (PNG). Otherwise save to "neighbor_sensitivity_{tile_label}.png".

    Returns
    -------
    summary_df : pd.DataFrame
        Table of metrics per (margin, window).
    """

    # Clean and sort
    df = df_tile[[year_col, score_col] + ([fraction_col] if fraction_col in df_tile.columns else [])].copy()
    df = df.dropna(subset=[score_col])  # must have score
    df[year_col] = df[year_col].astype(int)
    df.sort_values(year_col, inplace=True)

    years = df[year_col].values
    scores = df[score_col].values
    q = _empirical_quantiles(scores)  # empirical percentiles of the score series

    # Baseline extreme years by quantile threshold
    base_mask = q >= q_base
    years_base = years[base_mask]

    # Prepare summary rows
    rows = []

    # Optional fraction array for separation/over-inclusion checks
    have_frac = fraction_col in df.columns
    frac = df[fraction_col].values if have_frac else None

    # Iterate margins and neighbor windows
    for m in margins:
        q_neigh = q_base - m
        for k in windows:
            # Expanded selection: start with baseline, then add neighbors t±k if they meet q >= q_neigh
            years_exp = set(years_base.tolist())
            for y in years_base:
                for delta in (-k, +k):
                    y_nei = y + delta
                    # Find index of neighbor year
                    idx = np.where(years == y_nei)[0]
                    if idx.size > 0:
                        if q[idx[0]] >= q_neigh:
                            years_exp.add(y_nei)

            years_exp = sorted(list(years_exp))
            # Expanded-only years
            years_exp_only = sorted(list(set(years_exp) - set(years_base)))

            # Persistence metrics (runs)
            runs_base = _contiguous_blocks(years_base)
            runs_exp = _contiguous_blocks(years_exp)
            mean_run_base = np.mean(runs_base) if runs_base else 0.0
            mean_run_exp = np.mean(runs_exp) if runs_exp else 0.0
            singleton_rate_base = (np.sum(np.array(runs_base) == 1) / len(runs_base)) if runs_base else np.nan
            singleton_rate_exp = (np.sum(np.array(runs_exp) == 1) / len(runs_exp)) if runs_exp else np.nan

            # Stability vs baseline (Jaccard)
            jacc = _jaccard(years_base, years_exp)

            # Over-inclusion cost:
            # share of expanded-only years that are *below* the baseline threshold (q < q_base)
            # (they are by construction below q_base or just at q_neigh; we quantify how many are far below)
            expanded_q = []
            if len(years_exp_only) > 0:
                # map years -> q
                year_to_q = dict(zip(years, q))
                expanded_q = [year_to_q[y] for y in years_exp_only]
            share_below_base = np.mean([qq < q_base for qq in expanded_q]) if expanded_q else np.nan
            mean_q_exp_only = np.mean(expanded_q) if expanded_q else np.nan

            # Fraction separation (optional): difference in mean fraction selected vs not-selected
            sep_base, sep_exp = np.nan, np.nan
            if have_frac:
                mask_base = np.isin(years, years_base)
                mask_exp = np.isin(years, years_exp)
                sep_base = np.nanmean(frac[mask_base]) - np.nanmean(frac[~mask_base]) if np.any(mask_base) else np.nan
                sep_exp = np.nanmean(frac[mask_exp]) - np.nanmean(frac[~mask_exp]) if np.any(mask_exp) else np.nan

            rows.append({
                "tile": tile_label,
                "q_base": q_base,
                "margin": m,
                "window_k": k,
                "n_base": len(years_base),
                "n_expanded": len(years_exp),
                "n_exp_only": len(years_exp_only),
                "mean_run_base": mean_run_base,
                "mean_run_exp": mean_run_exp,
                "singleton_rate_base": singleton_rate_base,
                "singleton_rate_exp": singleton_rate_exp,
                "jaccard_vs_base": jacc,
                "share_exp_only_below_q_base": share_below_base,
                "mean_q_exp_only": mean_q_exp_only,
                "sep_fraction_base": sep_base,
                "sep_fraction_exp": sep_exp
            })

    summary_df = pd.DataFrame(rows)

    # ---------- Visualization ----------
    # Figure: (A) persistence (mean run length, singleton rate) vs margin (one line per k)
    #         (B) over-inclusion (share below base) vs margin (one line per k)
    #         (C) stability (Jaccard) vs margin
    #         (D) timeline: years_base and years_exp (light bands), for the *best* margin/window by persistence gain

    # Choose best (by largest delta mean_run)
    summary_df["delta_mean_run"] = summary_df["mean_run_exp"] - summary_df["mean_run_base"]
    best = summary_df.sort_values(["delta_mean_run", "margin"], ascending=[False, True]).iloc[0]
    best_m, best_k = best["margin"], int(best["window_k"])

    # Build the figure
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Panel A: persistence vs margin
    axA = fig.add_subplot(gs[0, 0])
    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        axA.plot(sel["margin"], sel["mean_run_exp"], marker="o", label=f"mean run (expanded), k={int(k)}")
        axA.plot(sel["margin"], sel["mean_run_base"], marker="x", ls="--", label=f"mean run (baseline), k={int(k)}")
    axA.set_title(f"(A) Persistence: mean run length vs margin | {tile_label}")
    axA.set_xlabel("Neighbor margin (quantile units)")
    axA.set_ylabel("Mean run length (years)")
    axA.legend(loc="best")

    # Panel B: over-inclusion vs margin
    axB = fig.add_subplot(gs[0, 1])
    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        axB.plot(sel["margin"], sel["share_exp_only_below_q_base"], marker="o", label=f"share expanded-only below q_base (k={int(k)})")
    axB.set_title(f"(B) Over-inclusion: share of expanded-only years below baseline q | {tile_label}")
    axB.set_xlabel("Neighbor margin (quantile units)")
    axB.set_ylabel("Share below q_base")
    axB.set_ylim(0, 1)
    axB.legend(loc="best")

    # Panel C: stability vs margin (Jaccard)
    axC = fig.add_subplot(gs[1, 0])
    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        axC.plot(sel["margin"], sel["jaccard_vs_base"], marker="o", label=f"Jaccard (k={int(k)})")
    axC.set_title(f"(C) Stability: Jaccard similarity vs margin | {tile_label}")
    axC.set_xlabel("Neighbor margin (quantile units)")
    axC.set_ylabel("Jaccard vs baseline")
    axC.set_ylim(0, 1)
    axC.legend(loc="best")

    # Panel D: timeline for best (margin, k)
    # recompute selections for the best setting
    q_neigh = q_base - best_m
    years_base = years[q >= q_base]
    years_exp = set(years_base.tolist())
    for y in years_base:
        for delta in (-best_k, +best_k):
            y_nei = y + delta
            idx = np.where(years == y_nei)[0]
            if idx.size > 0 and q[idx[0]] >= q_neigh:
                years_exp.add(y_nei)
    years_exp = sorted(list(years_exp))

    axD = fig.add_subplot(gs[1, 1])
    axD.plot(years, (frac if have_frac else q), lw=2, color="tab:blue",
             label=("wet_fraction" if have_frac else "score quantile q"))
    # Shade baseline extremes
    for y in years_base:
        axD.axvspan(y - 0.5, y + 0.5, color="tab:blue", alpha=0.15)
    # Shade expanded-only extremes in a different tint
    for y in set(years_exp) - set(years_base):
        axD.axvspan(y - 0.5, y + 0.5, color="tab:orange", alpha=0.20)
    axD.set_title(f"(D) Timeline: baseline (blue) vs expanded-only (orange) | {tile_label}\n"
                  f"best margin={best_m:.3f}, k={best_k}, q_base={q_base:.3f}")
    axD.set_xlabel("Year")
    axD.set_ylabel("Wet fraction" if have_frac else "Score quantile q")
    axD.xaxis.set_major_locator(MaxNLocator(integer=True))
    axD.legend(loc="upper left")

    out = savefig_path or f"neighbor_sensitivity_{tile_label}.png"
    plt.savefig(out, dpi=300)
    print(f"Saved sensitivity figure to: {out}")

    return summary_df

# -------------------------
# Example usage (single tile):
# df_tile = ...  # DataFrame with columns: year, wet_score, wet_fraction (optional)
# summary = neighbor_expansion_sensitivity(
#     df_tile, score_col="wet_score", fraction_col="wet_fraction",
#     q_base=0.91, margins=(0.0, 0.01, 0.02, 0.03), windows=(1, 2),
#     tile_label="05OG000", savefig_path="neighbor_sensitivity_05OG000.png"
# )
# print(summary.round(3))



# df_tile = r"D:\Research\FS-2dot0\results\newtop5\2000-2023\all_years.csv"
# df_tile = r"D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\1990-2023percentile_justification.csv"
# summary = neighbor_expansion_sensitivity(
#     df_tile, score_col="wet_score", fraction_col="wet_fraction",
#     q_base=0.91, margins=(0.0, 0.01, 0.02, 0.03), windows=(1, 2),
#     tile_label="05OG000", savefig_path="neighbor_sensitivity_05OG000.png"
# )
# print(summary.round(3))