# neighbor_expansion_batch_parallel.py
# ---------------------------------------------------------------------------------
# Purpose: Parallelized version of neighbor expansion sensitivity batch runner.
#          Designed for robust multi-process execution on Windows/macOS/Linux.

# ---------------------------------------------------------------------------------

# --- Use non-interactive backend for multi-process figure saving BEFORE pyplot ---
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Iterable, Tuple

sns.set(style="whitegrid", context="talk")

# ---------------------------------------------------------------------------------
# Helpers: robust axis scaling
# ---------------------------------------------------------------------------------

def _finite_min_max(arrs, default_ymin=0.0, default_ymax=1.0, pad_ratio=0.05):
    """
    Given one or more arrays, return (ymin, ymax) from all finite values with padding.
    If all values are NaN or equal, return a default span.
    """
    vals = []
    for a in arrs:
        a = np.asarray(a)
        vals.extend(list(a[np.isfinite(a)]))
    if len(vals) == 0:
        return default_ymin, default_ymax
    vmin, vmax = np.min(vals), np.max(vals)
    if np.isclose(vmin, vmax, atol=1e-12):
        span = max(abs(vmin), 1.0) * pad_ratio
        return vmin - span, vmax + span
    pad = max((vmax - vmin) * pad_ratio, 1e-6)
    return vmin - pad, vmax + pad


def _set_dynamic_xaxis(ax, margins):
    """
    Force x-axis ticks to the tested margins and add small padding to x-limits.
    """
    x_ticks = sorted(set(margins))
    ax.set_xticks(x_ticks)
    x_min, x_max = min(x_ticks), max(x_ticks)
    pad = max((x_max - x_min) * 0.05, 1e-4)
    ax.set_xlim(x_min - pad, x_max + pad)


# ---------------------------------------------------------------------------------
# Wide -> long reshape helper
# ---------------------------------------------------------------------------------


def reshape_all_years_wide_to_long(
    all_years_csv: str,
    years: Iterable[int] = range(2000, 2024),
    counts_are_fractions: bool = False,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Reshape a wide all_years.csv (columns like '2000_lt','2000_mid','2000_gt2','2000_gte')
    into a long DataFrame with:
      tile_id, year, wet_fraction, dry_fraction, wet_score, dry_score

    - If counts_are_fractions=False (typical): lt/gte/gt2 are counts; normalize by N=lt+gte.
    - If counts_are_fractions=True: lt/gte/gt2 treated as fractions already in [0,1].
    - Weights default to w_wet=1.0, w_vwet=1.5, w_dry=1.0, alpha=2.0, beta=1.0.
    - Enforces gt2 <= gte before normalization to avoid artifacts.
    """
    # Default weights consistent with your methods text
    W = dict(w_wet=1.0, w_vwet=1.5, w_dry=1.0, alpha=2.0, beta=1.0)
    if isinstance(weights, dict):
        W.update(weights)

    df_wide = pd.read_csv(all_years_csv, sep=None, engine="python")
    if "tile_id" not in df_wide.columns:
        raise ValueError("Expected 'tile_id' in all_years.csv")

    out_rows: list[dict] = []
    wide_cols = set(df_wide.columns)

    for _, r in df_wide.iterrows():
        tile_id = str(r["tile_id"])
        for yr in years:
            # Require at least lt & gte; gt2 optional; mid not used for normalization.
            need = {f"{yr}_lt", f"{yr}_gte"}
            if not need.issubset(wide_cols):
                continue

            lt  = float(r.get(f"{yr}_lt",  np.nan))
            gte = float(r.get(f"{yr}_gte", np.nan))
            gt2 = float(r.get(f"{yr}_gt2", 0.0)) if f"{yr}_gt2" in wide_cols else 0.0

            if not np.isfinite(lt) or not np.isfinite(gte):
                out_rows.append({
                    "tile_id": tile_id, "year": int(yr),
                    "wet_fraction": np.nan, "dry_fraction": np.nan,
                    "wet_score": np.nan, "dry_score": np.nan
                })
                continue

            # Clip: gt2 must be subset of wet (gte)
            if np.isfinite(gt2) and gt2 > gte:
                gt2 = gte

            if counts_are_fractions:
                # Treat lt, gte, gt2 as fractions; renormalize safeguard
                N = max((lt if np.isfinite(lt) else 0.0) + (gte if np.isfinite(gte) else 0.0), 1e-9)
                pct_dry  = (lt  if np.isfinite(lt) else 0.0) / N
                pct_wet  = (gte if np.isfinite(gte) else 0.0) / N
                pct_vwet = min((gt2 if np.isfinite(gt2) else 0.0) / N, pct_wet)
            else:
                N = lt + gte
                if N <= 0:
                    pct_dry = pct_wet = pct_vwet = np.nan
                else:
                    pct_dry  = lt  / N
                    pct_wet  = gte / N
                    pct_vwet = min((gt2 / N) if np.isfinite(gt2) else 0.0, pct_wet)

            if any(np.isnan([pct_dry, pct_wet, pct_vwet])):
                wet_score = dry_score = np.nan
            else:
                wet_score = (W["w_wet"] * pct_wet) + (W["w_vwet"] * pct_vwet) - (W["w_dry"] * pct_dry)
                dry_score = pct_dry - (W["alpha"] * pct_vwet) - (W["beta"] * pct_wet)

            out_rows.append({
                "tile_id": tile_id,
                "year": int(yr),
                "wet_fraction": float(pct_wet),
                "dry_fraction": float(pct_dry),
                "wet_score": float(wet_score) if np.isfinite(wet_score) else np.nan,
                "dry_score": float(dry_score) if np.isfinite(dry_score) else np.nan,
            })

    df_series = pd.DataFrame(out_rows).sort_values(["tile_id", "year"]).reset_index(drop=True)
    return df_series


def build_series_from_wide_and_scores(
    all_years_csv: str,
    scores_csv: str | None = None,
    years: Iterable[int] = range(2000, 2024),
    counts_are_fractions: bool = False,
    weights: dict | None = None,
    prefer_scores: bool = True,
) -> pd.DataFrame:
    """
    Build a single long 'series_df' using BOTH sources:
      1) Wide counts file -> wet_fraction, dry_fraction (+ scores if desired)
      2) Per-year scores file -> wet_score, dry_score (and optionally pct_wet/pct_dry)

    Preference:
      - If 'prefer_scores' is True, take wet_score/dry_score from the scores CSV when present;
        otherwise use the scores computed from the wide counts.
      - Fractions (wet_fraction, dry_fraction) come from the wide file. If the scores CSV
        offers 'pct_wet'/'pct_dry', they are used only to fill missing values.

    Output columns (guaranteed): tile_id, year, wet_fraction, dry_fraction, wet_score, dry_score
    """
    # 1) Long from wide
    wide_long = reshape_all_years_wide_to_long(
        all_years_csv=all_years_csv,
        years=years,
        counts_are_fractions=counts_are_fractions,
        weights=weights,
    )

    if scores_csv is None or not str(scores_csv).strip():
        # Use only wide-derived values
        return wide_long

    # 2) Read per-year scores CSV
    sc = pd.read_csv(scores_csv)
    # normalize column names if present: use pct_* to fill fractions
    colmap = {}
    if "pct_wet" in sc.columns and "wet_fraction" not in sc.columns:
        colmap["pct_wet"] = "wet_fraction"
    if "pct_dry" in sc.columns and "dry_fraction" not in sc.columns:
        colmap["pct_dry"] = "dry_fraction"
    if colmap:
        sc = sc.rename(columns=colmap)

    needed = {"tile_id", "year"}
    if not needed.issubset(set(sc.columns)):
        raise ValueError("scores_csv must contain 'tile_id' and 'year' columns.")

    # Keep only relevant columns; tolerate presence/absence
    keep_cols = ["tile_id", "year", "wet_score", "dry_score", "wet_fraction", "dry_fraction"]
    keep_cols = [c for c in keep_cols if c in sc.columns]
    sc = sc[keep_cols].copy()

    # Dtypes
    sc["tile_id"] = sc["tile_id"].astype(str)
    sc["year"] = sc["year"].astype(int)
    wide_long["tile_id"] = wide_long["tile_id"].astype(str)
    wide_long["year"] = wide_long["year"].astype(int)

    # 3) Merge
    merged = pd.merge(
        wide_long,
        sc,
        on=["tile_id", "year"],
        how="outer",
        suffixes=("_from_wide", "_from_scores"),
    ).sort_values(["tile_id", "year"]).reset_index(drop=True)

    # 4) Compose final columns with precedence
    def coalesce(*arrs):
        for a in arrs:
            if a is None:
                continue
            if pd.notna(a):
                return a
        return np.nan

    final_rows = []
    for _, r in merged.iterrows():
        tile = str(r["tile_id"])
        yr   = int(r["year"])

        wf_w = r.get("wet_fraction_from_wide", np.nan)
        df_w = r.get("dry_fraction_from_wide", np.nan)
        wf_s = r.get("wet_fraction_from_scores", np.nan)
        df_s = r.get("dry_fraction_from_scores", np.nan)

        # Fractions: prefer wide (authoritative), but fill from scores if wide missing
        wet_frac = coalesce(wf_w, wf_s)
        dry_frac = coalesce(df_w, df_s)

        # Scores: prefer external scores if requested and present
        ws_w = r.get("wet_score_from_wide", np.nan)
        ds_w = r.get("dry_score_from_wide", np.nan)
        ws_s = r.get("wet_score_from_scores", np.nan)
        ds_s = r.get("dry_score_from_scores", np.nan)

        if prefer_scores:
            wet_sc = coalesce(ws_s, ws_w)
            dry_sc = coalesce(ds_s, ds_w)
        else:
            wet_sc = coalesce(ws_w, ws_s)
            dry_sc = coalesce(ds_w, ds_s)

        final_rows.append({
            "tile_id": tile,
            "year": yr,
            "wet_fraction": wet_frac,
            "dry_fraction": dry_frac,
            "wet_score": wet_sc,
            "dry_score": dry_sc,
        })

    series_df = pd.DataFrame(final_rows).sort_values(["tile_id", "year"]).reset_index(drop=True)
    return series_df


# ---------------------------------------------------------------------------------
# Core sensitivity functions
# ---------------------------------------------------------------------------------

def _empirical_quantiles(x):
    x = np.asarray(x)
    ranks = pd.Series(x).rank(method="average")
    q = ranks / (len(x) + 1.0)
    return q.values


def _contiguous_blocks(years_selected):
    if len(years_selected) == 0:
        return []
    years = np.sort(np.array(years_selected, dtype=int))
    runs, run_len = [], 1
    for i in range(1, len(years)):
        if years[i] == years[i - 1] + 1:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return runs


def _jaccard(a, b):
    A, B = set(a), set(b)
    if len(A) == 0 and len(B) == 0:
        return 1.0
    u = len(A | B)
    i = len(A & B)
    return i / u if u else 0.0


def neighbor_expansion_sensitivity_single(
    df_tile: pd.DataFrame,
    score_col: str = "wet_score",
    fraction_col: Optional[str] = "wet_fraction",
    year_col: str = "year",
    q_base: float = 0.91,
    margins: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03),
    windows: Tuple[int, ...] = (1, 2),
    tile_label: str = "tile",
    savefig_path: Optional[str] = None,
):
    """
    Compute sensitivity metrics and produce a figure for one tile.
    Returns summary_df (metrics across margins/windows).
    """
    # Clean and sort
    cols = [year_col, score_col] + ([fraction_col] if (fraction_col and (fraction_col in df_tile.columns)) else [])
    df = df_tile[cols].dropna(subset=[score_col]).copy()
    df[year_col] = df[year_col].astype(int)
    df.sort_values(year_col, inplace=True)

    years = df[year_col].values
    scores = df[score_col].values
        
    # FRACTIONS (may have one or both). We'll compute separations for both if present.
    wet_frac = df["wet_fraction"].values if "wet_fraction" in df.columns else None
    dry_frac = df["dry_fraction"].values if "dry_fraction" in df.columns else None

    q = _empirical_quantiles(scores)
    years_base = years[q >= q_base]

    rows = []
    have_frac = (fraction_col is not None) and (fraction_col in df.columns)
    frac = df[fraction_col].values if have_frac else None

    for m in margins:
        q_neigh = q_base - m
        for k in windows:
            years_exp = set(years_base.tolist())
            for y in years_base:
                for delta in (-k, +k):
                    y_nei = y + delta
                    idx = np.where(years == y_nei)[0]
                    if idx.size > 0 and q[idx[0]] >= q_neigh:
                        years_exp.add(y_nei)
            years_exp = sorted(list(years_exp))
            years_exp_only = sorted(list(set(years_exp) - set(years_base)))

            runs_base = _contiguous_blocks(years_base)
            runs_exp = _contiguous_blocks(years_exp)
            mean_run_base = np.mean(runs_base) if runs_base else 0.0
            mean_run_exp = np.mean(runs_exp) if runs_exp else 0.0
            singleton_rate_base = (np.sum(np.array(runs_base) == 1) / len(runs_base)) if runs_base else np.nan
            singleton_rate_exp = (np.sum(np.array(runs_exp) == 1) / len(runs_exp)) if runs_exp else np.nan
            jacc = _jaccard(years_base, years_exp)

            expanded_q = []
            if len(years_exp_only) > 0:
                year_to_q = dict(zip(years, q))
                expanded_q = [year_to_q[y] for y in years_exp_only]
            share_below_base = np.mean([qq < q_base for qq in expanded_q]) if expanded_q else np.nan
            mean_q_exp_only = np.mean(expanded_q) if expanded_q else np.nan


            # Masks for baseline/expanded selections
            mask_base = np.isin(years, years_base)
            mask_exp  = np.isin(years, years_exp)

            # Separation helper
            def _sep(x, m):
                if x is None or not np.isfinite(x).any():
                    return np.nan
                return np.nanmean(x[m]) - np.nanmean(x[~m])

            sep_wet_base = _sep(wet_frac, mask_base)
            sep_wet_exp  = _sep(wet_frac, mask_exp)
            sep_dry_base = _sep(dry_frac, mask_base)
            sep_dry_exp  = _sep(dry_frac, mask_exp)


            
            rows.append({
                "tile_id": tile_label,
                "score_col": score_col,
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
                # NEW: both separations
                "sep_wet_base": sep_wet_base,
                "sep_wet_exp":  sep_wet_exp,
                "sep_dry_base": sep_dry_base,
                "sep_dry_exp":  sep_dry_exp,
            })


    summary_df = pd.DataFrame(rows)

    # Choose best by largest delta mean_run
    summary_df["delta_mean_run"] = summary_df["mean_run_exp"] - summary_df["mean_run_base"]
    best = summary_df.sort_values(["delta_mean_run", "margin"], ascending=[False, True]).iloc[0]
    best_m, best_k = best["margin"], int(best["window_k"])  # type: ignore

    # Build figure
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Panel A: persistence vs margin
    axA = fig.add_subplot(gs[0, 0])
    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        axA.plot(sel["margin"], sel["mean_run_exp"], marker="o", label=f"mean run (expanded), k={int(k)}")
        axA.plot(sel["margin"], sel["mean_run_base"], marker="x", ls="--", label=f"mean run (baseline), k={int(k)}")
    axA.set_title(f"(A) Persistence vs margin\n{tile_label}")
    axA.set_xlabel("Neighbor margin (quantile units)")
    axA.set_ylabel("Mean run length (years)")
    axA.legend(loc="best")

    # Panel B: over-inclusion vs margin
    axB = fig.add_subplot(gs[0, 1] )
    y_curves_B = []
    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        y = sel["share_exp_only_below_q_base"].values
        axB.plot(sel["margin"], y, marker="o", label=f"share expanded-only < q_base (k={int(k)})")
        y_curves_B.append(y)

    axB.set_title(f"(B) Over-inclusion vs margin\n{tile_label}")
    axB.set_xlabel("Neighbor margin (quantile units)")
    axB.set_ylabel("Share below q_base")
    _set_dynamic_xaxis(axB, summary_df["margin"].values)

    # NEW: robust dynamic y-limits for Panel B (ignores NaNs, pads; fallback [0,1])
    if y_curves_B:
        vals = np.concatenate([np.asarray(v, float) for v in y_curves_B])
        finite = vals[np.isfinite(vals)]
        if finite.size:
            ymin, ymax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if np.isclose(ymin, ymax, atol=1e-12):
                pad = max(abs(ymin), 1.0) * 0.08
                axB.set_ylim(ymin - pad, ymax + pad)
            else:
                pad = max((ymax - ymin) * 0.08, 1e-6)
                axB.set_ylim(ymin - pad, ymax + pad)
        else:
            axB.set_ylim(0.0, 1.0)
    else:
        axB.set_ylim(0.0, 1.0)

    axB.legend(loc="best")
    axB.grid(True, alpha=0.3)


    # Panel C: stability (Jaccard) vs margin
    axC = fig.add_subplot(gs[1, 0])
    y_curves_C = []
    style = {1: ("-", "o"), 2: ("--", "s")}  # k=1 solid, k=2 dashed

    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        y = sel["jaccard_vs_base"].values
        ls, mk = style.get(int(k), (":", "^"))
        axC.plot(sel["margin"], y, marker=mk, linestyle=ls, label=f"Jaccard (k={int(k)})")
        y_curves_C.append(y)
    axC.set_title(f"(C) Stability vs margin\n{tile_label}")
    axC.set_xlabel("Neighbor margin (quantile units)")
    axC.set_ylabel("Jaccard vs baseline")
    _set_dynamic_xaxis(axC, summary_df["margin"].values)
    
    # Robust dynamic y-limits for Panel C
    if y_curves_C:
        vals = np.concatenate([np.asarray(v, float) for v in y_curves_C])
        finite = vals[np.isfinite(vals)]
        if finite.size:
            ymin, ymax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if np.isclose(ymin, ymax, atol=1e-12):
                pad = max(abs(ymin), 1.0) * 0.08
                axC.set_ylim(ymin - pad, ymax + pad)
            else:
                pad = max((ymax - ymin) * 0.08, 1e-6)
                axC.set_ylim(ymin - pad, ymax + pad)
        else:
            axC.set_ylim(0.0, 1.0)
    else:
        axC.set_ylim(0.0, 1.0)

    axC.legend(loc="best")
    axC.grid(True, alpha=0.3)


    # Panel D: timeline of baseline vs expanded-only for best (m, k)
    q_neigh = q_base - best_m
    years = df[year_col].values
    q = _empirical_quantiles(df[score_col].values)
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
    show_series = df[fraction_col].values if (have_frac and (fraction_col in df.columns)) else q
    axD.plot(years, show_series, lw=2, color="tab:blue",
             label=("wet/dry fraction" if have_frac else "score quantile q"))
    for y in years_base:
        axD.axvspan(y - 0.5, y + 0.5, color="tab:blue", alpha=0.15)
    for y in set(years_exp) - set(years_base):
        axD.axvspan(y - 0.5, y + 0.5, color="tab:orange", alpha=0.20)
    axD.set_title((f"(D) Timeline: baseline (blue) vs expanded-only (orange)\n"
                   f"{tile_label}\n"
                   f"best margin={best_m:.3f}, k={best_k}, q_base={q_base:.3f}"))
    axD.set_xlabel("Year")
    axD.set_ylabel(("Wet/Dry fraction" if have_frac else "Score quantile q"))
    axD.xaxis.set_major_locator(MaxNLocator(integer=True))
    axD.legend(loc="upper left")

    out_path = savefig_path or f"neighbor_sensitivity_{tile_label}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved sensitivity figure to: {out_path}")

    return summary_df


# ---------------------------------------------------------------------------------
# Parallel worker + batch wrapper
# ---------------------------------------------------------------------------------

def _tile_arrays_from_series(series_df: pd.DataFrame, tile: str, years_col: str, score_col: str,
                             fraction_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (years, scores, fractions_or_None) arrays for one tile."""
    df = series_df.loc[series_df["tile_id"] == tile, [years_col, score_col] + ([fraction_col] if (fraction_col and (fraction_col in series_df.columns)) else [])].copy()
    df = df.dropna(subset=[score_col])
    df[years_col] = df[years_col].astype(int)
    df.sort_values(years_col, inplace=True)
    years = df[years_col].values
    scores = df[score_col].values
    fracs = df[fraction_col].values if (fraction_col and (fraction_col in df.columns)) else None
    return years, scores, fracs


def _run_tile_worker(args):
    """Worker that reconstructs a small DataFrame from arrays and runs single-tile analysis."""
    (tile,
     years,
     scores,
     fracs,
     score_col,
     fraction_col,
     years_col,
     q_base,
     margins,
     windows,
     fig_path) = args

    try:
        data = {years_col: years, score_col: scores}
        if (fraction_col is not None) and (fracs is not None):
            data[fraction_col] = fracs
        df_tile = pd.DataFrame(data)

        summary = neighbor_expansion_sensitivity_single(
            df_tile=df_tile,
            score_col=score_col,
            fraction_col=fraction_col if (fraction_col in df_tile.columns) else None,
            year_col=years_col,
            q_base=q_base,
            margins=margins,
            windows=windows,
            tile_label=str(tile),
            savefig_path=fig_path,
        )
        return (tile, summary, None)
    except Exception as e:
        return (tile, None, str(e))


def batch_neighbor_expansion_sensitivity(
    series_df: pd.DataFrame,
    justification_df: pd.DataFrame,
    score_type: str = "wet",  # "wet" or "dry"
    margins: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05),
    windows: Tuple[int, ...] = (1, 2, 3),
    output_dir: str = "neighbor_sensitivity_outputs",
    years_col: str = "year",
    n_workers: Optional[int] = None,
):
    """
    Parallelized batch runner: executes neighbor expansion sensitivity for every tile_id
    using ProcessPoolExecutor, writes a combined CSV, and saves per-tile figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_fig_dir = os.path.join(output_dir, f"figs_{score_type}")
    os.makedirs(out_fig_dir, exist_ok=True)

    score_col = f"{score_type}_score"
    fraction_col = f"{score_type}_fraction"
    q_col = f"recommended_q_{score_type}"

    # sanity checks
    if score_col not in series_df.columns:
        raise ValueError(f"series_df missing column: {score_col}")
    if years_col not in series_df.columns or "tile_id" not in series_df.columns:
        raise ValueError("series_df must include 'tile_id' and year column.")

    tiles = list(map(str, sorted(series_df["tile_id"].unique())))

    # q_base per tile
    q_map = {t: 0.91 for t in tiles}
    if q_col in justification_df.columns:
        just = justification_df[["tile_id", q_col]].dropna()
        for _, r in just.iterrows():
            q_map[str(r["tile_id"])] = float(r[q_col])

    # Prepare minimal args per tile (arrays only)
    tasks = []
    for tile in tiles:
        years, scores, fracs = _tile_arrays_from_series(series_df, tile, years_col, score_col, fraction_col)
        fig_path = os.path.join(out_fig_dir, f"neighbor_sensitivity_{score_type}_{tile}.png")
        tasks.append((tile, years, scores, fracs, score_col, (fraction_col if (fraction_col in series_df.columns) else None), years_col, q_map[tile], margins, windows, fig_path))

    # choose worker count
    if n_workers is None:
        cpu = os.cpu_count() or 1
        n_workers = max(1, min(cpu - 1, 6))  # leave 1 core free, cap at 6 by default for laptops

    print(f"Launching {len(tasks)} tiles on {n_workers} worker(s)...")

    # run in parallel
    all_summaries = []
    errors = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_run_tile_worker, args) for args in tasks]
        for fut in as_completed(futures):
            tile, summary, err = fut.result()
            if err is None:
                all_summaries.append(summary)
            else:
                errors.append((tile, err))
                print(f"[WARN] Tile {tile} failed in parallel: {err}")

    # Fallback: retry failed tiles sequentially to avoid pool-wide aborts
    if errors:
        print(f"Retrying {len(errors)} failed tile(s) sequentially...")
        for tile, _ in errors:
            years, scores, fracs = _tile_arrays_from_series(series_df, tile, years_col, score_col, fraction_col)
            fig_path = os.path.join(out_fig_dir, f"neighbor_sensitivity_{score_type}_{tile}.png")
            _, summary, err = _run_tile_worker((tile, years, scores, fracs, score_col,
                                                (fraction_col if (fraction_col in series_df.columns) else None),
                                                years_col, q_map[tile], margins, windows, fig_path))
            if err is None:
                all_summaries.append(summary)
            else:
                print(f"[ERROR] Tile {tile} failed again sequentially: {err}")

    if not all_summaries:
        raise RuntimeError("No successful tiles were processed.")

    # combine and write outputs
    combined = pd.concat(all_summaries, ignore_index=True)
    out_csv = os.path.join(output_dir, f"neighbor_sensitivity_summary_{score_type}.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Wrote summary CSV: {out_csv}")

    # aggregate medians
    agg = (combined
           .groupby(["margin", "window_k"], as_index=False)
           .agg({
               "delta_mean_run": "median",
               "singleton_rate_base": "median",
               "singleton_rate_exp": "median",
               "jaccard_vs_base": "median",
               "share_exp_only_below_q_base": "median",
              # "sep_fraction_base": "median",
              # "sep_fraction_exp": "median",
           }))
    out_agg_csv = os.path.join(output_dir, f"neighbor_sensitivity_aggregate_{score_type}.csv")
    agg.to_csv(out_agg_csv, index=False)
    print(f"Wrote aggregate CSV: {out_agg_csv}")

    return combined, agg


# ---------------------------------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import platform
    if platform.system() == "Windows":
        # for safe multiprocessing when frozen/executed on Windows
        import multiprocessing
        multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Parallel neighbor expansion sensitivity runner")
    parser.add_argument("--series", type=str, help="Path to long-format series CSV", required=False)
    parser.add_argument("--just", type=str, help="Path to justification CSV", required=False)
    parser.add_argument("--score", type=str, choices=["wet", "dry"], default="wet")
    parser.add_argument("--out", type=str, default="neighbor_sensitivity_outputs")
    parser.add_argument("--years_col", type=str, default="year")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--margins", type=str, default="0.0,0.01,0.02,0.03",
                        help="Comma-separated margins (quantile units)")
    parser.add_argument("--windows", type=str, default="1,2,3,5",
                        help="Comma-separated window sizes (k)")
    parser.add_argument("--reshape", action="store_true",
                        help="If set, expects --series to be a wide all_years.csv and reshapes it")

    args = parser.parse_args()

    # parse tuples
    margins = tuple(float(x) for x in args.margins.split(",") if x)
    windows = tuple(int(x) for x in args.windows.split(",") if x)

    # Example paths (edit or pass via CLI)
   
    
    # default: reshape wide -> long
    
    # Example paths (replace with your actuals or wire into argparse/config)
    ALL_YEARS_CSV = r"D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\data\\all_years-combined-final.csv"
    SCORES_CSV    = r"D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\results\\wet_dry_scores.csv"   # the one that has tile_id,year,wet_score,dry_score,(pct_*)

    series_df = build_series_from_wide_and_scores(
        all_years_csv=ALL_YEARS_CSV,
        scores_csv=SCORES_CSV,                # set to None to use only wide file
        years=range(2000, 2024),
        counts_are_fractions=False,           # your wide file holds counts, not fractions
        weights=dict(w_wet=1.0, w_vwet=1.5, w_dry=1.0, alpha=2.0, beta=1.0),
        prefer_scores=True  )                # prefer scores from the per-year CSV


    if args.just is None:
        JUST_CSV = r"D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\results\\2000-2023percentile_justification.csv"
        justification_df = pd.read_csv(JUST_CSV)
    else:
        justification_df = pd.read_csv(args.just)

    combined, agg = batch_neighbor_expansion_sensitivity(
        series_df=series_df,
        justification_df=justification_df,
        score_type=args.score,
        margins=margins,
        windows=windows,
        output_dir=args.out,
        years_col=args.years_col,
        n_workers=args.workers,
    )

    print("\nAggregate (median across tiles):")
    print(agg.round(3))
