# neighbor_expansion_batch_parallel.py
# ---------------------------------------------------------------------------------
# Purpose: Parallelized version of neighbor expansion sensitivity batch runner.
#          Designed for robust multi-process execution on Windows/macOS/Linux.
# Author: M365 Copilot (for Heather McGrath)
# Date: 2026-01-06
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

def reshape_all_years_wide_to_long(all_years_csv: str,
                                   years: Iterable[int] = range(1990, 2024),
                                   counts_are_fractions: bool = False) -> pd.DataFrame:
    """
    Reshape a wide all_years.csv (with columns like '1990_lt','1990_mid','1990_gt2','1990_gte')
    into a long DataFrame with: tile_id, year, wet_fraction, dry_fraction, wet_score, dry_score.
    """
    W_WET, W_VWET, W_DRY = 1.0, 2.0, 1.0
    ALPHA, BETA = 2.0, 1.0

    df_wide = pd.read_csv(all_years_csv, sep=None, engine="python")
    if "tile_id" not in df_wide.columns:
        raise ValueError("Expected 'tile_id' in all_years.csv")

    rows = []
    for _, r in df_wide.iterrows():
        tile_id = r["tile_id"]
        for yr in years:
            if not all(f"{yr}_{suffix}" in df_wide.columns for suffix in ["lt", "mid", "gt2", "gte"]):
                continue
            lt = float(r.get(f"{yr}_lt", np.nan))
            mid = float(r.get(f"{yr}_mid", np.nan))
            gt2 = float(r.get(f"{yr}_gt2", np.nan))
            gte = float(r.get(f"{yr}_gte", np.nan))

            if counts_are_fractions:
                N = max((lt if np.isfinite(lt) else 0.0) + (gte if np.isfinite(gte) else 0.0), 1e-9)
                pct_dry = (lt if np.isfinite(lt) else 0.0) / N
                pct_wet = (gte if np.isfinite(gte) else 0.0) / N
                pct_vwet = min((gt2 if np.isfinite(gt2) else 0.0) / N, pct_wet)
            else:
                if not (np.isfinite(lt) and np.isfinite(gte)):
                    pct_dry = pct_wet = pct_vwet = np.nan
                else:
                    N = lt + gte
                    if N <= 0:
                        pct_dry = pct_wet = pct_vwet = np.nan
                    else:
                        pct_dry = lt / N
                        pct_wet = gte / N
                        pct_vwet = np.nan if not np.isfinite(gt2) else min(gt2 / N, pct_wet)

            if any(np.isnan(x) for x in [pct_dry, pct_wet, pct_vwet]):
                wet_score = dry_score = np.nan
            else:
                wet_score = (W_WET * pct_wet) + (W_VWET * pct_vwet) - (W_DRY * pct_dry)
                dry_score = pct_dry - (ALPHA * pct_vwet) - (BETA * pct_wet)

            rows.append({
                "tile_id": tile_id,
                "year": int(yr),
                "wet_fraction": pct_wet,
                "dry_fraction": pct_dry,
                "wet_score": wet_score,
                "dry_score": dry_score,
            })

    df_series = pd.DataFrame(rows).sort_values(["tile_id", "year"]).reset_index(drop=True)
    return df_series


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

            sep_base = sep_exp = np.nan
            if have_frac:
                mask_base = np.isin(years, years_base)
                mask_exp = np.isin(years, years_exp)
                sep_base = np.nanmean(frac[mask_base]) - np.nanmean(frac[~mask_base]) if np.any(mask_base) else np.nan
                sep_exp = np.nanmean(frac[mask_exp]) - np.nanmean(frac[~mask_exp]) if np.any(mask_exp) else np.nan

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
                "sep_fraction_base": sep_base,
                "sep_fraction_exp": sep_exp,
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
    axB = fig.add_subplot(gs[0, 1])
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
    ymin_B, ymax_B = _finite_min_max(y_curves_B, default_ymin=0.0, default_ymax=1.0, pad_ratio=0.08)
    axB.set_ylim(ymin_B, ymax_B)
    axB.legend(loc="best")
    axB.grid(True, alpha=0.3)

    # Panel C: stability (Jaccard) vs margin
    axC = fig.add_subplot(gs[1, 0])
    y_curves_C = []
    for k in windows:
        sel = summary_df[summary_df["window_k"] == k].sort_values("margin")
        y = sel["jaccard_vs_base"].values
        axC.plot(sel["margin"], y, marker="o", label=f"Jaccard (k={int(k)})")
        y_curves_C.append(y)
    axC.set_title(f"(C) Stability vs margin\n{tile_label}")
    axC.set_xlabel("Neighbor margin (quantile units)")
    axC.set_ylabel("Jaccard vs baseline")
    _set_dynamic_xaxis(axC, summary_df["margin"].values)
    ymin_C, ymax_C = _finite_min_max(y_curves_C, default_ymin=0.0, default_ymax=1.0, pad_ratio=0.08)
    axC.set_ylim(ymin_C, ymax_C)
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
    margins: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03),
    windows: Tuple[int, ...] = (1, 2),
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
               "sep_fraction_base": "median",
               "sep_fraction_exp": "median",
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
    parser.add_argument("--windows", type=str, default="1,2",
                        help="Comma-separated window sizes (k)")
    parser.add_argument("--reshape", action="store_true",
                        help="If set, expects --series to be a wide all_years.csv and reshapes it")

    args = parser.parse_args()

    # parse tuples
    margins = tuple(float(x) for x in args.margins.split(",") if x)
    windows = tuple(int(x) for x in args.windows.split(",") if x)

    # Example paths (edit or pass via CLI)
    if args.series is None:
        ALL_YEARS_CSV = r"D:\\Research\\FS-2dot0\\results\\newtop5\\2000-2023\\all_years.csv"
        # default: reshape wide -> long
        series_df = reshape_all_years_wide_to_long(
            all_years_csv=ALL_YEARS_CSV,
            years=range(1990, 2024),
            counts_are_fractions=False,
        )
    else:
        if args.reshape:
            series_df = reshape_all_years_wide_to_long(
                all_years_csv=args.series,
                years=range(1990, 2024),
                counts_are_fractions=False,
            )
        else:
            series_df = pd.read_csv(args.series)

    if args.just is None:
        JUST_CSV = r"D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\1990-2023percentile_justification.csv"
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
