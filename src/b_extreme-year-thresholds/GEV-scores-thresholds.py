
# --- add these lines at the very top of your module, BEFORE importing pyplot ---
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in multiprocessing
import matplotlib.pyplot as plt

import os
import csv
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import concurrent.futures

#for small or zero values:

from statsmodels.tsa.stattools import acf
from scipy.stats import genextreme, kendalltau

# --- New function: Compute and plot ACF ---
def compute_and_plot_acf(score_series: pd.Series, tile_id: str, max_lag: int = 10, plot_dir: Optional[str] = None) -> np.ndarray:
    s = score_series.dropna()
    # If not enough data or variance is ~0, return NaNs rather than calling acf()
    if len(s) < 2 or np.nanstd(s.values) < 1e-12:  # tolerance to treat constant series
        acf_values = np.full(max_lag + 1, np.nan, dtype=float)
    else:
        acf_values = acf(s, nlags=max_lag, fft=False)

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 4))
        # Replace NaNs with 0 for pretty plotting, but keep the returned array as-is
        plot_vals = np.nan_to_num(acf_values, nan=0.0)
        plt.stem(range(len(plot_vals)), plot_vals)
        plt.xlabel('Lag (years)')
        plt.ylabel('Autocorrelation')
        plt.title(f'ACF of FS score for tile {tile_id}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'acf_tile_{tile_id}.png'))
        plt.close()
    return acf_values




# --- Helper functions ---
def wide_to_long(df_wide: pd.DataFrame, year_start: int = 1990, year_end: int = 2023, col_suffixes=('lt', 'mid', 'gt2', 'gte')) -> pd.DataFrame:
    rows = []
    for _, r in df_wide.iterrows():
        tile = r['tile_id']
        for y in range(year_start, year_end + 1):
            entry = {'tile_id': tile, 'year': y}
            for s in col_suffixes:
                col = f"{y}_{s}"
                entry[s] = r.get(col, np.nan)
            rows.append(entry)
    return pd.DataFrame(rows)

def compute_percentages(df_long: pd.DataFrame) -> pd.DataFrame:
    d = df_long.copy()
    d['N_total'] = d['lt'] + d['gte']
    d['pct_dry'] = np.where(d['N_total'] > 0, d['lt'] / d['N_total'], np.nan)
    d['pct_wet'] = np.where(d['N_total'] > 0, d['gte'] / d['N_total'], np.nan)
    d['pct_very_wet'] = np.where(d['N_total'] > 0, d['gt2'] / d['N_total'], np.nan)
    return d

def compute_scores(d: pd.DataFrame, w_wet=1.0, w_vwet=2.0, w_dry=1.0, alpha=2.0, beta=1.0) -> pd.DataFrame:
    x = d.copy()
    x['score_wet'] = (w_wet * x['pct_wet']) + (w_vwet * x['pct_very_wet']) - (w_dry * x['pct_dry'])
    x['score_dry'] = x['pct_dry'] - alpha * x['pct_very_wet'] - beta * x['pct_wet']
    return x

def mann_kendall(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 3:
        return float('nan'), float('nan')
    tau, p = kendalltau(s.index.values, s.values)
    return float(tau), float(p)

def sse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    mu = np.mean(y)
    return float(np.sum((y - mu) ** 2))

def best_split(y: np.ndarray, min_seg_len: int) -> tuple[Optional[int], float]:
    n = y.size
    if n < 2 * min_seg_len:
        return None, 0.0
    total = sse(y)
    best_k = None
    best_cost = float('inf')
    for k in range(min_seg_len, n - min_seg_len + 1):
        cost = sse(y[:k]) + sse(y[k:])
        if cost < best_cost:
            best_cost = cost
            best_k = k
    improvement = total - best_cost
    return best_k, improvement

def binary_segmentation(y: np.ndarray, years: np.ndarray, min_seg_len: int = 3, penalty_lambda: float = 5.0) -> list[int]:
    n = y.size
    cps = []
    segments = [(0, n)]
    while True:
        best_impr = 0.0
        best_seg = None
        best_k_global = None
        for (a, b) in segments:
            yy = y[a:b]
            k_rel, impr = best_split(yy, min_seg_len)
            if k_rel is None:
                continue
            k_abs = a + k_rel
            if impr > best_impr:
                best_impr = impr
                best_seg = (a, b)
                best_k_global = k_abs
        if best_k_global is None:
            break
        penalty = penalty_lambda * np.log(n)
        if best_impr <= penalty:
            break
        cps.append(int(years[best_k_global]))
        a, b = best_seg
        segments.remove(best_seg)
        segments.append((a, best_k_global))
        segments.append((best_k_global, b))
        cps.sort()
    return cps


def gev_return_level(y: np.ndarray, R: int) -> tuple[float, tuple[float, float, float]]:
    y = y[~np.isnan(y)]
    # Need enough data AND non-zero spread
    if len(y) < 5 or np.nanstd(y) < 1e-12:
        return float('nan'), (float('nan'), float('nan'), float('nan'))

    # Optional: clip extremes / add tiny jitter if truly identical values are common
    # y = y + np.random.default_rng().normal(0, 1e-9, size=len(y))  # only if needed

    try:
        c, loc, scale = genextreme.fit(y)
        p = 1.0 - 1.0 / float(R)
        rl = float(genextreme.ppf(p, c, loc=loc, scale=scale))
        return rl, (c, loc, scale)
    except Exception:
        # Any fit failure returns NaNs
        return float('nan'), (float('nan'), float('nan'), float('nan'))


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    b = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=n_blocks) if n - b + 1 > 0 else np.zeros(n_blocks, dtype=int)
    idx = np.concatenate([np.arange(s, s + b) for s in starts])
    return idx[:n]

def bootstrap_gev_threshold(y: np.ndarray, R: int, n_boot: int = 500, block_size: int = 3, random_state: Optional[int] = None) -> tuple[float, float, float]:
    y = y[~np.isnan(y)]
    if len(y) < 5:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(random_state)
    qs = []
    for _ in range(n_boot):
        idx = block_bootstrap_indices(len(y), block_size, rng)
        y_resamp = y[idx]
        rl, _ = gev_return_level(y_resamp, R)
        if np.isnan(rl):
            continue
        q_emp = np.mean(y <= rl)
        qs.append(q_emp)
    qs = np.array(qs)
    return float(np.nanmedian(qs)), float(np.nanpercentile(qs, 2.5)), float(np.nanpercentile(qs, 97.5))

def expand_neighbors(scores_by_year: dict[int, float], selected_years: set[int], threshold_value: float, direction: str, margin: float = 0.02) -> set[int]:
    expanded = set(selected_years)
    for y in list(selected_years):
        for ny in (y - 1, y + 1):
            if ny in scores_by_year:
                s = scores_by_year[ny]
                if direction == 'high' and s >= threshold_value - margin:
                    expanded.add(ny)
                elif direction == 'low' and s <= threshold_value + margin:
                    expanded.add(ny)
    return expanded



# --- New function: Sensitivity analysis over block sizes ---
def bootstrap_sensitivity_analysis(
    s: np.ndarray,
    R_return_period: int,
    block_sizes: List[int],
    n_boot: int,
    random_state: Optional[int] = None
) -> Dict[int, tuple]:
    """
    Run bootstrap_gev_threshold for multiple block sizes and return dict of results.
    """
    results = {}
    for bsize in block_sizes:
        q_med, q_lo, q_hi = bootstrap_gev_threshold(s, R_return_period, n_boot=n_boot, block_size=bsize, random_state=random_state)
        results[bsize] = (q_med, q_lo, q_hi)
    return results

# --- Updated parallel tile processing function with ACF and sensitivity ---
def process_tile_extended(tile_id, g, years, weights, R_return_period, n_boot, block_sizes, neighbor_margin, random_state, acf_plot_dir=None):
    g = g.sort_values('year')
    s_w = g['score_wet'].to_numpy()
    s_d = g['score_dry'].to_numpy()
    # Compute ACF and optionally plot
    s_w_series = pd.Series(s_w, index=years)
    s_d_series = pd.Series(s_d, index=years)
    acf_w = compute_and_plot_acf(s_w_series, tile_id, plot_dir=acf_plot_dir)
    acf_d = compute_and_plot_acf(s_d_series, tile_id, plot_dir=acf_plot_dir)
    # GEV return levels
    rl_w, _ = gev_return_level(s_w, R_return_period)
    rl_d, _ = gev_return_level(s_d, R_return_period)
    # Bootstrap sensitivity analysis for wet and dry scores
    sens_w = bootstrap_sensitivity_analysis(s_w, R_return_period, block_sizes, n_boot, random_state)
    sens_d = bootstrap_sensitivity_analysis(s_d, R_return_period, block_sizes, n_boot, random_state)
    # Select years exceeding threshold (using block_size = block_sizes[0] as default)
    q_w_med, _, _ = sens_w[block_sizes[0]]
    q_d_med, _, _ = sens_d[block_sizes[0]]
    wet_sel = set(years[s_w >= rl_w])
    dry_sel = set(years[s_d >= rl_d])

    """  Remove the “select all years” fallback
    if not wet_sel:
        wet_sel = set(years)
    if not dry_sel:
        dry_sel = set(years) """
            
    ##If no exceedance, use a small top-K fallback (or leave empty)
    K_fallback = 3
    if not wet_sel:
        # select top-K years by wet score
        top_idx = np.argsort(s_w)[-K_fallback:]
        wet_sel = set(years[top_idx])
    if not dry_sel:
        top_idx = np.argsort(s_d)[-K_fallback:]
        dry_sel = set(years[top_idx])

    #######################################
        # When dry_sel is empty OR too large, pick a controlled Top-K subset
    if (not dry_sel) or (len(dry_sel) > 12):

        # Prefer selecting from non-wet years
        candidates = [y for y in years if y not in wet_sel]

        if candidates:
            s_d_nonwet = np.array([
                s_d[np.where(years == y)[0][0]] for y in candidates
            ])

            # Select Top-K driest among candidates
            K = min(K_fallback, len(candidates))
            top_idx_local = np.argsort(s_d_nonwet)[-K:]

            dry_sel = set(candidates[i] for i in top_idx_local)
            used_topk_dry = True

        else:
            # If everything is wet (rare), fall back to global Top-K
            K = min(K_fallback, len(years))
            top_idx = np.argsort(s_d)[-K:]

            dry_sel = set(years[top_idx])
            used_topk_dry = True




    # Neighbor expansion
    wet_neighbors = sorted(list(expand_neighbors(dict(zip(years,s_w)), wet_sel, rl_w, 'high', neighbor_margin)))
    dry_neighbors = sorted(list(expand_neighbors(dict(zip(years,s_d)), dry_sel, rl_d, 'high', neighbor_margin)))
    dry_sel -= wet_sel  # Prioritize wet if overlap
    dry_neighbors = [y for y in dry_neighbors if y not in wet_sel]
    
    # Change-points
    cps_wet = binary_segmentation(s_w, years)
    cps_dry = binary_segmentation(s_d, years)
    # Mann-Kendall
    tau_w, p_w = mann_kendall(s_w_series)
    tau_d, p_d = mann_kendall(s_d_series)
    return {f
        'tile_id': tile_id,
        'recommended_q_wet': q_w_med,
        'RL_target_wet': rl_w,
        'selected_wet_years': ';'.join(map(str, sorted(list(wet_sel)))),
        'selected_wet_years_neighbors': ';'.join(map(str, wet_neighbors)),
        'change_points_wet': ';'.join(map(str, cps_wet)),
        'mk_tau_wet': tau_w,
        'mk_p_wet': p_w,
        'recommended_q_dry': q_d_med,
        'RL_target_dry': rl_d,
        'selected_dry_years': ';'.join(map(str, sorted(list(dry_sel)))),
        'selected_dry_years_neighbors': ';'.join(map(str, dry_neighbors)),
        'change_points_dry': ';'.join(map(str, cps_dry)),
        'mk_tau_dry': tau_d,
        'mk_p_dry': p_d,
        'bootstrap_sensitivity_wet': sens_w,
        'bootstrap_sensitivity_dry': sens_d,
        'acf_wet': acf_w,
        'acf_dry': acf_d
    }

# --- Updated main driver with parallel processing and new features ---
def run_selection_robust_parallel_extended(
    df_wide: pd.DataFrame,
    outfolder: str,
    year_start: int = 1990,
    year_end: int = 2023,
    R_return_period: int = 10,
    n_boot: int = 200,
    block_sizes: List[int] = [2, 3, 5],
    weights: dict = {'w_wet':1.0,'w_vwet':2.0,'w_dry':1.0,'alpha':2.0,'beta':1.0},
    neighbor_margin: float = 0.02,
    random_state: Optional[int] = 42,
    export_config: bool = True,
    acf_plot_dir: Optional[str] = None
) -> str:
    os.makedirs(outfolder, exist_ok=True)
    summary_path = os.path.join(outfolder, 'percentile_justification.csv')
    df_long = wide_to_long(df_wide, year_start, year_end)
    df_long = compute_percentages(df_long)
    df_long = compute_scores(df_long, **weights)
    years = np.arange(year_start, year_end + 1)
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for tile_id, g in df_long.groupby('tile_id', sort=False):
            futures.append(executor.submit(process_tile_extended, tile_id, g, years, weights, R_return_period, n_boot, block_sizes, neighbor_margin, random_state, acf_plot_dir))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    # Write results to CSV
    # Flatten bootstrap sensitivity dicts for CSV output
    fieldnames = [
        'tile_id', 'recommended_q_wet', 'RL_target_wet', 'selected_wet_years', 'selected_wet_years_neighbors',
        'change_points_wet', 'mk_tau_wet', 'mk_p_wet',
        'recommended_q_dry', 'RL_target_dry', 'selected_dry_years', 'selected_dry_years_neighbors',
        'change_points_dry', 'mk_tau_dry', 'mk_p_dry'
    ]
    # Add bootstrap sensitivity columns for each block size
    for bsize in block_sizes:
        fieldnames.append(f'bootstrap_q_wet_b{bsize}')
        fieldnames.append(f'bootstrap_q_wet_lo_b{bsize}')
        fieldnames.append(f'bootstrap_q_wet_hi_b{bsize}')
        fieldnames.append(f'bootstrap_q_dry_b{bsize}')
        fieldnames.append(f'bootstrap_q_dry_lo_b{bsize}')
        fieldnames.append(f'bootstrap_q_dry_hi_b{bsize}')
    with open(summary_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames if k in r}
            # Add bootstrap sensitivity values
            for bsize in block_sizes:
                q_w, q_w_lo, q_w_hi = r['bootstrap_sensitivity_wet'][bsize]
                q_d, q_d_lo, q_d_hi = r['bootstrap_sensitivity_dry'][bsize]
                row[f'bootstrap_q_wet_b{bsize}'] = q_w
                row[f'bootstrap_q_wet_lo_b{bsize}'] = q_w_lo
                row[f'bootstrap_q_wet_hi_b{bsize}'] = q_w_hi
                row[f'bootstrap_q_dry_b{bsize}'] = q_d
                row[f'bootstrap_q_dry_lo_b{bsize}'] = q_d_lo
                row[f'bootstrap_q_dry_hi_b{bsize}'] = q_d_hi
            w.writerow(row)
    if export_config:
        cfg_path = os.path.join(outfolder, 'recommended_q_config.csv')
        pd.DataFrame([{k: r[k] for k in ['tile_id', 'recommended_q_wet', 'recommended_q_dry']} for r in results]).to_csv(cfg_path, index=False)
        print(f"[DONE] Wrote config: {cfg_path}")
    return os.path.abspath(summary_path)

# Example usage
if __name__ == '__main__':
    #import sys
    #incsv = sys.argv[1] if len(sys.argv) > 1 else "all_years.csv"
    #outfolder = sys.argv[2] if len(sys.argv) > 2 else "JustificationRobust_parallel"
    #acf_dir = sys.argv[3] if len(sys.argv) > 3 else "acf_plots"
    incsv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\data\\all_years_99.csv"
    #ncsv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\data\\all_years-combined-final-sample.csv"
    outcsv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\results\\99-wp15-change"
    #incsv = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/extremeyears/data/all_years-combined-final.csv"
    #outcsv = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/extremeyears/results"

    acf_dir = outcsv + "\\_acf_plots"

    df = pd.read_csv(incsv)
    summary_path = run_selection_robust_parallel_extended(
        df_wide=df,
        outfolder=outcsv,
        year_start=2000,
        year_end=2023,
        R_return_period=10,
        n_boot=300,
        block_sizes=[ 3, 2, 5, 10],
        weights={'w_wet':1.0,'w_vwet':1.5,'w_dry':1.0,'alpha':2.0,'beta':1.0},
        neighbor_margin=0.02,
        random_state=42,
        export_config=True,
        acf_plot_dir=acf_dir
    )
    print("Finished. Summary written to:", summary_path)
