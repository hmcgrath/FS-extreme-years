CSV_PATH = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\1990-2023percentile_justification.csv"  # <-- set to your justification CSV path
Q_CONFIG_PATH = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\recommended_q_config.csv"  # <-- set to your justification CSV path
# validation_diagnostics_tailored.py
# Purpose: Generate a 2x2 validation & diagnostics panel using your exact column names.
# Works with justification CSV alone; optionally uses a per-year series CSV for QQ/PP and ACF/PACF.
# Requirements: pandas, numpy, matplotlib, seaborn, scipy, statsmodels

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import genextreme, rankdata, kendalltau
from statsmodels.tsa.stattools import acf, pacf

# ----------------------------
# User-configurable parameters
# ----------------------------

JUSTIFICATION_CSV_PATH = CSV_PATH          # <-- your justification table (columns listed below)
RECOMMENDED_Q_CONFIG_PATH = "recommended_q_config.csv"   # <-- optional overrides: tile_id, recommended_q_wet, recommended_q_dry
SERIES_CSV_PATH = None                                   # <-- optional per-year series CSV; set to path if available

# Tile to highlight in panels (a), (b), and (d)
TILE_TO_PLOT = "05OG000"

# Columns expected in justification.csv
JUST_COLS = [
    "tile_id", "recommended_q_wet", "RL_target_wet", "selected_wet_years", "selected_wet_years_neighbors",
    "change_points_wet", "mk_tau_wet", "mk_p_wet",
    "recommended_q_dry", "RL_target_dry", "selected_dry_years", "selected_dry_years_neighbors",
    "change_points_dry", "mk_tau_dry", "mk_p_dry",
    "bootstrap_q_wet_b3", "bootstrap_q_wet_lo_b3", "bootstrap_q_wet_hi_b3",
    "bootstrap_q_dry_b3", "bootstrap_q_dry_lo_b3", "bootstrap_q_dry_hi_b3",
    "bootstrap_q_wet_b5", "bootstrap_q_wet_lo_b5", "bootstrap_q_wet_hi_b5",
    "bootstrap_q_dry_b5", "bootstrap_q_dry_lo_b5", "bootstrap_q_dry_hi_b5",
    "bootstrap_q_wet_b10", "bootstrap_q_wet_lo_b10", "bootstrap_q_wet_hi_b10",
    "bootstrap_q_dry_b10", "bootstrap_q_dry_lo_b10", "bootstrap_q_dry_hi_b10"
]

# If you have a per-year series CSV, specify its columns here:
SERIES_COLS = {
    "tile_id": "tile_id",
    "year": "year",
    "wet_score": "wet_score",       # annual watershed wet score
    "dry_score": "dry_score",       # annual watershed dry score (optional if you only plot wet)
    "wet_fraction": "wet_fraction", # pixel exceedance fraction per year (optional)
    "fs_median": "fs_median"        # per-year FS median (optional)
}

# Plot aesthetics
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14

# ----------------------------
# Utility functions
# ----------------------------

def safe_load_csv(path, required_subset=None):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if required_subset is not None:
        missing = set(required_subset) - set(df.columns)
        if missing:
            raise ValueError(f"CSV '{path}' is missing required columns: {missing}")
    return df

def parse_year_list(cell):
    """
    Parse a cell that may contain a list-like string of years, e.g. "[2018,2019]" or "2018;2019" or "2018, 2019".
    Returns sorted list[int].
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    # Try AST literal_eval first (handles Python list strings)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return sorted([int(x) for x in val])
    except Exception:
        pass
    # Fallback: split on common delimiters
    for sep in [";", ",", "|", " "]:
        if sep in s:
            try:
                return sorted([int(x) for x in s.split(sep) if x.strip().isdigit()])
            except Exception:
                continue
    # Single int?
    if s.isdigit():
        return [int(s)]
    return []

def benjamini_hochberg(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR control.
    Returns adjusted p-values (same order) and boolean array of significance at alpha.
    """
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj_ranked = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        adj_p = ranked[i] * n / rank
        cummin = min(cummin, adj_p)
        adj_ranked[i] = cummin
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adj_ranked
    significant = adjusted <= alpha
    return adjusted, significant

def fit_gev(series_1d):
    x = np.asarray(series_1d)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Insufficient data points (<10) to fit GEV reliably.")
    c, loc, scale = genextreme.fit(x)  # SciPy parameterization
    return c, loc, scale

def qq_pp_data(series_1d, gev_params, n_points=200):
    x = np.asarray(series_1d)
    x = x[np.isfinite(x)]
    emp_q = np.quantile(x, np.linspace(0.01, 0.99, n_points))
    c, loc, scale = gev_params
    theo_q = genextreme.ppf(np.linspace(0.01, 0.99, n_points), c, loc=loc, scale=scale)
    ranks = rankdata(x, method="average")
    emp_cdf = ranks / (len(x) + 1.0)
    model_cdf = genextreme.cdf(np.sort(x), c, loc=loc, scale=scale)
    return {"qq_x": theo_q, "qq_y": emp_q, "pp_x": model_cdf, "pp_y": np.sort(emp_cdf)}

def acf_pacf_data(series_1d, nlags=10):
    x = np.asarray(series_1d)
    x = x[np.isfinite(x)]
    x = x - np.nanmean(x)
    acf_vals = acf(x, nlags=nlags, fft=False, missing='drop')
    pacf_vals = pacf(x, nlags=nlags, method='ywunbiased')
    return acf_vals, pacf_vals

# ----------------------------
# Load justification and config
# ----------------------------

df_just = safe_load_csv(JUSTIFICATION_CSV_PATH, required_subset=JUST_COLS)
if df_just is None:
    raise FileNotFoundError(f"Justification CSV not found: {JUSTIFICATION_CSV_PATH}")

df_cfg = safe_load_csv(RECOMMENDED_Q_CONFIG_PATH)  # optional
if df_cfg is not None:
    # Merge overrides (if present)
    df_just = df_just.merge(df_cfg[["tile_id", "recommended_q_wet", "recommended_q_dry"]],
                            on="tile_id", how="left", suffixes=("", "_cfg"))
    # Prefer config values when available (non-null)
    for col in ["recommended_q_wet", "recommended_q_dry"]:
        cfg_col = f"{col}_cfg"
        if cfg_col in df_just.columns:
            df_just[col] = np.where(df_just[cfg_col].notna(), df_just[cfg_col], df_just[col])
            df_just.drop(columns=[cfg_col], inplace=True)

# Extract the tile row to plot
row = df_just.loc[df_just["tile_id"] == TILE_TO_PLOT]
if row.empty:
    raise ValueError(f"TILE_TO_PLOT='{TILE_TO_PLOT}' was not found in justification CSV.")

row = row.iloc[0]

# Parse year lists
wet_years = parse_year_list(row["selected_wet_years"])
wet_neighbors = parse_year_list(row["selected_wet_years_neighbors"])

# ----------------------------
# Optionally load per-year series for QQ/PP, ACF/PACF, and panel (d)
# ----------------------------

df_series = safe_load_csv(SERIES_CSV_PATH)
have_series = df_series is not None

if have_series:
    # Check columns
    needed = [SERIES_COLS["tile_id"], SERIES_COLS["year"], SERIES_COLS["wet_score"]]
    missing_series = set(needed) - set(df_series.columns)
    if missing_series:
        raise ValueError(f"Series CSV missing columns: {missing_series}")

    df_ws = (df_series
             .loc[df_series[SERIES_COLS["tile_id"]] == TILE_TO_PLOT,
                  [SERIES_COLS["year"], SERIES_COLS["wet_score"],
                   SERIES_COLS.get("wet_fraction", None),
                   SERIES_COLS.get("fs_median", None)]]
             .rename(columns={
                 SERIES_COLS["year"]: "year",
                 SERIES_COLS["wet_score"]: "wet_score",
                 SERIES_COLS.get("wet_fraction", "wet_fraction"): "wet_fraction",
                 SERIES_COLS.get("fs_median", "fs_median"): "fs_median"
             }))
    df_ws.sort_values("year", inplace=True)
else:
    df_ws = None

# ----------------------------
# Build the figure canvas
# ----------------------------

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.28)

# Panel (a): QQ–PP OR bootstrap quantile CIs
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = ax_a1.inset_axes([0.53, 0.12, 0.44, 0.76])  # PP inset

if have_series and df_ws is not None and df_ws["wet_score"].notna().sum() >= 10:
    # Fit GEV and produce QQ/PP
    params = fit_gev(df_ws["wet_score"].values)
    qqpp = qq_pp_data(df_ws["wet_score"].values, params)
    # QQ plot
    ax_a1.plot(qqpp["qq_x"], qqpp["qq_y"], color="tab:blue", lw=2, label="Empirical vs GEV")
    ax_a1.plot([min(qqpp["qq_x"]), max(qqpp["qq_x"])],
               [min(qqpp["qq_x"]), max(qqpp["qq_x"])], color="gray", lw=1, ls="--", label="1:1")
    ax_a1.set_title(f"(a) GEV QQ–PP | {TILE_TO_PLOT}")
    ax_a1.set_xlabel("GEV theoretical quantiles")
    ax_a1.set_ylabel("Empirical quantiles")
    ax_a1.legend(loc="lower right")
    # PP plot
    ax_a2.plot(qqpp["pp_x"], qqpp["pp_y"], color="tab:green", lw=2, label="Empirical vs GEV CDF")
    ax_a2.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="1:1")
    ax_a2.set_xlabel("Model CDF")
    ax_a2.set_ylabel("Empirical CDF")
    ax_a2.legend(loc="lower right")
else:
    # Fallback: show bootstrap quantile CIs (wet) across block sizes
    q_wet = row["recommended_q_wet"]
    # Each block size: (median, lo, hi)
    block_data = [
        ("b3", row["bootstrap_q_wet_b3"], row["bootstrap_q_wet_lo_b3"], row["bootstrap_q_wet_hi_b3"]),
        ("b5", row["bootstrap_q_wet_b5"], row["bootstrap_q_wet_lo_b5"], row["bootstrap_q_wet_hi_b5"]),
        ("b10", row["bootstrap_q_wet_b10"], row["bootstrap_q_wet_lo_b10"], row["bootstrap_q_wet_hi_b10"]),
    ]
    xs, meds, los, his = [], [], [], []
    for label, med, lo, hi in block_data:
        xs.append(label)
        meds.append(med)
        los.append(lo)
        his.append(hi)
    ax_a1.errorbar(xs, meds, yerr=[np.array(meds)-np.array(los), np.array(his)-np.array(meds)],
                   fmt="o", color="tab:blue", ecolor="tab:blue", capsize=6, label="Bootstrap q (median ± 95% CI)")
    ax_a1.axhline(q_wet, color="black", ls="--", lw=1.2, label=f"recommended_q_wet={q_wet:.3f}")
    ax_a1.set_ylim(0, 1)
    ax_a1.set_title(f"(a) Bootstrap quantile CIs (wet) | {TILE_TO_PLOT}")
    ax_a1.set_xlabel("Block size")
    ax_a1.set_ylabel("Quantile level q")
    ax_a1.legend(loc="lower right")
    ax_a2.axis("off")

# Panel (b): ACF/PACF OR neighbor-year expansion summary
ax_b = fig.add_subplot(gs[0, 1])

if have_series and df_ws is not None and df_ws["wet_score"].notna().sum() >= 10:
    acf_vals, pacf_vals = acf_pacf_data(df_ws["wet_score"].values, nlags=10)
    lags = np.arange(len(acf_vals))
    width = 0.35
    ax_b.bar(lags - width/2, acf_vals, width=width, color="tab:blue", label="ACF")
    ax_b.bar(lags + width/2, pacf_vals, width=width, color="tab:orange", label="PACF")
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_xticks(lags)
    ax_b.set_title(f"(b) ACF/PACF (lags 0–{len(lags)-1}) | {TILE_TO_PLOT}")
    ax_b.set_xlabel("Lag")
    ax_b.set_ylabel("Correlation")
    ax_b.legend(loc="upper right")
else:
    # Fallback: summarize neighbor-year expansion (wet)
    y = wet_years
    yn = wet_neighbors
    ax_b.eventplot([y, yn], colors=["tab:blue", "tab:orange"], lineoffsets=[1, 0.5], linelengths=[0.4, 0.4])
    ax_b.set_title(f"(b) Neighbor-year expansion (wet) | {TILE_TO_PLOT}")
    ax_b.set_xlabel("Year")
    ax_b.set_yticks([1, 0.5])
    ax_b.set_yticklabels(["Selected wet years", "Neighbor-expanded"])
    if len(y) > 0:
        ax_b.set_xlim(min(y) - 1, max(y) + 1)

# Panel (c): MK with FDR control across all tiles
ax_c = fig.add_subplot(gs[1, 0])

mk_df = df_just[["tile_id", "mk_p_wet", "mk_tau_wet"]].dropna()
mk_df["p_adj"], mk_df["sig"] = benjamini_hochberg(mk_df["mk_p_wet"].values, alpha=0.05)

ax_c.scatter(mk_df["mk_p_wet"], mk_df["p_adj"], s=18, alpha=0.7, color="tab:purple")
ax_c.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
ax_c.axhline(0.05, color="red", lw=1, ls=":", label="α=0.05 (adjusted)")
ax_c.set_title("(c) Field significance: raw vs FDR-adjusted p-values (wet)")
ax_c.set_xlabel("Raw p-values (mk_p_wet)")
ax_c.set_ylabel("FDR-adjusted p-values")
ax_c.legend(loc="lower right")

# Panel (d): Pixel exceedance validation OR timeline vs RL target
ax_d = fig.add_subplot(gs[1, 1])

if have_series and df_ws is not None and ("wet_fraction" in df_ws.columns) and df_ws["wet_fraction"].notna().sum() > 0:
    years = df_ws["year"].values
    wet_frac = df_ws["wet_fraction"].values
    fs_med = df_ws["fs_median"].values if "fs_median" in df_ws.columns else np.full_like(wet_frac, np.nan, dtype=float)
    ax_d.plot(years, wet_frac, color="tab:blue", lw=2, label="Wet fraction (left)")
    ax_d2 = ax_d.twinx()
    ax_d2.plot(years, fs_med, color="black", lw=1.5, label="FS median (right)")
    # Shade selected wet years
    for yr in wet_years:
        ax_d.axvspan(yr-0.5, yr+0.5, color="tab:blue", alpha=0.15)
    title_d = f"(d) Pixel exceedance validation | {TILE_TO_PLOT}"
    ax_d.set_title(title_d)
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Wet fraction")
    ax_d2.set_ylabel("FS median")
    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
else:
    # Fallback: timeline of selected extremes vs RL target and recommended quantile
    y = wet_years
    yn = wet_neighbors
    ax_d.eventplot([y, yn], colors=["tab:blue", "tab:orange"], lineoffsets=[1, 0.5], linelengths=[0.4, 0.4])
    ax_d.set_title(f"(d) Selected wet extremes & neighbors | {TILE_TO_PLOT}\n"
                   f"RL_target_wet={row['RL_target_wet']}, recommended_q_wet={row['recommended_q_wet']:.3f}")
    ax_d.set_xlabel("Year")
    ax_d.set_yticks([1, 0.5])
    ax_d.set_yticklabels(["Selected wet years", "Neighbor-expanded"])
    if len(y) > 0:
        ax_d.set_xlim(min(y) - 1, max(y) + 1)

# Final layout and save
plt.suptitle("Validation & Diagnostics Panel", y=0.98, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = f"validation_diagnostics_{TILE_TO_PLOT}.png"
plt.savefig(out_path, dpi=300)
print(f"Saved figure to: {out_path}")
