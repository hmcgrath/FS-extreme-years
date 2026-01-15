
# diagnostics_from_wide_series.py
# Build per-year long-format series from wide 'all_years.csv', compute fractions & scores,
# then generate Validation & Diagnostics figure tailored to your justification schema.

import os
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from scipy.stats import genextreme, rankdata, kendalltau
from statsmodels.tsa.stattools import acf, pacf

#########
'''
wet_score → dry_score
wet_fraction → dry_fraction
selected_wet_years → selected_dry_years
recommended_q_wet → recommended_q_dry
RL_target_wet → RL_target_dry
'''
""" from pathlib import Path

try:
    script_path = Path(__file__).resolve()
except NameError:
    # Interactive shell fallback: assume CWD is script directory
    script_path = Path.cwd()

config_file = (
    script_path.parent
    / "configs"
    / f"{script_path.stem}.json"
) """
# -------------------------------   
# Config loader
# -------------------------------
config_file = "D://Research//FS-2dot0/results//WetDryTrendsPaper//scripts//configs//compute-scores.json"

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return json.load(f)
    


# validation_diagnostics_from_wide.py
# ----------------------------------------------------------------------------- 
# Purpose: Build per-year long-format series from wide 'all_years.csv', compute
#          wet/dry fractions & scores, then generate a 2x2 Validation &
#          Diagnostics figure tailored to the user's justification schema.
# Author: M365 Copilot (for Heather McGrath)
# Date: 2026-01-05
# -----------------------------------------------------------------------------
# Inputs required:
#   - ALL_YEARS_CSV: wide-format table with columns like '1990_lt','1990_mid',
#                    '1990_gt2','1990_gte', ... '2023_gte', plus 'tile_id'.
#   - JUSTIFICATION_CSV_PATH: table with columns (tile_id, recommended_q_wet,
#         RL_target_wet, selected_wet_years, selected_wet_years_neighbors,
#         mk_tau_wet, mk_p_wet, ... plus bootstrap quantile CI fields).
#   - RECOMMENDED_Q_CONFIG_PATH (optional): overrides for recommended_q_wet/dry.
# Output:
#   - validation_diagnostics_{TILE_TO_PLOT}.png
# -----------------------------------------------------------------------------

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import MaxNLocator
from scipy.stats import genextreme, rankdata
from statsmodels.tsa.stattools import acf, pacf


# If your wide columns are *fractions* already (not counts), set True.
COUNTS_ARE_FRACTIONS = False

# Weights used in your Methods
W_WET = 1.0
W_VWET = 1.5
W_DRY = 1.0
ALPHA = 2.0
BETA = 1.0

# Plot aesthetics
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14

# ----------------------------
# Helper functions
# ----------------------------
def parse_year_list(cell):
    """
    Parse year list strings such as '2020;2021' or '[2020, 2021]'.
    Returns a sorted list[int].
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    # Try literal list first
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return sorted(int(x) for x in val)
    except Exception:
        pass
    # Semicolon- or comma-delimited fallback
    for sep in [";", ",", "|", " "]:
        if sep in s:
            toks = [t.strip() for t in s.split(sep) if t.strip()]
            yrs = [int(t) for t in toks if t.isdigit()]
            return sorted(yrs)
    # Single integer?
    return [int(s)] if s.isdigit() else []

def benjamini_hochberg(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR control.
    Returns adjusted p-values and a boolean array of significance at alpha.
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
    c, loc, scale = genextreme.fit(x)  # SciPy: shape c, loc, scale
    return c, loc, scale

def qq_pp_data(series_1d, gev_params, n_points=200):
    x = np.asarray(series_1d)
    x = x[np.isfinite(x)]
    # Guard against constant series
    if np.nanstd(x) < 1e-9:
        raise ValueError("Series is near-constant; QQ/PP not informative.")
    grid = np.linspace(0.01, 0.99, n_points)
    emp_q = np.quantile(x, grid)
    c, loc, scale = gev_params
    theo_q = genextreme.ppf(grid, c, loc=loc, scale=scale)
    ranks = rankdata(x, method="average")
    emp_cdf = ranks / (len(x) + 1.0)
    model_cdf = genextreme.cdf(np.sort(x), c, loc=loc, scale=scale)
    return {"qq_x": theo_q, "qq_y": emp_q, "pp_x": model_cdf, "pp_y": np.sort(emp_cdf)}

def acf_pacf_data(series_1d, nlags=10):
    x = np.asarray(series_1d)
    x = x[np.isfinite(x)]
    # Ensure enough points
    if x.size < 3:
        raise ValueError("Series too short for ACF/PACF.")
    nlags_eff = int(min(nlags, max(1, x.size - 2)))
    x = x - np.nanmean(x)
    acf_vals = acf(x, nlags=nlags_eff, fft=False, missing='drop')
    # Statsmodels pacf method names changed across versions; choose robust default
    try:
        pacf_vals = pacf(x, nlags=nlags_eff, method='yw')
    except Exception:
        pacf_vals = pacf(x, nlags=nlags_eff, method='ywmle')
    return acf_vals, pacf_vals


def single_tile_plot(config_file, TILE_TO_PLOT):

    cfg = load_config(config_file)

    # ---- Paths ----
    ALL_YEARS_CSV = cfg["paths"]["input_csv_local"]
    JUSTIFICATION_CSV_PATH = cfg["paths"]["justification_local"]
    RECOMMENDED_Q_CONFIG_PATH = cfg["paths"]["recommended_q_local"]
    outdir = cfg["paths"]["output_dir_local"]


    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "validation_diagnostics")  # output figure path
    os.makedirs(outpath, exist_ok=True)
    TILE_TO_PLOT = TILE_TO_PLOT          # tile_id to render in panels (a), (b), (d)
    YEARS_RANGE = range(cfg["analysis"]["year_start"], cfg["analysis"]["year_end"] + 1)  # inclusive range


    # ----------------------------
    # 1) Load and reshape wide-format series
    # ----------------------------
    df_wide = pd.read_csv(ALL_YEARS_CSV, sep=None, engine="python")
    if "tile_id" not in df_wide.columns:
        raise ValueError("Expected 'tile_id' column in all_years.csv")

    # Build long-format records per year
    rows = []
    for _, r in df_wide.iterrows():
        tile_id = r["tile_id"]
        for yr in YEARS_RANGE:
            # Expect columns: f"{yr}_lt", f"{yr}_mid", f"{yr}_gt2", f"{yr}_gte"
            if not all(f"{yr}_{suffix}" in df_wide.columns for suffix in ["lt", "mid", "gt2", "gte"]):
                # Missing year columns → skip gracefully
                continue
            # Cast to float for safe math
            lt = float(r[f"{yr}_lt"]) if pd.notna(r[f"{yr}_lt"]) else np.nan
            mid = float(r[f"{yr}_mid"]) if pd.notna(r[f"{yr}_mid"]) else np.nan
            gt2 = float(r[f"{yr}_gt2"]) if pd.notna(r[f"{yr}_gt2"]) else np.nan
            gte = float(r[f"{yr}_gte"]) if pd.notna(r[f"{yr}_gte"]) else np.nan

            # Counts to fractions (exclude 'mid' per Methods)
            if COUNTS_ARE_FRACTIONS:
                N = max((lt if np.isfinite(lt) else 0.0) + (gte if np.isfinite(gte) else 0.0), 1e-9)
                pct_dry = (lt if np.isfinite(lt) else 0.0) / N
                pct_wet = (gte if np.isfinite(gte) else 0.0) / N
                pct_vwet = min((gt2 if np.isfinite(gt2) else 0.0) / N, pct_wet)
            else:
                if not (np.isfinite(lt) and np.isfinite(gte)):
                    pct_dry = np.nan
                    pct_wet = np.nan
                    pct_vwet = np.nan
                else:
                    N = lt + gte
                    if N <= 0:
                        pct_dry = np.nan
                        pct_wet = np.nan
                        pct_vwet = np.nan
                    else:
                        pct_dry = lt / N
                        pct_wet = gte / N
                        # Clip gt2 <= gte to avoid rounding artifacts
                        pct_vwet = np.nan
                        if np.isfinite(gt2):
                            pct_vwet = min(gt2 / N, pct_wet)

            # Scores
            if np.isnan(pct_dry) or np.isnan(pct_wet) or np.isnan(pct_vwet):
                wet_score = np.nan
                dry_score = np.nan
            else:
                wet_score = (W_WET * pct_wet) + (W_VWET * pct_vwet) - (W_DRY * pct_dry)
                dry_score = pct_dry - (ALPHA * pct_vwet) - (BETA * pct_wet)

            rows.append({
                "tile_id": tile_id,
                "year": int(yr),
                "lt": lt, "mid": mid, "gt2": gt2, "gte": gte,
                "wet_fraction": pct_wet,
                "dry_fraction": pct_dry,
                "wet_score": wet_score,
                "dry_score": dry_score
            })

    df_series = pd.DataFrame(rows).sort_values(["tile_id", "year"])\
                        .reset_index(drop=True)

    # ----------------------------
    # 2) Load justification and config overrides
    # ----------------------------
    df_just = pd.read_csv(JUSTIFICATION_CSV_PATH)

    # Optional overrides
    if os.path.exists(RECOMMENDED_Q_CONFIG_PATH):
        df_cfg = pd.read_csv(RECOMMENDED_Q_CONFIG_PATH)
        if not set(["tile_id", "recommended_q_wet", "recommended_q_dry"]).issubset(df_cfg.columns):
            raise ValueError("recommended_q_config.csv must contain tile_id, recommended_q_wet, recommended_q_dry")
        df_just = df_just.merge(df_cfg[["tile_id", "recommended_q_wet", "recommended_q_dry"]],
                                on="tile_id", how="left", suffixes=("", "_cfg"))
        # Prefer config overrides when present
        for col in ["recommended_q_wet", "recommended_q_dry"]:
            cfg_col = f"{col}_cfg"
            if cfg_col in df_just.columns:
                df_just[col] = np.where(df_just[cfg_col].notna(), df_just[col], df_just[col])
                df_just.drop(columns=[cfg_col], inplace=True)

    # Extract tile-specific row and selected years
    row = df_just.loc[df_just["tile_id"] == TILE_TO_PLOT]
    if row.empty:
        raise ValueError(f"TILE_TO_PLOT='{TILE_TO_PLOT}' not found in justification CSV.")
    row = row.iloc[0]

    wet_years = parse_year_list(row.get("selected_wet_years", np.nan))
    wet_neighbors = parse_year_list(row.get("selected_wet_years_neighbors", np.nan))

    # Tile series
    df_ws = df_series.loc[df_series["tile_id"] == TILE_TO_PLOT].copy()
    if df_ws.empty:
        raise ValueError(f"No per-year records found for tile_id='{TILE_TO_PLOT}' in all_years.csv")


    # ----------------------------
    # 3) Build the figure
    # ----------------------------
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)  # let constrained_layout handle spacing

    # ---- Panel (a): split into (a1) QQ and (a2) PP (side-by-side; no inset) ----
    ax_a1 = fig.add_subplot(gs[0, 0])  # (a1) QQ on the left
    ax_a2 = fig.add_subplot(gs[0, 1])  # (a2) PP on the right

    # Extract wet-score series for the tile
    series = df_ws["wet_score"].values
    series = series[np.isfinite(series)]
    has_enough = (series.size >= 10)
    has_variance = (np.nanstd(series) > 1e-9)

    if has_enough and has_variance:
        # Try GEV fit → QQ and PP
        fit_ok = False
        try:
            params = fit_gev(series)
            qqpp = qq_pp_data(series, params)
            fit_ok = True
        except Exception:
            fit_ok = False

        if fit_ok:
            # (a1) QQ plot
            ax_a1.plot(qqpp["qq_x"], qqpp["qq_y"], color="tab:blue", lw=2, label="Empirical vs GEV")
            ax_a1.plot([np.nanmin(qqpp["qq_x"]), np.nanmax(qqpp["qq_x"])],
                    [np.nanmin(qqpp["qq_x"]), np.nanmax(qqpp["qq_x"])],
                    color="gray", lw=1, ls="--", label="1:1")
            ax_a1.set_title(f"(a1) GEV QQ | {TILE_TO_PLOT}")
            ax_a1.set_xlabel("GEV theoretical quantiles")
            ax_a1.set_ylabel("Empirical quantiles")
            ax_a1.legend(loc="lower right")

            # (a2) PP plot (Empirical vs GEV CDF)
            ax_a2.plot(qqpp["pp_x"], qqpp["pp_y"], color="tab:green", lw=2, label="Empirical vs GEV CDF")
            ax_a2.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="1:1")
            ax_a2.set_title(f"(a2) GEV PP | {TILE_TO_PLOT}")
            ax_a2.set_xlabel("Model CDF")
            ax_a2.set_ylabel("Empirical CDF")
            ax_a2.set_xlim(0, 1)
            ax_a2.set_ylim(0, 1)
            ax_a2.legend(loc="lower right")
        else:
            # GEV fit failed → fallback: show bootstrap CI in (a1), PP fallback in (a2)
            q_wet = float(row.get("recommended_q_wet", np.nan))
            block_data = [
                ("b3", row.get("bootstrap_q_wet_b3", np.nan),
                    row.get("bootstrap_q_wet_lo_b3", np.nan),
                    row.get("bootstrap_q_wet_hi_b3", np.nan)),
                ("b5", row.get("bootstrap_q_wet_b5", np.nan),
                    row.get("bootstrap_q_wet_lo_b5", np.nan),
                    row.get("bootstrap_q_wet_hi_b5", np.nan)),
                ("b10", row.get("bootstrap_q_wet_b10", np.nan),
                        row.get("bootstrap_q_wet_lo_b10", np.nan),
                        row.get("bootstrap_q_wet_hi_b10", np.nan)),
            ]
            xs, meds, los, his = [], [], [], []
            for label, med, lo, hi in block_data:
                if pd.notna(med) and pd.notna(lo) and pd.notna(hi):
                    xs.append(label); meds.append(float(med)); los.append(float(lo)); his.append(float(hi))
            if xs:
                ax_a1.errorbar(xs, meds,
                            yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                            fmt="o", color="tab:blue", ecolor="tab:blue", capsize=6,
                            label="Bootstrap q (median ± 95% CI)")
            if pd.notna(q_wet):
                ax_a1.axhline(q_wet, color="black", ls="--", lw=1.2,
                            label=f"recommended_q_wet={q_wet:.3f}")
            ax_a1.set_ylim(0, 1)
            ax_a1.set_title(f"(a1) Bootstrap quantile CIs (wet) | {TILE_TO_PLOT}")
            ax_a1.set_xlabel("Block size")
            ax_a1.set_ylabel("Quantile level q")
            ax_a1.legend(loc="lower right")

            # PP fallback (Normal reference) so you still see Empirical vs Reference CDF
            if series.size >= 3:
                x = np.sort(series)
                ranks = (np.arange(1, x.size + 1)) / (x.size + 1.0)
                mu, sigma = np.nanmean(series), np.nanstd(series)
                sigma = sigma if sigma > 1e-12 else 1.0
                from scipy.stats import norm
                model_cdf_ref = norm.cdf(x, loc=mu, scale=sigma)

                ax_a2.plot(model_cdf_ref, ranks, color="tab:green", lw=2, label="Empirical vs Normal CDF (fallback)")
                ax_a2.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="1:1")
                ax_a2.set_title(f"(a2) PP (fallback to Normal) | {TILE_TO_PLOT}")
                ax_a2.set_xlabel("Reference CDF")
                ax_a2.set_ylabel("Empirical CDF")
                ax_a2.set_xlim(0, 1)
                ax_a2.set_ylim(0, 1)
                ax_a2.legend(loc="lower right")
            else:
                ax_a2.text(0.5, 0.5, "PP unavailable (series too short)", ha="center", va="center")
                ax_a2.set_axis_off()
    else:
        # Not enough points or near-constant → show bootstrap CI on (a1) + PP fallback on (a2)
        q_wet = float(row.get("recommended_q_wet", np.nan))
        block_data = [
            ("b3", row.get("bootstrap_q_wet_b3", np.nan),
                row.get("bootstrap_q_wet_lo_b3", np.nan),
                row.get("bootstrap_q_wet_hi_b3", np.nan)),
            ("b5", row.get("bootstrap_q_wet_b5", np.nan),
                row.get("bootstrap_q_wet_lo_b5", np.nan),
                row.get("bootstrap_q_wet_hi_b5", np.nan)),
            ("b10", row.get("bootstrap_q_wet_b10", np.nan),
                    row.get("bootstrap_q_wet_lo_b10", np.nan),
                    row.get("bootstrap_q_wet_hi_b10", np.nan)),
        ]
        xs, meds, los, his = [], [], [], []
        for label, med, lo, hi in block_data:
            if pd.notna(med) and pd.notna(lo) and pd.notna(hi):
                xs.append(label); meds.append(float(med)); los.append(float(lo)); his.append(float(hi))
        if xs:
            ax_a1.errorbar(xs, meds,
                        yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                        fmt="o", color="tab:blue", ecolor="tab:blue", capsize=6,
                        label="Bootstrap q (median ± 95% CI)")
        if pd.notna(q_wet):
            ax_a1.axhline(q_wet, color="black", ls="--", lw=1.2,
                        label=f"recommended_q_wet={q_wet:.3f}")
        ax_a1.set_ylim(0, 1)
        ax_a1.set_title(f"(a1) Bootstrap quantile CIs (wet) | {TILE_TO_PLOT}")
        ax_a1.set_xlabel("Block size")
        ax_a1.set_ylabel("Quantile level q")
        ax_a1.legend(loc="lower right")

        # PP fallback (Normal reference)
        if series.size >= 3:
            x = np.sort(series)
            ranks = (np.arange(1, x.size + 1)) / (x.size + 1.0)
            mu, sigma = np.nanmean(series), np.nanstd(series)
            sigma = sigma if sigma > 1e-12 else 1.0
            from scipy.stats import norm
            model_cdf_ref = norm.cdf(x, loc=mu, scale=sigma)

            ax_a2.plot(model_cdf_ref, ranks, color="tab:green", lw=2, label="Empirical vs Normal CDF (fallback)")
            ax_a2.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="1:1")
            ax_a2.set_title(f"(a2) PP (fallback to Normal) | {TILE_TO_PLOT}")
            ax_a2.set_xlabel("Reference CDF")
            ax_a2.set_ylabel("Empirical CDF")
            ax_a2.set_xlim(0, 1)
            ax_a2.set_ylim(0, 1)
            ax_a2.legend(loc="lower right")
        else:
            ax_a2.text(0.5, 0.5, "PP unavailable (series too short)", ha="center", va="center")
            ax_a2.set_axis_off()

    # (Optional) Add a suptitle early; constrained_layout will account for it.
    fig.suptitle("Validation & Diagnostics Panel", fontsize=16)


    # # Panel (b): ACF/PACF of wet_score OR neighbor-year expansion summary
    # ax_b = fig.add_subplot(gs[0, 1])
    # try:
    #     acf_vals, pacf_vals = acf_pacf_data(df_ws["wet_score"].values, nlags=10)
    #     lags = np.arange(len(acf_vals))
    #     width = 0.35
    #     ax_b.bar(lags - width / 2, acf_vals, width=width, color="tab:blue", label="ACF")
    #     ax_b.bar(lags + width / 2, pacf_vals, width=width, color="tab:orange", label="PACF")
    #     ax_b.axhline(0, color="black", lw=0.8)
    #     ax_b.set_xticks(lags)
    #     ax_b.set_title(f"(b) ACF/PACF (lags 0–{len(lags) - 1}) | {TILE_TO_PLOT}")
    #     ax_b.set_xlabel("Lag")
    #     ax_b.set_ylabel("Correlation")
    #     ax_b.legend(loc="upper right")
    # except Exception:
    #     # Fallback: neighbor-year expansion visualization
    #     y = wet_years
    #     yn = wet_neighbors
    #     ax_b.eventplot([y, yn], colors=["tab:blue", "tab:orange"],
    #                    lineoffsets=[1, 0.5], linelengths=[0.4, 0.4])
    #     ax_b.set_title(f"(b) Neighbor-year expansion (wet) | {TILE_TO_PLOT}")
    #     ax_b.set_xlabel("Year")
    #     ax_b.set_yticks([1, 0.5])
    #     ax_b.set_yticklabels(["Selected wet years", "Neighbor-expanded"])
    #     if len(y) > 0:
    #         ax_b.set_xlim(min(y) - 1, max(y) + 1)

    # Panel (c): MK with FDR control across all tiles (wet)
    ax_c = fig.add_subplot(gs[1, 0])
    mk_df = df_just[["tile_id", "mk_p_wet", "mk_tau_wet"]].dropna()
    if not mk_df.empty:
        mk_df["p_adj"], mk_df["sig"] = benjamini_hochberg(mk_df["mk_p_wet"].values, alpha=0.05)
        ax_c.scatter(mk_df["mk_p_wet"], mk_df["p_adj"], s=18, alpha=0.7, color="tab:purple")
        ax_c.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        ax_c.axhline(0.05, color="red", lw=1, ls=":", label="α=0.05 (adjusted)")
        ax_c.set_title("(c) Field significance: raw vs FDR-adjusted p-values (wet)")
        ax_c.set_xlabel("Raw p-values (mk_p_wet)")
        ax_c.set_ylabel("FDR-adjusted p-values")
        ax_c.legend(loc="lower right")
    else:
        ax_c.text(0.5, 0.5, "No MK data available", ha='center', va='center')
        ax_c.set_axis_off()

    # Panel (d): Pixel exceedance validation (wet_fraction) with shaded extremes
    ax_d = fig.add_subplot(gs[1, 1])

    years = df_ws["year"].values.astype(int)  # ensure integer axis
    wet_frac = df_ws["wet_fraction"].values
    ax_d.plot(years, wet_frac, color="tab:blue", lw=2, label="Wet fraction")

    # Shade selected wet years (from justification CSV)
    for yr in wet_years:
        ax_d.axvspan(int(yr) - 0.5, int(yr) + 0.5, color="tab:blue", alpha=0.15)

    rl_wet = row.get("RL_target_wet", np.nan)
    rec_q_wet = row.get("recommended_q_wet", np.nan)
    subtitle = []
    if pd.notna(rl_wet):
        subtitle.append(f"RL_target_wet={float(rl_wet):.3f}")
    if pd.notna(rec_q_wet):
        subtitle.append(f"recommended_q_wet={float(rec_q_wet):.3f}")
    subtitle_txt = " | ".join(subtitle) if subtitle else ""

    ax_d.set_title(f"(d) Pixel exceedance validation | {TILE_TO_PLOT}\n{subtitle_txt}")
    ax_d.set_xlabel("Year")
    ax_d.set_ylabel("Wet fraction")

    # Force integer ticks only
    ax_d.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = [y for y in sorted(years) if y % 5 == 0]
    ax_d.set_xticks(ticks)
    ax_d.set_xticklabels([str(y) for y in ticks], rotation=45)


    # Layout & save
    plt.suptitle("Validation & Diagnostics Panel", y=0.98, fontsize=16)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # give room for suptitle


    out_path = os.path.join(outpath, f"validation_diagnostics_{TILE_TO_PLOT}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")


tile_a = '01AL000'
tile_b = '05OG000'
tile_c = '09AB000'
single_tile_plot(config_file,tile_a)
single_tile_plot(config_file,tile_b)
single_tile_plot(config_file,tile_c)