
#!/usr/bin/env python3
# Recreate the Figure 7-like 3xN plot controlled by a config.json (no CLI args).
# Middle row is replaced with two stacked sub-panels (Wet trend on top, Dry trend below) per tile.
import os
import json
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WET_COLOR = "#39698b"  # blue
DRY_COLOR = "#af649c"  # orange
WET_BAND  = "lightblue"
DRY_BAND  = "lightcoral"

def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    # Required keys
    for k in ("summary_csv", "output_png"):
        if k not in cfg or not str(cfg[k]).strip():
            raise ValueError(f"config.json missing required key '{k}'")
    # Optional
    cfg.setdefault("scores_csv", None)
    cfg.setdefault("tiles", ["01AL000", "05OG000", "09AB000"])
    cfg.setdefault("years_range", [1990, 2023])
    return cfg

def parse_years(s):
    """Convert '2018;2019;2020' -> [2018, 2019, 2020]; empty -> []"""
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split(";") if str(x).strip().isdigit()]

def read_summary(path):
    df = pd.read_csv(path)
    req = [
        "tile_id","mk_tau_wet", "mk_p_wet", "mk_tau_dry", "mk_p_dry",
        "RL_target_wet", "RL_target_dry",
 
        "selected_wet_years_neighbors", "selected_dry_years_neighbors",
        # Wet, b=3/5/10 (median and 95% CI bounds)
        "bootstrap_q_wet_b3",  "bootstrap_q_wet_lo_b3",  "bootstrap_q_wet_hi_b3",
        "bootstrap_q_wet_b5",  "bootstrap_q_wet_lo_b5",  "bootstrap_q_wet_hi_b5",
        "bootstrap_q_wet_b10", "bootstrap_q_wet_lo_b10", "bootstrap_q_wet_hi_b10",
        # Dry, b=3/5/10 (median and 95% CI bounds)
        "bootstrap_q_dry_b3",  "bootstrap_q_dry_lo_b3",  "bootstrap_q_dry_hi_b3",
        "bootstrap_q_dry_b5",  "bootstrap_q_dry_lo_b5",  "bootstrap_q_dry_hi_b5",
        "bootstrap_q_dry_b10", "bootstrap_q_dry_lo_b10", "bootstrap_q_dry_hi_b10",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Summary CSV missing columns: {missing}")
    return df

def read_scores_optional(path):
    if path is None or str(path).strip() == "":
        return None
    df = pd.read_csv(path)
    needed = ["tile_id", "year", "wet_score", "dry_score"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Scores CSV missing columns: {miss}")
    return df

# ---------- Mann-Kendall (MK) test (normal approximation, tie-corrected) ----------
def mk_test(y):
    """Mann-Kendall trend test on 1D array-like y. Returns (tau, p)."""
    import math
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return np.nan, np.nan
    S = 0
    for i in range(n - 1):
        diffs = y[i+1:] - y[i]
        S += np.sum(np.sign(diffs))
    # Tie correction
    unique, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_S = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_S == 0:
        return 0.0, 1.0
    # Continuity correction
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0.0
    # Two-sided p using normal CDF via erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(Z) / math.sqrt(2.0))))
    tau = S / (0.5 * n * (n - 1))
    return float(tau), float(p)

# ---------- Top row: combined time series with shaded GEV years ----------
def plot_top_row(ax, tile_id, years, wet_series, dry_series, wet_sel, dry_sel, rl_target_wet, rl_target_dry):
    ax.plot(years, wet_series, color=WET_COLOR, lw=2, label="Wet score")
    ax.plot(years, dry_series, color=DRY_COLOR, lw=2, label="Dry score")
    for y in wet_sel:
        ax.axvspan(y - 0.5, y + 0.5, color=WET_BAND, alpha=0.25)
    for y in dry_sel:
        ax.axvspan(y - 0.5, y + 0.5, color=DRY_BAND, alpha=0.25)
    
    if rl_target_wet is not None and np.isfinite(rl_target_wet):
        ax.axhline(rl_target_wet, color=WET_COLOR, ls="--", lw=1.2, alpha=0.8, label="RL target (wet)")
    
    if rl_target_dry is not None and np.isfinite(rl_target_dry):
            ax.axhline(rl_target_dry, color=DRY_COLOR, ls="--", lw=1.2, alpha=0.8, label="RL target (dry)")


    ax.set_title(tile_id, fontsize=12, fontweight="bold")
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("Year"); ax.set_ylabel("Score")
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)

def plot_top_row_missing(ax, tile_id, wet_sel, dry_sel, years):
    ax.set_title(tile_id, fontsize=12, fontweight="bold")
    for y in wet_sel:
        ax.axvspan(y - 0.5, y + 0.5, color=WET_BAND, alpha=0.25)
    for y in dry_sel:
        ax.axvspan(y - 0.5, y + 0.5, color=DRY_BAND, alpha=0.25)
    ax.text(0.5, 0.5, "per-year scores missing", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("Year"); ax.set_ylabel("Score")
    ax.grid(True, ls=":", alpha=0.35)

# ---------- New: split middle row ----------
def plot_middle_wet(ax, years, wet_series, tile_id):
    ax.scatter(years, wet_series, color=WET_COLOR, s=25, label="Wet score")
    mask = np.isfinite(wet_series)
    ys = years[mask]
    ws = wet_series[mask]
    if ys.size >= 2:
        z = np.polyfit(ys, ws, 1)
        ax.plot(ys, np.polyval(z, ys), color=WET_COLOR, lw=2)
    tau, p = mk_test(ws)
    ax.set_title(f"{tile_id} - MK tau={tau:.3f}, p={p:.3g}", fontsize=10)
    ax.set_ylabel("Wet score")
    ax.grid(True, ls=":", alpha=0.35)

def plot_middle_dry(ax, years, dry_series, tile_id):
    ax.scatter(years, dry_series, color=DRY_COLOR, s=25, label="Dry score")
    mask = np.isfinite(dry_series)
    ys = years[mask]
    ds = dry_series[mask]
    if ys.size >= 2:
        z = np.polyfit(ys, ds, 1)
        ax.plot(ys, np.polyval(z, ys), color=DRY_COLOR, lw=2)
    tau, p = mk_test(ds)
    ax.set_title(f"{tile_id} - MK tau={tau:.3f}, p={p:.3g}", fontsize=10)
    ax.set_xlabel("Year"); ax.set_ylabel("Dry score")
    ax.grid(True, ls=":", alpha=0.35)

# ---------- Bottom row ----------

def plot_bottom_row(ax, row):
    # X positions: wet series at 3, 5, 10; dry series offset by +0.5
    wet_x = np.array([3.0, 5.0, 10.0], dtype=float)
    dry_x = wet_x + 0.5

    # --- Wet medians & CI (b=3,5,10) ---
    wet_med = np.array([
        float(row["bootstrap_q_wet_b3"]),
        float(row["bootstrap_q_wet_b5"]),
        float(row["bootstrap_q_wet_b10"])
    ])
    wet_lo = np.array([
        float(row["bootstrap_q_wet_lo_b3"]),
        float(row["bootstrap_q_wet_lo_b5"]),
        float(row["bootstrap_q_wet_lo_b10"])
    ])
    wet_hi = np.array([
        float(row["bootstrap_q_wet_hi_b3"]),
        float(row["bootstrap_q_wet_hi_b5"]),
        float(row["bootstrap_q_wet_hi_b10"])
    ])
    wet_yerr = np.vstack([wet_med - wet_lo, wet_hi - wet_med])

    # --- Dry medians & CI (b=3,5,10), plotted at 3.5, 5.5, 10.5 ---
    dry_med = np.array([
        float(row["bootstrap_q_dry_b3"]),
        float(row["bootstrap_q_dry_b5"]),
        float(row["bootstrap_q_dry_b10"])
    ])
    dry_lo = np.array([
        float(row["bootstrap_q_dry_lo_b3"]),
        float(row["bootstrap_q_dry_lo_b5"]),
        float(row["bootstrap_q_dry_lo_b10"])
    ])
    dry_hi = np.array([
        float(row["bootstrap_q_dry_hi_b3"]),
        float(row["bootstrap_q_dry_hi_b5"]),
        float(row["bootstrap_q_dry_hi_b10"])
    ])
    dry_yerr = np.vstack([dry_med - dry_lo, dry_hi - dry_med])

    # Plot (offset dry series so points don't overlay)
    ax.errorbar(
        wet_x, wet_med, yerr=wet_yerr, fmt="o", color=WET_COLOR,
        capsize=4, label="Wet q (median ± 95% CI)"
    )
    ax.errorbar(
        dry_x, dry_med, yerr=dry_yerr, fmt="o", color=DRY_COLOR,
        capsize=4, label="Dry q (median ± 95% CI)"
    )


    # X ticks: show only WET positions and labels
    wet_x = np.array([3.0, 5.0, 10.0], dtype=float)
    dry_x = wet_x + 0.5

    # ... (your errorbar calls above) ...

    # Use only wet ticks and labels
    ax.set_xticks(wet_x)
    ax.set_xticklabels(["  3", "  5", " 10"])

    # Make sure the full extent (incl. dry offset at 10.5) is visible
    xmin = wet_x.min() - 0.4
    xmax = (dry_x.max() if len(dry_x) else wet_x.max()) + 0.4
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Block size")
    ax.set_ylabel("Exceedance quantile (q)")
    ax.grid(True, ls=":", alpha=0.85)
    ax.legend(loc="lower right", fontsize=9)



def main():
    # Load config (you kept a fixed path; keeping that behavior)
    configfile = "D:/Research/FS-2dot0/results/WetDryTrendsPaper/scripts/configs/fig7.json"
    cfg = load_config(configfile)

    summary_path = os.path.abspath(cfg["summary_csv"])
    scores_path  = os.path.abspath(cfg["scores_csv"]) if cfg["scores_csv"] else None
    tiles        = [str(t) for t in cfg["tiles"]]
    y0, y1       = cfg["years_range"]
    years_axis   = np.arange(y0, y1 + 1)

    df_sum = read_summary(summary_path)
    dfts = df_sum[df_sum["tile_id"].astype(str).isin(tiles)].copy()
    if dfts.empty:
        raise ValueError(f"None of {tiles} found in summary CSV.")
    scores = read_scores_optional(scores_path)

    # --- Build figure layout using GridSpec with a sub-GridSpec in the middle row ---
    ncols = len(tiles)
    fig = plt.figure(figsize=(5 * ncols, 9), constrained_layout=True)
    # height_ratios: top (1.8), middle (split into 2), bottom (1.6) - tweak if you want
    gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[1.8, 2.4, 1.6])

    for col, tile_id in enumerate(tiles):
        row = dfts[dfts["tile_id"].astype(str) == tile_id].iloc[0]
        wet_sel = parse_years(row.get("selected_wet_years"))
        wet_nei = parse_years(row.get("selected_wet_years_neighbors"))
        dry_sel = parse_years(row.get("selected_dry_years"))
        dry_nei = parse_years(row.get("selected_dry_years_neighbors"))

        # Extract scores for tile
        if scores is not None and tile_id in scores["tile_id"].astype(str).unique():
            s = scores[scores["tile_id"].astype(str) == tile_id].sort_values("year")
            ser_w = pd.Series(s["wet_score"].values, index=s["year"]).reindex(years_axis)
            ser_d = pd.Series(s["dry_score"].values, index=s["year"]).reindex(years_axis)
        else:
            ser_w = pd.Series(index=years_axis, dtype=float)
            ser_d = pd.Series(index=years_axis, dtype=float)

        # --- Top row: timeseries with bands ---
        ax_top = fig.add_subplot(gs[0, col])
        if ser_w.isna().all() or ser_d.isna().all():
            plot_top_row_missing(ax_top, tile_id, wet_sel + wet_nei, dry_sel + dry_nei, years_axis)
        else:
            plot_top_row(ax_top, tile_id, years_axis, ser_w.values, ser_d.values, wet_sel + wet_nei, dry_sel + dry_nei, row["RL_target_wet"], row["RL_target_dry"])

        # --- Middle row: split into Wet (top) and Dry (bottom) sub-panels ---
        subgs = gs[1, col].subgridspec(nrows=2, ncols=1, hspace=0.15, height_ratios=[1, 1])
        ax_mid_w = fig.add_subplot(subgs[0, 0])
        ax_mid_d = fig.add_subplot(subgs[1, 0], sharex=ax_mid_w)  # share x for alignment

        if ser_w.isna().all():
            ax_mid_w.text(0.5, 0.5, "scores missing", transform=ax_mid_w.transAxes,
                          ha="center", va="center", color="gray")
            ax_mid_w.set_ylabel("Wet score"); ax_mid_w.grid(True, ls=":", alpha=0.35)
        else:
            plot_middle_wet(ax_mid_w, years_axis, ser_w.values, tile_id)

        if ser_d.isna().all():
            ax_mid_d.text(0.5, 0.5, "scores missing", transform=ax_mid_d.transAxes,
                          ha="center", va="center", color="gray")
            ax_mid_d.set_xlabel("Year"); ax_mid_d.set_ylabel("Dry score"); ax_mid_d.grid(True, ls=":", alpha=0.35)
        else:
            plot_middle_dry(ax_mid_d, years_axis, ser_d.values, tile_id)

        # --- Bottom row: bootstrap CIs ---
        ax_bot = fig.add_subplot(gs[2, col])
        plot_bottom_row(ax_bot, row)
        ax_bot.set_title(f"{tile_id}: Wet/Dry q (median ± 95% CI)", fontsize=10)

    # Save
    out_path = os.path.abspath(cfg["output_png"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"[OK] Wrote figure -> {out_path}")

if __name__ == "__main__":
    main()
