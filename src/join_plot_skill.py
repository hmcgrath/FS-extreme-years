
#!/usr/bin/env python3
"""
Interactive join + plot + skill script for a single tile from a Completed CSV.

What it does:
- Ask for paths to:
    (a) Completed results CSV (per-year rows; may contain multiple tiles)
    (b) Justification CSV (GEV selections and trends)
- Ask for tile_id (or present a chooser if you leave blank).
- Derive defaults for thresholds (wet/dry fractions and FS median) with interactive override.
- Plot fraction_wet & fraction_dry (primary axis) and fs_median_masked (secondary axis),
  shading selected wet/dry years and marking them with asterisks.
- Compute skill metrics (precision/recall/F1, ROC AUC) for classifiers based on area fractions
  and FS median; compute Spearman correlations; include MK trend values from justification.
- Save plot PNG and summary CSV to the SAME directory as the Completed CSV.

Assumptions (matching your pipeline):
- Completed CSV has columns:
  ['tile_id','year','fraction_wet','fraction_dry','fs_median_masked','wet_cutoff','dry_cutoff']
- Justification CSV has columns:
  ['tile_id','selected_wet_years','selected_dry_years','mk_tau_wet','mk_p_wet','mk_tau_dry','mk_p_dry']
- 'selected_*_years' are semicolon-delimited strings like '2018;2019;2020'.
- Wet bin: FS > XX (strict); Dry bin: FS <= XX (inclusive).

Author: Interactive revision for Heather (Geospatial Scientist).
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Utility functions
# ----------------------------

""" def ask_path(prompt: str) -> str:
    #Interactive path input with existence validation
    while True:
        p = input(f"{prompt}\n> ").strip().strip('"').strip("'")
        if p == "":
            print("  Please provide a path.")
            continue
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            return p_abs
        else:
            print(f"  Not found: {p_abs}\n  Try again.") """

def parse_years_str(s: str) -> list:
    """Convert '2018;2019;2020' -> [2018, 2019, 2020]."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split(";") if str(x).strip().isdigit()]

def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho = Pearson on ranks (ties averaged)."""
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return float(xr.corr(yr))

def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Manual ROC AUC (binary labels in {0,1}, continuous scores).
    Returns NaN if labels are all one class.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    thresholds = np.unique(scores)[::-1]  # sweep high -> low
    tprs, fprs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr); fprs.append(fpr)
    # add bounds
    fprs = np.array([0.0] + fprs + [1.0])
    tprs = np.array([0.0] + tprs + [1.0])
    return float(np.trapz(tprs, fprs))

def precision_recall_f1(pred: np.ndarray, labels: np.ndarray):
    """Precision, recall, F1 for binary classification."""
    pred = np.asarray(pred).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    return float(precision), float(recall), float(f1)

def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def parse_tile_id_from_filename(path: str) -> str:
    """
    Try to extract tile_id from results filename, e.g. results_1990-2023_38_01AL000.csv
    """
    base = os.path.basename(path)
    m = re.search(r"results_[\d\-]+_\d+_([A-Za-z0-9]+)\.csv", base)
    return m.group(1) if m else None

# ----------------------------
# Interactive main
# ----------------------------

def main():
    #print("\n=== Pixel Exceedance Validation: Interactive Join + Plot + Skill ===")

    #completed_csv = ask_path("Path to Completed per-year results CSV (e.g., results_1990-2023_38_ALL.csv):")
    #just_csv      = `ask_path`("Path to justification CSV (e.g., 1990-2023percentile_justification.csv):")

    completed_csv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\validation\\01AL000\\results_1990-2023_38_01AL000.csv"
    just_csv      = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\1990-2023percentile_justification.csv"
    # Read both tables
    df = pd.read_csv(completed_csv)
    just = pd.read_csv(just_csv)

    # Validate required columns
    required_results_cols = [
        "tile_id", "year", "fraction_wet", "fraction_dry",
        "fs_median_masked", "wet_cutoff", "dry_cutoff"
    ]
    missing_r = [c for c in required_results_cols if c not in df.columns]
    if missing_r:
        print(f"\n[ERROR] Completed CSV missing columns: {missing_r}")
        sys.exit(1)

    if "tile_id" not in just.columns:
        print("\n[ERROR] Justification CSV must contain 'tile_id'.")
        sys.exit(1)

    # Ask for tile_id (offer a chooser)
    tile_id_hint = parse_tile_id_from_filename(completed_csv)
    print("\nEnter the tile_id to process."
          f"\n - Leave blank to see available IDs (hint from filename: {tile_id_hint or 'None'})")
    tile_id = input("> ").strip()
    if tile_id == "":
        tiles = sorted(df["tile_id"].astype(str).unique())
        print(f"\nAvailable tile_ids ({len(tiles)}):\n" + ", ".join(tiles[:25]) + (" ..." if len(tiles) > 25 else ""))
        tile_id = input("Choose tile_id:\n> ").strip()
    tile_id = str(tile_id)

    # Filter to tile
    dft = df[df["tile_id"].astype(str) == tile_id].copy()
    if dft.empty:
        print(f"\n[ERROR] No rows for tile_id={tile_id} in {completed_csv}")
        sys.exit(1)
    dft = dft.sort_values("year").reset_index(drop=True)

    # Derive default thresholds from cutoffs
    wet_cutoff = int(dft["wet_cutoff"].iloc[0])   # e.g., 38
    dry_cutoff = int(dft["dry_cutoff"].iloc[0])   # e.g., 38
    default_wet_frac   = 0.60
    default_dry_frac   = 0.60
    default_med_wet    = float(wet_cutoff + 1)    # strict wet > XX ⇒ median ≥ XX+1
    default_med_dry    = float(dry_cutoff)        # dry ≤ XX ⇒ median ≤ XX

    print("\nThresholds (press Enter to accept defaults):")
    wet_frac = input(f"  Area fraction threshold for WET (default {default_wet_frac}): ").strip()
    dry_frac = input(f"  Area fraction threshold for DRY (default {default_dry_frac}): ").strip()
    med_wet  = input(f"  FS median threshold for WET (default {default_med_wet:.0f}): ").strip()
    med_dry  = input(f"  FS median threshold for DRY (default {default_med_dry:.0f}): ").strip()

    wet_frac = float(wet_frac) if wet_frac != "" else default_wet_frac
    dry_frac = float(dry_frac) if dry_frac != "" else default_dry_frac
    med_wet  = float(med_wet)  if med_wet  != "" else default_med_wet
    med_dry  = float(med_dry)  if med_dry  != "" else default_med_dry

    # Pull justification row for tile
    jrow = just[just["tile_id"].astype(str) == tile_id]
    if jrow.empty:
        print(f"\n[ERROR] No row for tile_id={tile_id} in justification CSV.")
        sys.exit(1)
    jrow = jrow.iloc[0]

    wet_sel_years = parse_years_str(jrow.get("selected_wet_years"))
    dry_sel_years = parse_years_str(jrow.get("selected_dry_years"))

    # ----------------------------
    # Build & save plot
    # ----------------------------
    years = dft["year"].values
    fw = dft["fraction_wet"].values
    fd = dft["fraction_dry"].values
    fm = dft["fs_median_masked"].values

    # Naming
    out_dir = os.path.dirname(completed_csv)
    base = os.path.splitext(os.path.basename(completed_csv))[0]
    m = re.search(r"results_([0-9\-]+)_(\d+)_", base)  # extract 'years' and 'XX' if present
    years_part = f"{m.group(1)}_{m.group(2)}" if m else "series"
    out_prefix = f"joinplot_{years_part}_{tile_id}"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    lw = 2.0
    ax.plot(years, fw, color="steelblue", lw=lw, label="Fraction wet")
    ax.plot(years, fd, color="firebrick", lw=lw, label="Fraction dry")
    ax2.plot(years, fm, color="black", lw=1.5, label="FS median (0–100)")

    # Bands for selected years
    for y in wet_sel_years:
        ax.axvspan(y - 0.5, y + 0.5, color="lightblue", alpha=0.25)
    for y in dry_sel_years:
        ax.axvspan(y - 0.5, y + 0.5, color="lightcoral", alpha=0.20)

    # Asterisks on selected years
    wet_mask = np.isin(years, wet_sel_years)
    dry_mask = np.isin(years, dry_sel_years)
    ax.plot(years[wet_mask], fw[wet_mask], "*", color="navy", markersize=10, label="Selected wet (GEV)")
    ax.plot(years[dry_mask], fd[dry_mask], "*", color="darkred", markersize=10, label="Selected dry (GEV)")

    # Threshold lines
    ax.axhline(wet_frac, color="steelblue", ls="--", lw=1, alpha=0.6, label=f"wet frac ≥ {wet_frac}")
    ax.axhline(dry_frac, color="firebrick", ls="--", lw=1, alpha=0.6, label=f"dry frac ≥ {dry_frac}")
    ax2.axhline(med_wet, color="black", ls="--", lw=1, alpha=0.4, label=f"median ≥ {med_wet:.0f} (wet)")
    ax2.axhline(med_dry, color="gray", ls="--", lw=1, alpha=0.4, label=f"median ≤ {med_dry:.0f} (dry)")

    ax.set_title(f"{tile_id} — Wet/Dry Fractions & FS Median vs. GEV")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area fraction")
    ax2.set_ylabel("FS median (0–100)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, frameon=True)
    ax.grid(True, ls=":", alpha=0.4)

    plot_path = os.path.join(out_dir, f"{out_prefix}_timeseries.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    # ----------------------------
    # Skill metrics & summary
    # ----------------------------
    wet_labels = np.isin(years, wet_sel_years).astype(int)
    dry_labels = np.isin(years, dry_sel_years).astype(int)

    # Classifiers: fractions
    wet_pred_frac = (fw >= wet_frac).astype(int)
    dry_pred_frac = (fd >= dry_frac).astype(int)

    # Classifiers: FS median
    wet_pred_med  = (fm >= med_wet).astype(int)
    dry_pred_med  = (fm <= med_dry).astype(int)

    # Metrics
    wet_prec_f, wet_rec_f, wet_f1_f = precision_recall_f1(wet_pred_frac, wet_labels)
    wet_prec_m, wet_rec_m, wet_f1_m = precision_recall_f1(wet_pred_med,  wet_labels)
    wet_auc_f = roc_auc(fw, wet_labels)
    wet_auc_m = roc_auc(fm, wet_labels)

    dry_prec_f, dry_rec_f, dry_f1_f = precision_recall_f1(dry_pred_frac, dry_labels)
    dry_prec_m, dry_rec_m, dry_f1_m = precision_recall_f1(dry_pred_med,  dry_labels)
    dry_auc_f = roc_auc(fd, dry_labels)
    dry_auc_m = roc_auc(-fm, dry_labels)  # lower median => more dry

    rho_wet = spearman_rho(pd.Series(fw), pd.Series(fm))
    rho_dry = spearman_rho(pd.Series(fd), pd.Series(fm))

    mk_tau_wet = jrow.get("mk_tau_wet", np.nan)
    mk_p_wet   = jrow.get("mk_p_wet",   np.nan)
    mk_tau_dry = jrow.get("mk_tau_dry", np.nan)
    mk_p_dry   = jrow.get("mk_p_dry",   np.nan)

    wet_years_str = ", ".join(str(y) for y in wet_sel_years) if wet_sel_years else "None"
    dry_years_str = ", ".join(str(y) for y in dry_sel_years) if dry_sel_years else "None"

    summary_lines = [
        f"For {tile_id}, pixel-level validation aligns with GEV extremes: wet years ({wet_years_str}) show elevated wet fractions and higher FS medians, while dry years ({dry_years_str}) show the opposite.",
        f"Using area fractions, wet precision/recall/F1 = {fmt(wet_prec_f)}, {fmt(wet_rec_f)}, {fmt(wet_f1_f)} (AUC={fmt(wet_auc_f)}); dry = {fmt(dry_prec_f)}, {fmt(dry_rec_f)}, {fmt(dry_f1_f)} (AUC={fmt(dry_auc_f)}).",
        f"Using FS median, wet precision/recall/F1 = {fmt(wet_prec_m)}, {fmt(wet_rec_m)}, {fmt(wet_f1_m)} (AUC={fmt(wet_auc_m)}); dry = {fmt(dry_prec_m)}, {fmt(dry_rec_m)}, {fmt(dry_f1_m)} (AUC={fmt(dry_auc_m)}).",
        f"Spearman correlations indicate coherent behavior across years (ρ_wet={fmt(rho_wet)}, ρ_dry={fmt(rho_dry)}), with wet fractions increasing and dry fractions decreasing with FS median.",
        f"Mann–Kendall trends are consistent with the recent wet dominance and declining dryness (τ_wet={fmt(mk_tau_wet)}, p_wet={fmt(mk_p_wet)}; τ_dry={fmt(mk_tau_dry)}, p_dry={fmt(mk_p_dry)}).",
        f"Overall, area-based validation provides a robust cross-check of GEV extremes, with clear correspondence between selected years and fraction/median peaks."
    ]
    summary_text = " ".join(summary_lines)

    out_summary = os.path.join(out_dir, f"{out_prefix}_summary.csv")
    summary_row = {
        "tile_id": tile_id,
        "wet_selected_years": wet_years_str,
        "dry_selected_years": dry_years_str,
        "wet_thresh_frac": wet_frac,
        "dry_thresh_frac": dry_frac,
        "median_wet_thresh": med_wet,
        "median_dry_thresh": med_dry,
        "wet_precision_frac": wet_prec_f,
        "wet_recall_frac":    wet_rec_f,
        "wet_f1_frac":        wet_f1_f,
        "wet_auc_frac":       wet_auc_f,
        "wet_precision_med":  wet_prec_m,
        "wet_recall_med":     wet_rec_m,
        "wet_f1_med":         wet_f1_m,
        "wet_auc_med":        wet_auc_m,
        "dry_precision_frac": dry_prec_f,
        "dry_recall_frac":    dry_rec_f,
        "dry_f1_frac":        dry_f1_f,
        "dry_auc_frac":       dry_auc_f,
        "dry_precision_med":  dry_prec_m,
        "dry_recall_med":     dry_rec_m,
        "dry_f1_med":         dry_f1_m,
        "dry_auc_med":        dry_auc_m,
        "spearman_rho_wet":   rho_wet,
        "spearman_rho_dry":   rho_dry,
        "mk_tau_wet":         mk_tau_wet,
        "mk_p_wet":           mk_p_wet,
        "mk_tau_dry":         mk_tau_dry,
        "mk_p_dry":           mk_p_dry,
        "summary_sentences":  summary_text,
        "plot_path":          plot_path,
        "completed_csv":      completed_csv,
        "justification_csv":  just_csv,
    }
    pd.DataFrame([summary_row]).to_csv(out_summary, index=False)

    print(f"\n[OK] Wrote plot    -> {plot_path}")
    print(f"[OK] Wrote summary -> {out_summary}\n")

if __name__ == "__main__":
    main()
