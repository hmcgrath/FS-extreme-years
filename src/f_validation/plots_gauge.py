
# -*- coding: utf-8 -*-
"""
Hydrometric summary panels for validation sites:
- Annual maximum discharge (bars)
- Standardized anomaly (z-score) overlay
- Vertical shading for FS-identified wet/dry extreme years

Sites covered: 01AL, 05OG, 09AB (one panel each)

Notes:
- This script reuses the same HYDAT DLY_FLOWS parsing approach you use in streamdata.py
  (sheet name DLY_FLOWS, melt of daily columns, date assembly). It will behave identically
  for reading flows. If later you have stage, you can point HYDAT_LEVELS_XLSX to DLY_LEVELS.xlsx.

Author: hm
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# User paths (edit as needed)
# ---------------------------
HYDAT_XLSX = r"D:\Research\FS-2dot0\hydat\DLY_FLOWS.xlsx"  # current file only has discharge
FS_CSV     = r"D:\Research\FS-2dot0\results\WetDryTrendsPaper\working\processed\2000-2023percentile_justification_processed.csv"
OUTDIR     = r"D:\Research\FS-2dot0\results\WetDryTrendsPaper\supplement\gauge_hydro_panels"

SITES      = ["01AL", "05OG", "09AB"]
START_YEAR = 2000
END_YEAR   = 2023


# ---------------------------
# FS extremes (wet/dry years)
# ---------------------------

def parse_years_cell(cell):
    """Extract 4-digit years from a cell like '2018,2019; 2021'."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, set)):
        return sorted({int(x) for x in cell})
    text = str(cell)
    yrs = re.findall(r"\b(?:19|20)\d{2}\b", text)
    return sorted({int(y) for y in yrs})

def load_fs_extremes(fs_csv_path):
    """
    Return dict: extremes[site_prefix] = {'wet': [years], 'dry': [years]}
    Site prefix is derived from codes like '01AL000' -> '01AL'.
    Columns are detected heuristically: any containing 'wet'+'year' or 'dry'+'year';
    fallback to 'wet'/'dry' if needed.
    """
    df = pd.read_csv(fs_csv_path)
    # Guess a WU/site column that carries codes like 01AL000
    wu_col = None
    for c in df.columns:
        if re.search(r"(wu|tile|site|unit|id)", c, flags=re.I):
            wu_col = c
            break
    if wu_col is None:
        wu_col = df.columns[0]

    wet_col = next((c for c in df.columns if re.search(r"wet.*year", c, flags=re.I)), None)
    dry_col = next((c for c in df.columns if re.search(r"dry.*year", c, flags=re.I)), None)
    if wet_col is None:
        wet_col = next((c for c in df.columns if re.search(r"\bwet\b", c, flags=re.I)), None)
    if dry_col is None:
        dry_col = next((c for c in df.columns if re.search(r"\bdry\b", c, flags=re.I)), None)

    extremes = {}
    for _, row in df.iterrows():
        code = str(row[wu_col])
        m = re.match(r"^([0-9]{2}[A-Z]{2})", code)  # e.g., 01AL000 -> 01AL
        site = m.group(1) if m else code[:4]
        wet_years = parse_years_cell(row[wet_col]) if wet_col in df.columns else []
        dry_years = parse_years_cell(row[dry_col]) if dry_col in df.columns else []
        if site not in extremes:
            extremes[site] = {"wet": [], "dry": []}
        extremes[site]["wet"] = sorted(set(extremes[site]["wet"] + wet_years))
        extremes[site]["dry"] = sorted(set(extremes[site]["dry"] + dry_years))
    return extremes

# ---------------------------
# HYDAT reading / transforms
# ---------------------------

def _coerce_full_month(x):
    """Coerce FULL_MONTH (TRUE/False/True/False/1/0) into boolean."""
    if isinstance(x, str):
        return x.strip().lower() in ("true", "t", "1", "y", "yes")
    if isinstance(x, (int, float, np.integer, np.floating)):
        if np.isnan(x): return False
        return x != 0
    return bool(x)

def read_hydat_monthly_table(xlsx_path, station_numbers=None, site_prefix=None):
    """
    Read raw DLY_FLOWS sheet to a monthly table with columns:
    [STATION_NUMBER, YEAR, MONTH, FULL_MONTH(bool), MAX (monthly maximum)]
    """
    df = pd.read_excel(xlsx_path, sheet_name="DLY_FLOWS", engine="openpyxl")
    # Station filter
    if station_numbers is not None:
        df = df[df["STATION_NUMBER"].isin(station_numbers)].copy()
    elif site_prefix is not None:
        df = df[df["STATION_NUMBER"].astype(str).str.startswith(site_prefix)].copy()

    # Keep needed columns; tolerate missing FULL_MONTH
    keep = ["STATION_NUMBER", "YEAR", "MONTH", "FULL_MONTH", "MAX"]
    for k in keep:
        if k not in df.columns:
            # If MAX is missing, caller will fall back to daily melt
            pass
    # Coerce FULL_MONTH
    if "FULL_MONTH" in df.columns:
        df["FULL_MONTH"] = df["FULL_MONTH"].apply(_coerce_full_month)
    else:
        df["FULL_MONTH"] = True

    # numeric MAX
    if "MAX" in df.columns:
        df["MAX"] = pd.to_numeric(df["MAX"], errors="coerce")
    return df

def annual_max_from_monthly_max(monthly_df, start_year=START_YEAR, end_year=END_YEAR,
                                require_full_month=True):
    """
    Compute annual maximum per station using monthly 'MAX'.
    Optionally require FULL_MONTH=True months.
    Returns df: [STATION_NUMBER, year, annual_max]
    """
    if monthly_df.empty or "MAX" not in monthly_df.columns:
        return pd.DataFrame(columns=["STATION_NUMBER", "year", "annual_max"])

    df = monthly_df.copy()
    df = df[df["YEAR"].between(start_year, end_year)]
    if require_full_month and "FULL_MONTH" in df.columns:
        df = df[df["FULL_MONTH"]]

    grp = df.groupby(["STATION_NUMBER", "YEAR"])["MAX"].max().reset_index()
    grp = grp.rename(columns={"YEAR": "year", "MAX": "annual_max"})
    return grp

def daily_fallback_annual_max(xlsx_path, station_number, start_year=START_YEAR, end_year=END_YEAR):
    """
    Fallback in case MAX is absent: melt FLOW1..FLOW31, ignore FLOW_SYMBOL*, compute annual maxima.
    """
    raw = pd.read_excel(xlsx_path, sheet_name="DLY_FLOWS", engine="openpyxl")
    raw = raw[(raw["STATION_NUMBER"] == station_number) & raw["YEAR"].between(start_year, end_year)].copy()

    # Identify day columns precisely: FLOW1..FLOW31 but NOT FLOW_SYMBOL*
    day_cols = [c for c in raw.columns if re.fullmatch(r"FLOW([1-9]|[12]\d|3[01])", c)]
    if not day_cols:
        return pd.DataFrame(columns=["STATION_NUMBER", "year", "annual_max"])

    # Melt days
    long_df = raw.melt(
        id_vars=["STATION_NUMBER", "YEAR", "MONTH", "NO_DAYS"],
        value_vars=day_cols,
        var_name="DAY_COL",
        value_name="FLOW"
    )
    # Derive day number
    long_df["DAY"] = long_df["DAY_COL"].str.replace("FLOW", "", regex=False).astype(int)
    long_df = long_df[(long_df["DAY"] <= long_df["NO_DAYS"]) & (~long_df["FLOW"].isna())].copy()

    # Annual max
    ann = long_df.groupby(["STATION_NUMBER", "YEAR"])["FLOW"].max().reset_index()
    ann = ann.rename(columns={"YEAR": "year", "FLOW": "annual_max"})
    return ann

def choose_primary_station(annual_df):
    """
    Pick station with the best annual coverage (# of non-null years).
    """
    if annual_df.empty:
        return None
    cov = (annual_df.dropna(subset=["annual_max"])
                    .groupby("STATION_NUMBER")["year"]
                    .nunique()
                    .sort_values(ascending=False))
    return None if cov.empty else cov.index[0]

def zscore(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu) / sd

# ---------------------------
# Plotting
# ---------------------------

def plot_site_panel(site_prefix, hydat_xlsx, fs_extremes, outdir,
                    start_year=START_YEAR, end_year=END_YEAR):
    """
    Build a single panel using monthly MAX -> annual_max.
    Falls back to daily melt if MAX is missing for the chosen station.
    """
    monthly = read_hydat_monthly_table(hydat_xlsx, site_prefix=site_prefix)
    ann_all = annual_max_from_monthly_max(monthly, start_year, end_year, require_full_month=True)

    # If nothing came out (unlikely), try fallback by scanning all stations with daily melt
    station = choose_primary_station(ann_all)
    if station is None:
        # Try to discover candidate stations first
        candidates = sorted(set(monthly["STATION_NUMBER"])) if not monthly.empty else []
        for stn in candidates:
            ann_try = daily_fallback_annual_max(hydat_xlsx, stn, start_year, end_year)
            if not ann_try.empty:
                ann_all = pd.concat([ann_all, ann_try], ignore_index=True)
        station = choose_primary_station(ann_all)

    if station is None:
        print(f"[WARN] {site_prefix}: no usable annual maxima found. Skipping.")
        return None

    ann = ann_all[ann_all["STATION_NUMBER"] == station].sort_values("year").copy()
    # Fill any missing years within the range with NaN (for visual continuity)
    full_years = pd.DataFrame({"year": np.arange(start_year, end_year + 1)})
    ann = full_years.merge(ann[["year", "annual_max"]], on="year", how="left")

    ann["z"] = zscore(ann["annual_max"].to_numpy(dtype=float))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.bar(ann["year"], ann["annual_max"], color="#4C72B0", alpha=0.65,
           edgecolor="0.35", linewidth=0.4, label="Annual max (m³/s)")

    ax2 = ax.twinx()
    ax2.plot(ann["year"], ann["z"], color="#C44E52", linewidth=1.6,
             label="Standardized anomaly (z)")
    ax2.axhline(0, color="#C44E52", linestyle="--", linewidth=0.9, alpha=0.6)

    # Shading for FS extremes
    wet_years = fs_extremes.get(site_prefix, {}).get("wet", [])
    dry_years = fs_extremes.get(site_prefix, {}).get("dry", [])
    for y in wet_years:
        if start_year <= y <= end_year:
            ax.axvspan(y - 0.5, y + 0.5, color="#1f78b4", alpha=0.20, lw=0)  # wet
    for y in dry_years:
        if start_year <= y <= end_year:
            ax.axvspan(y - 0.5, y + 0.5, color="#ff7f00", alpha={True:0.20, False:0.20}[True], lw=0)  # dry

    
    # Cosmetics
    ax.set_xlim(start_year - 0.5, end_year + 0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Discharge (m³/s)")
    ax2.set_ylabel("Standardized anomaly (z)")
    ax.set_title(f"{site_prefix}: primary gauge {station}")


    # --- NEW: allow extra headroom for legend above the axes ---
    # (adjust if your journal template has different whitespace)
    fig.subplots_adjust(top=0.75)

    # Legend combining both axes + patches
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    wet_patch = plt.Rectangle((0, 0), 1, 1, color="#1f78b4", alpha=0.20)
    dry_patch = plt.Rectangle((0, 0), 1, 1, color="#ff7f00", alpha=0.20)
    handles = h1 + h2 + [wet_patch, dry_patch]
    labels  = l1 + l2 + ["FS wet years", "FS dry years"]

    # --- UPDATED: place legend outside the axes, centered above the plot ---
    # bbox_to_anchor y=1.12 puts the legend just above the axes (below the title)
    
    ax.legend(
        handles, labels,
        ncol=4, frameon=True, fontsize=9,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0, columnspacing=1.0, handlelength=1.6
    ) 

    """
    fig.legend(
        handles, labels,
        ncol=4, frameon=True, fontsize=9,
        loc="upper center", bbox_to_anchor=(0.5, 0.94)  # figure coordinates
    ) 
    """
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"hydro_panel_{site_prefix}.png")
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fs_ext = load_fs_extremes(FS_CSV)

    chosen = {}
    for site in SITES:
        stn = plot_site_panel(site, HYDAT_XLSX, fs_ext, OUTDIR, START_YEAR, END_YEAR)
        if stn:
            chosen[site] = stn

    # Log which gauge was used
    if chosen:
        pd.DataFrame({"site": list(chosen.keys()), "station_used": list(chosen.values())}) \
          .to_csv(os.path.join(OUTDIR, "hydro_panels_station_log.csv"), index=False)

if __name__ == "__main__":
    main()
