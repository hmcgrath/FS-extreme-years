
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert wide per-year fields to a long series for one Work Unit (tile_id).

Input schema (wide):
  tile_id, 2000_lt, 2000_mid, 2000_gt2, 2000_gte, 2001_lt, 2001_mid, 2001_gt2, 2001_gte, ...

Output schema (long):
  year,pct_wet,pct_wet_plus,pct_dry
  2000,0.21,0.03,0.79
  2001,0.18,0.02,0.82
  ...

Definitions:
  total_year_cells = _lt + _gte     (per year)
  pct_wet          = _gte / total
  pct_wet_plus     = _gt2 / total
  pct_dry          = _lt  / total

Usage:
  python make_wu_long_series.py \
      --in big_wide.csv \
      --tile 01AL000 \
      --out 01AL000_fractions.csv

Notes:
  - Missing year components are treated as 0 (e.g., if _gt2 absent for a year).
  - Years are inferred from column names like "YYYY_suffix".
  - The script ignores suffixes other than _lt, _gte, _gt2.
"""

import argparse
import re
import sys
import pandas as pd
import numpy as np

SUFFIXES = {"lt", "gte", "gt2"}  # we only use these for calculations
COL_RE = re.compile(r"^(?P<year>\d{4})_(?P<suffix>[A-Za-z0-9]+)$")


def parse_args():
    ap = argparse.ArgumentParser(description="Extract long series for one tile_id from wide CSV.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input wide CSV")
    ap.add_argument("--tile", dest="tile_id", required=True, help="Tile ID to extract (e.g., 01AL000)")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV (year,pct_wet,pct_wet_plus,pct_dry)")
    ap.add_argument("--round", dest="round_ndec", type=int, default=6,
                    help="Decimal places to round proportions (default 6).")
    return ap.parse_args()


def select_row(df: pd.DataFrame, tile: str) -> pd.Series:
    if "tile_id" not in df.columns:
        raise ValueError("Input CSV must have a 'tile_id' column")
    row = df.loc[df["tile_id"] == tile]
    if row.empty:
        raise SystemExit(f"[ERROR] tile_id '{tile}' not found in input.")
    if len(row) > 1:
        print(f"[WARN] Multiple rows for tile_id '{tile}', taking the first.", file=sys.stderr)
    return row.iloc[0]


def extract_year_suffix_values(row: pd.Series) -> pd.DataFrame:
    """Return a tidy dataframe: columns = year, lt, gte, gt2 (missing -> 0)."""
    buckets = {}  # year -> {suffix: value}
    for col, val in row.items():
        if col == "tile_id":
            continue
        m = COL_RE.match(col)
        if not m:
            continue
        year = int(m.group("year"))
        suf = m.group("suffix").lower()
        if suf not in SUFFIXES:
            continue
        buckets.setdefault(year, {})
        try:
            buckets[year][suf] = float(val)
        except Exception:
            buckets[year][suf] = np.nan

    if not buckets:
        raise SystemExit("[ERROR] No year_* columns with required suffixes were found.")

    years = sorted(buckets.keys())
    data = []
    for y in years:
        lt  = buckets[y].get("lt", 0.0)
        gte = buckets[y].get("gte", 0.0)
        gt2 = buckets[y].get("gt2", 0.0)
        # Convert NaNs to 0 for safety
        lt  = 0.0 if pd.isna(lt)  else lt
        gte = 0.0 if pd.isna(gte) else gte
        gt2 = 0.0 if pd.isna(gt2) else gt2
        data.append([y, lt, gte, gt2])

    return pd.DataFrame(data, columns=["year", "lt", "gte", "gt2"]).sort_values("year")


def compute_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Add fraction columns; handle zero totals gracefully."""
    total = df["lt"].fillna(0) + df["gte"].fillna(0)
    total = total.replace(0, np.nan)  # avoid divide-by-zero; will fill later

    pct_wet      = df["gte"] / total
    pct_wet_plus = df["gt2"] / total
    pct_dry      = df["lt"]  / total

    out = pd.DataFrame({
        "year": df["year"].astype(int),
        "pct_wet": pct_wet,
        "pct_wet_plus": pct_wet_plus,
        "pct_dry": pct_dry
    })

    # If total was 0, set safe defaults (0) rather than NaN
    return out.fillna(0.0)


def main():
    #"args = parse_args()"
    incsv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\gisdata\\data\\all_years-combined-final.csv"
    # Read CSV (engine handles large files too)
    df = pd.read_csv(incsv)
    tile_id = "09AB000"
    row = select_row(df, tile_id)
    round_ndec = 4

    tidy = extract_year_suffix_values(row)
    longdf = compute_fractions(tidy)

    if round_ndec is not None:
        longdf["pct_wet"] = longdf["pct_wet"].round(round_ndec)
        longdf["pct_wet_plus"] = longdf["pct_wet_plus"].round(round_ndec)
        longdf["pct_dry"] = longdf["pct_dry"].round(round_ndec)

    outcsv = f"D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\gisdata\\data\\{tile_id}-long-series.csv"
    longdf.to_csv(outcsv, index=False)
    print(f"[OK] Wrote {len(longdf)} rows to: long-series.csv")


if __name__ == "__main__":
    main()
