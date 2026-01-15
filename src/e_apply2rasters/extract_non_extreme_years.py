#non-extemere years

#!/usr/bin/env python3
"""
Extract non-extreme years per tile_id from a justification CSV.

Input CSV must contain columns:
  - tile_id
  - selected_wet_years_neighbors
  - selected_dry_years_neighbors

Behavior:
  - User sets start/end year (inclusive).
  - For each tile_id, compute the set of years in [start, end] that are NOT
    present in either neighbor list (wet or dry).
  - Write "non-extreme.csv" with columns:
      tile_id, n_sum, n_years
    where:
      n_years = semicolon-separated list of non-extreme years (ascending)
      n_sum   = sum(years in n_years) or 0 if empty.

Usage:
  python extract_non_extreme_years.py /path/to/justification.csv 2000 2023 [non-extreme.csv]

If output path is omitted, "non-extreme.csv" is written next to the input CSV.
"""

import os
import sys
import pandas as pd
import numpy as np

def parse_years_field(s):
    """
    Parse a semicolon-separated years string like '2001;2002'
    into a set of ints. Returns empty set on empty/NaN.
    """
    if s is None:
        return set()
    if isinstance(s, float) and np.isnan(s):
        return set()
    s = str(s).strip()
    if s == "":
        return set()
    out = set()
    for tok in s.split(";"):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out

def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_non_extreme_years.py justification.csv START_YEAR END_YEAR [out_csv]")
        sys.exit(1)

    in_csv = os.path.abspath(sys.argv[1])
    start_year = int(sys.argv[2])
    end_year   = int(sys.argv[3])
    out_csv = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else os.path.join(
        os.path.dirname(in_csv), "non-extreme.csv"
    )

    # Read and validate
    df = pd.read_csv(in_csv)
    required = ["tile_id", "selected_wet_years_neighbors", "selected_dry_years_neighbors"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Build the year universe for the range (inclusive)
    year_universe = set(range(start_year, end_year + 1))

    rows = []
    for _, r in df.iterrows():
        tid = str(r["tile_id"])

        wet_neighbors = parse_years_field(r.get("selected_wet_years_neighbors"))
        dry_neighbors = parse_years_field(r.get("selected_dry_years_neighbors"))

        extreme_neighbors = wet_neighbors.union(dry_neighbors)
        non_extreme = sorted(year_universe.difference(extreme_neighbors))

        # Build outputs
        non_extreme_str = ";".join(str(y) for y in non_extreme)
        n_sum = int(len(non_extreme)) if non_extreme else 0

        rows.append({
            "tile_id": tid,
            "n_sum": n_sum,
            "n_years": non_extreme_str
        })

    out_df = pd.DataFrame(rows, columns=["tile_id", "n_sum", "n_years"])
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(out_df)} rows -> {out_csv}")

if __name__ == "__main__":
    main()

#python extract_non_extreme_years.py D:/Research/FS-2dot0/results/WetDryTrendsPaper/supplement/results/2000-2023percentile_justification.csv 2000 2023