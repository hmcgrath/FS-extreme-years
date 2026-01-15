
#!/usr/bin/env python3
"""
Create two CSVs (wet-years.csv and dry-years.csv) and print the top-5 most frequent wet/dry years.

Inputs (columns needed minimally):
- tile_id
- final_count_wet
- selected_wet_years_neighbors  (semicolon-separated years, e.g., "2017;2020;2021")
- final_count_dry
- selected_dry_years_neighbors

Outputs:
1) wet-years.csv with columns:
   - tile_id
   - n_years (from final_count_wet)
   - years_list (semicolon-separated)
   - year_1, year_2, ...

2) dry-years.csv with columns:
   - tile_id
   - n_years (from final_count_dry)
   - years_list (semicolon-separated)
   - year_1, year_2, ...

Additionally prints:
- Top 5 most frequent wet years (global across rows), most to least
- Top 5 most frequent dry years (global across rows), most to least

Usage:
    python create_wet_dry_years.py input.csv [wet_output] [dry_output]
"""

import sys
import pandas as pd
from typing import List, Tuple
from collections import Counter


def _split_years(value: str) -> List[str]:
    """Split a semicolon-separated year string into a clean list of strings.
    Handles empty / NaN robustly.
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    # Normalize delimiters and remove extra spaces
    s = s.replace(" ", "")
    parts = [p for p in s.split(";") if p != ""]
    cleaned = []
    for p in parts:
        if p.isdigit():
            cleaned.append(p)
        else:
            try:
                int(p)  # allow negative or non-4-digit integers just in case
                cleaned.append(p)
            except Exception:
                pass
    return cleaned


def _expand_year_columns(year_lists: List[List[str]]) -> pd.DataFrame:
    """Create a DataFrame with year_1..year_k columns based on the max length across rows."""
    max_len = max((len(lst) for lst in year_lists), default=0)
    data = {}
    for i in range(max_len):
        col = f"year_{i+1}"
        data[col] = [lst[i] if i < len(lst) else "" for lst in year_lists]
    return pd.DataFrame(data)


def _build_output(df: pd.DataFrame, id_col: str, count_col: str, years_col: str) -> Tuple[pd.DataFrame, List[List[str]]]:
    """Build the output DataFrame and also return the parsed year lists for frequency tallies."""
    year_lists = df[years_col].apply(_split_years)

    out = pd.DataFrame({
        "tile_id": df[id_col].astype(str),
        "n_years": df[count_col].fillna(0).astype(int),
        "years_list": year_lists.apply(lambda lst: ";".join(lst)),
    })

    expanded = _expand_year_columns(list(year_lists))
    out = pd.concat([out, expanded], axis=1)
    return out, list(year_lists)


def _top_k_years(year_lists: List[List[str]], k: int = 5) -> List[Tuple[int, int]]:
    """Return top-k (year, count), sorted desc by count, then asc by year."""
    # Flatten and convert to int where possible
    flat = []
    for lst in year_lists:
        for y in lst:
            try:
                flat.append(int(y))
            except Exception:
                # ignore tokens that do not convert to int cleanly
                pass

    counter = Counter(flat)
    # sort by (-count, +year)
    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:k]


def process_years_csv(input_csv: str, wet_output: str = "wet-years.csv", dry_output: str = "dry-years.csv") -> None:
    """Read input_csv, produce wet-years.csv/dry-years.csv, and print top-5 wet/dry year frequencies."""
    df = pd.read_csv(input_csv)

    # Validate required columns exist
    required_wet = {"tile_id", "count_wet_neighbors", "selected_wet_years_neighbors"}
    required_dry = {"tile_id", "count_dry_neighbors", "selected_dry_years_neighbors"}
    missing_wet = [c for c in required_wet if c not in df.columns]
    missing_dry = [c for c in required_dry if c not in df.columns]
    if missing_wet:
        raise ValueError(f"Missing wet columns: {missing_wet}")
    if missing_dry:
        raise ValueError(f"Missing dry columns: {missing_dry}")

    wet_df, wet_year_lists = _build_output(df, "tile_id", "count_wet_neighbors", "selected_wet_years_neighbors")
    dry_df, dry_year_lists = _build_output(df, "tile_id", "count_dry_neighbors", "selected_dry_years_neighbors")

    wet_df.to_csv(wet_output, index=False)
    dry_df.to_csv(dry_output, index=False)

    # Compute and print frequency summaries
    top5_wet = _top_k_years(wet_year_lists, k=20)
    top5_dry = _top_k_years(dry_year_lists, k=20)

    print(f"Wrote {wet_output} (rows: {len(wet_df)}) and {dry_output} (rows: {len(dry_df)})")
    print("\nTop 5 most frequent wet years (year: count):")
    if top5_wet:
        for y, c in top5_wet:
            print(f"  {y}: {c}")
    else:
        print("  (no wet years found)")

    print("\nTop 5 most frequent dry years (year: count):")
    if top5_dry:
        for y, c in top5_dry:
            print(f"  {y}: {c}")
    else:
        print("  (no dry years found)")


if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #    print("Usage: python create_wet_dry_years.py input.csv [wet_output] [dry_output]")
    #    sys.exit(1)
    input_csv = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\working\\processed\\2000-2023percentile_justification_processed.csv" #sys.argv[1]
    wet_out = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\results\\wet-years.csv" #sys.argv[2] if len(sys.argv) >= 3 else "wet-years.csv"
    dry_out = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\results\\dry-years.csv" #sys.argv[3] if len(sys.argv) >= 4 else "dry-years.csv"
    process_years_csv(input_csv, wet_out, dry_out)
