
import json
import pandas as pd
import numpy as np
import os
import warnings

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def parse_year_list(year_str):
    """Split year strings like '2017;2020;2021' into integer list.
       Empty/None → empty list. Ignore blanks and strip spaces."""
    if pd.isna(year_str) or year_str is None or str(year_str).strip() == "":
        return []
    parts = [p.strip() for p in str(year_str).split(";") if p.strip() != ""]
    years = []
    for p in parts:
        try:
            y = int(p)
            years.append(y)
        except ValueError:
            warnings.warn(f"Invalid year token '{p}' encountered and ignored.")
    return years


def count_valid_decade(years, decade_start, decade_end):
    """Count years that fall into the specified decade range."""
    return sum(1 for y in years if decade_start <= y <= decade_end)


def weighted_decade_selection(decade_counts, decade_weights):
    """Compute primary and secondary decades based on weighted frequency.
       decade_counts: dict {decade_start: raw_count}
       decade_weights: dict {decade_start: divisor}
    """
    # Weighted score = raw_count / weight
    weighted = {
        d: (decade_counts[d] / decade_weights[d]) if decade_weights[d] > 0 else 0
        for d in decade_counts
    }

    # If all zero → return 0,0 according to user rule
    if all(v == 0 for v in weighted.values()):
        return 0, 0

    # Sort by score descending
    sorted_dec = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_dec[0][0]
    secondary = sorted_dec[1][0] if len(sorted_dec) > 1 else 0
    return primary, secondary


# -------------------------------------------------------
# Main script
# -------------------------------------------------------

def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, "..", "configs", "wet-dry.json"))
    
    with open(config_path, "r") as f:
        cfg = json.load(f)

    csv_in = cfg["csv_input"]
    csv_out = cfg["csv_output"]

    year_cols = cfg["year_columns"]
    decades = cfg["decades"]  # [2000, 2010, 2020]
    decade_weights = {int(k): v for k, v in cfg["decade_weights"].items()}

    # Load CSV
    df = pd.read_csv(csv_in)

    # Define decade ranges
    # e.g., 2000–2009, 2010–2019, 2020–2025
    decade_ranges = {
        2000: (2000, 2009),
        2010: (2010, 2019),
        2020: (2020, 2025)
    }

    # -------------------------------------------------------
    # 1. Year count fields
    # -------------------------------------------------------
    for label, col in year_cols.items():
        df[f"count_{label}"] = (
            df[col]
            .apply(parse_year_list)
            .apply(len)
        )

    # -------------------------------------------------------
    # 2. Decade counts
    # -------------------------------------------------------
    for label, col in year_cols.items():
        df[f"{label}_years_list"] = df[col].apply(parse_year_list)

        for d in decades:
            start, end = decade_ranges[d]
            df[f"count_decade_{label}_{d}"] = df[f"{label}_years_list"].apply(
                lambda yrs: count_valid_decade(yrs, start, end)
            )

    # -------------------------------------------------------
    # 3. Primary / Secondary decade selection
    # -------------------------------------------------------
    # Wet
    df["primary_wet"], df["secondary_wet"] = zip(*df.apply(
        lambda r: weighted_decade_selection(
            {d: r[f"count_decade_wet_{d}"] for d in decades},
            decade_weights
        ),
        axis=1
    ))

    # Dry
    df["primary_dry"], df["secondary_dry"] = zip(*df.apply(
        lambda r: weighted_decade_selection(
            {d: r[f"count_decade_dry_{d}"] for d in decades},
            decade_weights
        ),
        axis=1
    ))

    # -------------------------------------------------------
    # 4. Decade distance (wet & dry)
    # -------------------------------------------------------
    def compute_decade_dist(primary, secondary):
        if pd.isna(primary) or pd.isna(secondary):
            return np.nan
        if primary == 0 or secondary == 0:
            # user rule: zero-data → 0
            return 0
        return min(abs(int(primary) - int(secondary)) // 10, 2)

    df["decade_dist_wet"] = df.apply(
        lambda r: compute_decade_dist(r["primary_wet"], r["secondary_wet"]), axis=1
    )
    df["decade_dist_dry"] = df.apply(
        lambda r: compute_decade_dist(r["primary_dry"], r["secondary_dry"]), axis=1
    )

    # -------------------------------------------------------
    # Cleanup temporary columns
    # -------------------------------------------------------
    for label, col in year_cols.items():
        del df[f"{label}_years_list"]

    # -------------------------------------------------------
    # Write enhanced CSV
    # -------------------------------------------------------
    out_dir = os.path.dirname(csv_out)
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(csv_out, index=False)
    print(f"✅ Enhanced CSV with all new fields saved to:\n{csv_out}")


if __name__ == "__main__":
    main()
