
#!/usr/bin/env python3
# Compute wet and dry scores per tile_id, per year from a wide CSV using a config.json.
# Wide CSV format: tile_id, <YYYY>_lt, <YYYY>_mid, <YYYY>_gt2, <YYYY>_gte, ...
# Note: only *_lt, *_gte, *_gt2 are used; *_mid is ignored for the score definitions below.

import os
import re
import json
import sys
import numpy as np
import pandas as pd

inpathconfig = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\scripts\\configs\\wet-dry-score-calcs.json"

YEAR_COL_RE = re.compile(r'^(\d{4})_(lt|gte|gt2|mid)$')

DEFAULTS = {
    "weights": {
        "w_wet": 1.0,
        "w_vwet": 1.5,
        "w_dry": 1.0,
        "alpha": 2.0,
        "beta": 1.0
    },
    "years": None,            # e.g., [2000, 2023]; None => use all detected years
    "include_flags": False    # include diagnostic columns in output CSV
}

def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)

    # Validate required keys
    for k in ("input_csv", "output_csv"):
        if k not in cfg or not str(cfg[k]).strip():
            raise ValueError(f"config.json missing required key '{k}'")

    # Merge defaults
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in cfg.items() if k not in ("weights",)})
    weights = dict(DEFAULTS["weights"])
    weights.update(cfg.get("weights", {}))
    merged["weights"] = weights

    # Normalize years
    yrs = merged.get("years", None)
    if yrs is not None:
        if not (isinstance(yrs, (list, tuple)) and len(yrs) == 2 and all(isinstance(x, int) for x in yrs)):
            raise ValueError("config['years'] must be [start_year, end_year] or null")
        if yrs[0] > yrs[1]:
            raise ValueError("config['years']: start_year must be <= end_year")
    return merged

def detect_years_and_columns(df):
    """Return a sorted list of years present and mapping: year -> {suffix: column_name}."""
    years = set()
    colmap = {}
    for col in df.columns:
        m = YEAR_COL_RE.match(col)
        if not m:
            continue
        y = int(m.group(1)); suf = m.group(2)
        years.add(y)
        colmap.setdefault(y, {})[suf] = col
    return sorted(years), colmap

def compute_scores_for_row(row, years, colmap, w_wet, w_vwet, w_dry, alpha, beta):
    out = []
    for y in years:
        cols = colmap.get(y, {})
        # Require lt and gte; gt2 optional
        if "lt" not in cols or "gte" not in cols:
            continue

        # Read counts
        try:
            lt  = float(row.get(cols["lt"],  np.nan))
            gte = float(row.get(cols["gte"], np.nan))
            gt2 = float(row.get(cols.get("gt2", None), 0.0)) if cols.get("gt2", None) else 0.0
        except Exception:
            lt, gte, gt2 = np.nan, np.nan, 0.0

        if np.isnan(lt) or np.isnan(gte):
            continue

        # Clip: gt2 <= gte
        clipped = False
        if gt2 > gte:
            gt2 = gte
            clipped = True

        N = lt + gte
        if N <= 0:
            continue

        pct_dry  = float(np.clip(lt  / N, 0.0, 1.0))
        pct_wet  = float(np.clip(gte / N, 0.0, 1.0))
        pct_vwet = float(np.clip(gt2 / N, 0.0, 1.0))

        # Scores
        wet_score = w_wet * pct_wet + w_vwet * pct_vwet - w_dry * pct_dry
        dry_score = pct_dry - alpha * pct_vwet - beta * pct_wet

        out.append({
            "year": int(y),
            "wet_score": float(wet_score),
            "dry_score": float(dry_score),
            "clipped_gt2": bool(clipped),
            "lt": float(lt), "gte": float(gte), "gt2": float(gt2),
            "N": float(N),
            "pct_dry": pct_dry, "pct_wet": pct_wet, "pct_vwet": pct_vwet
        })
    return out

def main():
    #if len(sys.argv) < 2:
    #    print("Usage: python compute_wet_dry_scores_config.py /path/to/config.json")
    #    sys.exit(1)

    #cfg = load_config(sys.argv[1])
    cfg_path = inpathconfig
    cfg = load_config(cfg_path)
    inp = os.path.abspath(cfg["input_csv"])
    outp = os.path.abspath(cfg["output_csv"])
    w = cfg["weights"]
    include_flags = bool(cfg.get("include_flags", False))

    # Read input
    df = pd.read_csv(inp)
    if "tile_id" not in df.columns:
        raise ValueError("Input CSV must contain 'tile_id' column")

    years_all, colmap = detect_years_and_columns(df)
    if not years_all:
        raise ValueError("No year columns detected (need <YYYY>_(lt|gte|gt2)[, mid optional]).")

    # Restrict years if configured
    if cfg["years"] is not None:
        y0, y1 = cfg["years"]
        years = [y for y in years_all if y0 <= y <= y1]
    else:
        years = years_all

    out_rows = []
    for _, r in df.iterrows():
        tid = str(r.get("tile_id"))
        rows = compute_scores_for_row(
            r, years, colmap,
            w_wet=w["w_wet"], w_vwet=w["w_vwet"], w_dry=w["w_dry"],
            alpha=w["alpha"], beta=w["beta"]
        )
        for rr in rows:
            base = {"tile_id": tid, "year": rr["year"], "wet_score": rr["wet_score"], "dry_score": rr["dry_score"]}
            if include_flags:
                base.update({
                    "clipped_gt2": rr["clipped_gt2"],
                    "lt": rr["lt"], "gte": rr["gte"], "gt2": rr["gt2"],
                    "N": rr["N"],
                    "pct_dry": rr["pct_dry"], "pct_wet": rr["pct_wet"], "pct_vwet": rr["pct_vwet"]
                })
            out_rows.append(base)

    if not out_rows:
        print("[WARN] No output rows produced—check input format and year range.")

    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        out_df["year"] = out_df["year"].astype(int)
        out_df = out_df.sort_values(["tile_id", "year"]).reset_index(drop=True)
    out_df.to_csv(outp, index=False)
    print(f"[OK] Wrote {len(out_df)} rows -> {outp}")

if __name__ == "__main__":
    main()
