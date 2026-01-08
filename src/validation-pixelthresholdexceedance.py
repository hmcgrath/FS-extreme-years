
#!/usr/bin/env python3
"""
Loop across years (default 1990..2023) with minimal re-clipping:
- Reuse per-tile shape.shp and frequency clip (clipped once per tile).
- Clip FS per year, compute wet/dry stats, write one CSV.

Usage:
  python filtname_years.py XX [ "TILE_ID" ] [--start-year 1990 --end-year 2023] [--keep-fs-clips]

Hard-coded inputs:
- Shapefile:
    /gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/shape/nhn-land.shp
- FS dir (values 0..100, nodata=255; file pattern fsm-YYYY-historic-mc.tif):
    /gpfs/fs5/nrcan/nrcan_geobase/work/data/fs-historic/
- Frequency dir (values 0..100, nodata=255; file pattern frequency_threshold-XX.tif):
    /gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/trends/frequency/

Working/output:
- outdir:
    /gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/results/frequency-validation
- Per tile:
    {outdir}/{tile_id}/shape.shp
    {outdir}/{tile_id}/{tile_id}_freq.tif           # clipped once & reused
    {outdir}/{tile_id}/{tile_id}_fs_{YYYY}.tif      # clipped FS per year
- CSV:
    {outdir}/results_{start}-{end}_{xx}_{mode}.csv

Definitions:
- Domain: frequency > 0 and both rasters valid (nodata=255 excluded).
- Wet: fs > XX (strict).
- Dry: fs <= XX (inclusive).
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

# --------------------------
# Hard-coded paths & fields
# --------------------------
SHP_PATH = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/shape/nhn-land.shp"
FS_DIR   = "/gpfs/fs5/nrcan/nrcan_geobase/work/data/fs-historic"
FREQ_DIR = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/trends/frequency"

OUTDIR   = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/results/frequency-validation"
TILE_FIELD = "tile_id"

# Nodata/constants
NODATA_VAL = 255
FS_MIN, FS_MAX = 0, 100
FREQ_MIN, FREQ_MAX = 0, 100

def check_gdalwarp():
    try:
        subprocess.run(["gdalwarp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        raise RuntimeError("gdalwarp not found in PATH. Load GDAL or update your environment.") from e

def build_fs_path(year: int) -> str:
    return os.path.join(FS_DIR, f"fsm-{year}-historic-mc.tif")

def build_freq_path(xx: int) -> str:
    return os.path.join(FREQ_DIR, f"frequency_threshold-{xx}.tif")

def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def write_single_feature_shapefile(gdf: gpd.GeoDataFrame, tile_id: str, tile_dir: str) -> str:
    os.makedirs(tile_dir, exist_ok=True)
    tile_gdf = gdf[gdf[TILE_FIELD].astype(str) == str(tile_id)]
    if tile_gdf.empty:
        raise ValueError(f"Tile '{tile_id}' not found in shapefile.")
    shp_path = os.path.join(tile_dir, "shape.shp")
    tile_gdf.to_file(shp_path, driver="ESRI Shapefile")
    return shp_path

def gdalwarp_cutline(raster_path: str, cutline_path: str, out_raster_path: str, resampling="near"):
    # Explicit nodata handling for both FS and frequency clips
    cmd = [
        "gdalwarp",
        "-cutline", cutline_path,
        "-crop_to_cutline",
        "-of", "GTiff",
        "-overwrite",
        "-r", resampling,
        "-srcnodata", str(NODATA_VAL),
        "-dstnodata", str(NODATA_VAL),
    ]
    cmd += [raster_path, out_raster_path]
    subprocess.run(cmd, check=True)

def read_band_and_mask(path: str, value_range: tuple):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata_tag = src.nodata
    invalid = np.zeros(arr.shape, dtype=bool)
    if nodata_tag is not None:
        invalid |= (arr == nodata_tag)
    invalid |= (arr == NODATA_VAL)
    vmin, vmax = value_range
    invalid |= (arr < vmin) | (arr > vmax)
    return arr, invalid

def compute_stats(fs_clip_path: str, freq_clip_path: str, wet_cutoff: int, dry_cutoff: int) -> dict:
    """
    Domain: frequency > 0 AND valid (nodata excluded)
    Wet: fs > wet_cutoff (strict)
    Dry: fs <= dry_cutoff (inclusive)
    """
    fs_arr, fs_invalid = read_band_and_mask(fs_clip_path, value_range=(FS_MIN, FS_MAX))
    fr_arr, fr_invalid = read_band_and_mask(freq_clip_path, value_range=(FREQ_MIN, FREQ_MAX))

    freq_domain = (fr_arr > 0)
    eval_mask = (~fs_invalid) & (~fr_invalid) & freq_domain

    valid_count = int(eval_mask.sum())
    if valid_count == 0:
        return dict(
            valid_masked_count=0, wet_count=0, dry_count=0,
            fraction_wet=np.nan, fraction_dry=np.nan,
            wet_median_rank=np.nan, dry_median_rank=np.nan,
            fs_mean_masked=np.nan, fs_median_masked=np.nan
        )

    fs_vals = fs_arr[eval_mask].astype(float)

    wet_sel = (fs_vals > float(wet_cutoff))     # strict
    dry_sel = (fs_vals <= float(dry_cutoff))    # inclusive

    wet_count = int(np.count_nonzero(wet_sel))
    dry_count = int(np.count_nonzero(dry_sel))
    fraction_wet = wet_count / valid_count
    fraction_dry = dry_count / valid_count

    # Percent ranks (0..1) within domain FS distribution
    order = np.argsort(fs_vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = (np.arange(fs_vals.size, dtype=float) + 0.5) / fs_vals.size

    wet_median_rank = float(np.median(ranks[wet_sel])) if wet_count > 0 else np.nan
    dry_median_rank = float(np.median(ranks[dry_sel])) if dry_count > 0 else np.nan

    return dict(
        valid_masked_count=valid_count,
        wet_count=wet_count,
        dry_count=dry_count,
        fraction_wet=float(fraction_wet),
        fraction_dry=float(fraction_dry),
        wet_median_rank=wet_median_rank,
        dry_median_rank=dry_median_rank,
        fs_mean_masked=float(np.mean(fs_vals)),
        fs_median_masked=float(np.median(fs_vals)),
    )

def main():
    parser = argparse.ArgumentParser(description="Validate wet/dry exceedance across years with minimal re-clipping.")
    parser.add_argument("xx", type=int, help="Threshold cutoff (0..100). Wet: fs>XX; Dry: fs<=XX.")
    parser.add_argument("tile_id", nargs="?", default=None, help="Optional tile_id (process ALL if omitted).")
    parser.add_argument("--start-year", type=int, default=1990, help="Start year (default 1990)")
    parser.add_argument("--end-year", type=int, default=2023, help="End year inclusive (default 2023)")
    parser.add_argument("--keep-fs-clips", action="store_true", help="Keep per-year FS clips (default: delete after use).")
    args = parser.parse_args()

    xx = args.xx
    wet_cutoff = xx
    dry_cutoff = xx

    check_gdalwarp()
    ensure_outdir()

    # Load shapefile and decide tiles
    gdf = gpd.read_file(SHP_PATH)
    if TILE_FIELD not in gdf.columns:
        raise ValueError(f"Field '{TILE_FIELD}' not found in {SHP_PATH}. Available: {list(gdf.columns)}")
    if args.tile_id is None:
        tile_ids = list(gdf[TILE_FIELD].astype(str).dropna().unique())
        mode = "ALL"
    else:
        tile_ids = [str(args.tile_id)]
        mode = str(args.tile_id)

    # Verify frequency source exists
    freq_src = build_freq_path(xx)
    if not os.path.exists(freq_src):
        raise FileNotFoundError(f"Frequency raster not found: {freq_src}")

    rows = []
    years = list(range(args.start_year, args.end_year + 1))

    for tid in tile_ids:
        print(f"[Tile {tid}] Preparing cutline & frequency clip...")
        tile_dir = os.path.join(OUTDIR, tid)
        os.makedirs(tile_dir, exist_ok=True)

        # Save single-feature cutline once per tile
        shp_cutline = os.path.join(tile_dir, "shape.shp")
        if not os.path.exists(shp_cutline):
            shp_cutline = write_single_feature_shapefile(gdf, tid, tile_dir)

        # Clip frequency once per tile (reused for all years)
        fr_clip = os.path.join(tile_dir, f"{tid}_freq.tif")
        if not os.path.exists(fr_clip):
            gdalwarp_cutline(raster_path=freq_src, cutline_path=shp_cutline, out_raster_path=fr_clip, resampling="near")

        # Loop years: clip FS per year, compute stats, (optionally delete FS clip)
        for year in years:
            fs_src = build_fs_path(year)
            if not os.path.exists(fs_src):
                print(f"  [WARN] FS raster missing for year {year}: {fs_src} (skipping)")
                continue

            fs_clip = os.path.join(tile_dir, f"{tid}_fs_{year}.tif")
            if not os.path.exists(fs_clip):
                gdalwarp_cutline(raster_path=fs_src, cutline_path=shp_cutline, out_raster_path=fs_clip, resampling="near")

            stats = compute_stats(fs_clip_path=fs_clip, freq_clip_path=fr_clip,
                                  wet_cutoff=wet_cutoff, dry_cutoff=dry_cutoff)

            rows.append({
                "tile_id": tid,
                "year": year,
                "wet_cutoff": wet_cutoff,
                "dry_cutoff": dry_cutoff,
                "valid_masked_count": stats["valid_masked_count"],
                "wet_count": stats["wet_count"],
                "dry_count": stats["dry_count"],
                "fraction_wet": stats["fraction_wet"],
                "fraction_dry": stats["fraction_dry"],
                "wet_median_rank": stats["wet_median_rank"],
                "dry_median_rank": stats["dry_median_rank"],
                "fs_mean_masked": stats["fs_mean_masked"],
                "fs_median_masked": stats["fs_median_masked"],
            })

            # Optionally clean up FS clip to save space
            if not args.keep_fs_clips:
                try:
                    os.remove(fs_clip)
                except Exception:
                    pass

    out_csv = os.path.join(OUTDIR, f"results_{years[0]}-{years[-1]}_{xx}_{mode}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(df)} rows -> {out_csv}")

if __name__ == "__main__":
    main()

#python /gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/flood-susceptibility-main/src/trends/validation-pixelthresholdexceedance.py 38
#
#python /gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/flood-susceptibility-main/src/trends/validation-pixelthresholdexceedance.py 38 "01AL000" --start-year 1995 --end-year 2023
#python /gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/flood-susceptibility-main/src/trends/validation-pixelthresholdexceedance.py 38 "01AL000" --keep-fs-clips