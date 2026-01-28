
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create envelope rasters from paired wet/dry extreme-year rasters.

For each wet raster in wet_folder, find the matching dry raster by tile_id
(parsed as the token after the final '_' and before '.tif' in the filename),
apply thresholds (>38), create per-layer threshold TIFs (wet=1, dry=2, NoData=255),
and mosaic them into a single envelope_<tile_id>.tif (0=none, 1=wet-only, 2=dry-or-both).

Author: hm
"""

import re
import sys
import shutil
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

# ----------------------------
# CONFIG (edit as needed)
# ----------------------------
DRY_FOLDER = Path("/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/trends/extremes/dry-years")
WET_FOLDER = Path("/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/trends/extremes/wet-years")
OUT_FOLDER = Path("/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/trends/extremes/envelope")

# Threshold and codes
THRESH = 38
NODATA_VAL = 255
WET_CODE = 1
DRY_CODE = 2
BG_CODE = 0  # for "no exceedance"

# Temp folder root (created under OUT_FOLDER)
TEMP_ROOT = OUT_FOLDER / "temp"


# ----------------------------
# Helpers
# ----------------------------
TILE_RE = re.compile(r"_(?P<tile>[A-Za-z0-9]+)\.tif$", re.IGNORECASE)

def tile_id_from_name(p: Path) -> str | None:
    m = TILE_RE.search(p.name)
    return m.group("tile") if m else None

def build_dry_index(dry_folder: Path) -> dict:
    idx = {}
    for p in dry_folder.glob("*.tif"):
        t = tile_id_from_name(p)
        if t:
            idx[t] = p
    return idx

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def read_raster(path: Path):
    ds = rasterio.open(path)
    arr = ds.read(1, masked=False)  # read into memory (files are small per user)
    profile = ds.profile.copy()
    ds.close()
    return arr, profile

def write_raster(path: Path, arr: np.ndarray, profile: dict):
    prof = profile.copy()
    # Force to Byte with NoData 255
    prof.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=NODATA_VAL,
        compress="lzw"
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.uint8), 1)

def make_threshold_mask(arr: np.ndarray, gt_thresh_val: int, code: int, nodata_val: int) -> np.ndarray:
    """Create a mask with `code` where arr > gt_thresh_val; preserve nodata; else 0."""
    out = np.full(arr.shape, BG_CODE, dtype=np.uint8)
    # Preserve nodata 255 exactly
    nodata_mask = (arr == nodata_val)
    # Apply threshold to valid pixels
    valid = ~nodata_mask
    sel = (arr > gt_thresh_val) & valid
    out[sel] = code
    out[nodata_mask] = nodata_val
    return out

def mosaic_envelope(wet_mask: np.ndarray, dry_mask: np.ndarray, nodata_val: int) -> np.ndarray:
    """
    Combine wet and dry masks:
      - Start with 0
      - Set to 1 where wet==1
      - Override to 2 where dry==2 (dry wins when both exceed)
      - Keep 255 where BOTH are nodata; if only one is nodata, treat the other normally
    """
    out = np.full(wet_mask.shape, BG_CODE, dtype=np.uint8)

    wet_is_data = wet_mask != nodata_val
    dry_is_data = dry_mask != nodata_val
    both_nodata = (~wet_is_data) & (~dry_is_data)

    # Wet-only
    out[wet_mask == WET_CODE] = WET_CODE
    # Dry (or both) wins
    out[dry_mask == DRY_CODE] = DRY_CODE
    # Where both are nodata -> nodata
    out[both_nodata] = nodata_val

    return out


# ----------------------------
# Main
# ----------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dirs(OUT_FOLDER, TEMP_ROOT)

    dry_idx = build_dry_index(DRY_FOLDER)
    wet_files = sorted(WET_FOLDER.glob("*.tif"))

    if not wet_files:
        logging.warning("No wet rasters found in %s", WET_FOLDER)
        return 0

    n_processed = 0
    n_skipped = 0

    for wet_path in wet_files:
        tile = tile_id_from_name(wet_path)
        if not tile:
            logging.warning("Skipping wet file (cannot parse tile_id): %s", wet_path.name)
            n_skipped += 1
            continue

        dry_path = dry_idx.get(tile)
        if not dry_path or not dry_path.exists():
            logging.info("No matching DRY file for tile_id=%s — skipping.", tile)
            n_skipped += 1
            continue

        # Temp directory for this tile
        tmp_dir = TEMP_ROOT / tile
        # clean any previous remnants
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        ensure_dirs(tmp_dir)

        logging.info("Processing tile_id=%s", tile)

        # Read rasters
        wet_arr, wet_prof = read_raster(wet_path)
        dry_arr, dry_prof = read_raster(dry_path)

        # Basic sanity checks on alignment
        same_shape = (wet_arr.shape == dry_arr.shape)
        same_crs = (wet_prof.get("crs") == dry_prof.get("crs"))
        same_transform = (wet_prof.get("transform") == dry_prof.get("transform"))

        if not (same_shape and same_crs and same_transform):
            logging.error(
                "Alignment mismatch for tile_id=%s. Shapes=%s/%s CRS=%s/%s Transform=%s/%s",
                tile, wet_arr.shape, dry_arr.shape,
                wet_prof.get("crs"), dry_prof.get("crs"),
                wet_prof.get("transform"), dry_prof.get("transform")
            )
            logging.error("Skipping tile_id=%s. Please resample/reproject to a common grid.", tile)
            n_skipped += 1
            continue

        # Threshold to masks
        wet_mask = make_threshold_mask(wet_arr, THRESH, WET_CODE, NODATA_VAL)
        dry_mask = make_threshold_mask(dry_arr, THRESH, DRY_CODE, NODATA_VAL)

        # Save threshold masks
        wet_thr_path = tmp_dir / f"wet_threshold_{tile}.tif"
        dry_thr_path = tmp_dir / f"dry_threshold_{tile}.tif"
        write_raster(wet_thr_path, wet_mask, wet_prof)
        write_raster(dry_thr_path, dry_mask, dry_prof)

        # Mosaic to envelope
        env_arr = mosaic_envelope(wet_mask, dry_mask, NODATA_VAL)

        # Save final envelope
        env_path = OUT_FOLDER / f"envelope_{tile}.tif"
        write_raster(env_path, env_arr, wet_prof)

        n_processed += 1
        logging.info("Completed tile_id=%s → %s", tile, env_path.name)

    logging.info("Done. Processed=%d, Skipped=%d", n_processed, n_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
