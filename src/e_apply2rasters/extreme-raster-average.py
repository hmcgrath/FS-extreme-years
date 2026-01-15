#!/usr/bin/env python3
"""
weighted_variable_rasters.py

Process a *wide-format* CSV (tile_id, n_years, years_list) to:
 1) clip per-year FSI rasters to the watershed polygon (tile_id) and
 2) compute a nodata-aware 3×3 focal mean across a **variable number** of rasters.

Differences vs. previous script:
- Accepts *wide* CSV with differing # years per tile (not fixed to 5).
- Generalizes focal mean across N rasters (N >= 1).
- Safer handling of nodata and dtype; explicit shape checks.

Usage
-----
python weighted_variable_rasters.py /path/to/selected_years_wide.csv

CSV format (wide)
-----------------
Expected header: tile_id,n_years,years_list
Example rows:
0201000,7,1993,1997,2015,2017,2018,2020,2023
0203000,5,1991,1995,2002,2014,2019

Notes
-----
- years_list may be comma-separated values; whitespace is ignored.
- Output rasters are written per tile in a working directory, and a final
  mosaic name uses tile_id.
"""

import sys
import csv
import os
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import fiona
from scipy.ndimage import uniform_filter

# ================================================================
# Configuration
# ================================================================
trim_value = 0.3     # UNUSED; kept for compatibility
NODATA_DEFAULT = 255 # default nodata if source nodata is missing

# HPC paths (edit if needed)
historic_full_paths = '/gpfs/fs5/nrcan/nrcan_geobase/work/data/fs-historic'
shapefile          = '/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/shape/nhn_all.shp'
maindir            = '/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/trends/extremes'
os.makedirs(maindir, exist_ok=True)

# Kernel size for focal mean
KERNEL_SIZE = 3  # 3×3 neighborhood

# ================================================================
# CSV (wide) reader
# ================================================================

def read_tile_years_wide(csv_file: str) -> List[Tuple[str, List[int]]]:
    """Read a wide-format CSV and return (tile_id, [years]).
    Expects columns: tile_id,n_years,years_list
    years_list can be comma-separated (e.g., "1991,1995,2002").
    """
    results: List[Tuple[str, List[int]]] = []
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Allow flexibility: if extra columns are present, we only use first 3
        # But enforce that tile_id is first and years_list is last.
        # Expected header names are not strictly needed; position is used.
        for row in reader:
            if not row:
                continue
            tile_id = str(row[0]).strip()
            # years_list can be in column 2 or 3 depending on presence of n_years
            if len(row) >= 3:
                years_tokens = row[2:]
            elif len(row) == 2:
                years_tokens = row[1:]
            else:
                years_tokens = []
            # Flatten tokens and split by comma
            tokens: List[str] = []
            for tok in years_tokens:
                if tok is None:
                    continue
                tokens.extend(str(tok).split(','))
            years = []
            for t in tokens:
                t2 = t.strip()
                if t2.isdigit():
                    years.append(int(t2))
            results.append((tile_id, years))
            print(f"Tile: {tile_id}, Years: {years}")
    print(f"[INFO] Finished reading CSV. Total tiles: {len(results)}")
    return results

# ================================================================
# Focal mean across variable N rasters
# ================================================================

def focal_mean_variable(stack: np.ndarray, nodata: int, kernel_size: int = KERNEL_SIZE) -> np.ndarray:
    """Nodata-aware focal mean over a kernel_size×kernel_size window across N rasters.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, rows, cols) with values in [0, 100] and nodata indicated by `nodata`.
    nodata : int
        Nodata marker value (e.g., 255).
    kernel_size : int
        Neighborhood size for the focal window (default 3).

    Returns
    -------
    np.ndarray
        2D result (rows, cols) with nodata where all contributing pixels are nodata.
    """
    print(f"[INFO] Computing nodata-aware {kernel_size}×{kernel_size} focal mean across {stack.shape[0]} rasters...")
    rows, cols = stack.shape[1], stack.shape[2]
    result = np.full((rows, cols), nodata, dtype=np.float32)

    # Valid mask per raster
    valid_mask = (stack != nodata)

    # Replace nodata with 0 for summation
    stack_filled = np.where(valid_mask, stack, 0).astype(np.float32)

    # uniform_filter returns local mean; multiply by kernel_size**2 to get local sum
    k_area = float(kernel_size * kernel_size)

    # Local sums and valid counts per raster
    raster_sums = []
    valid_counts = []
    for i in range(stack.shape[0]):
        s = uniform_filter(stack_filled[i], size=kernel_size, mode='constant', cval=0.0) * k_area
        c = uniform_filter(valid_mask[i].astype(np.float32), size=kernel_size, mode='constant', cval=0.0) * k_area
        raster_sums.append(s)
        valid_counts.append(c)

    raster_sums = np.stack(raster_sums, axis=0)
    valid_counts = np.stack(valid_counts, axis=0)

    # Sum across rasters
    total_sum = np.sum(raster_sums, axis=0)
    total_count = np.sum(valid_counts, axis=0)

    good = (total_count > 0)
    result[good] = total_sum[good] / total_count[good]

    return result

# ================================================================
# Clipping per tile
# ================================================================

def clip_rasters(years: List[int], shapefile: str, tile_id: str, output_dir: str, nodata: int = NODATA_DEFAULT) -> List[str]:
    """Clip each year's raster to the watershed polygon for tile_id.
    Returns list of clipped raster paths.
    """
    with fiona.open(shapefile, 'r') as shp:
        geoms = [
            feature['geometry']
            for feature in shp
            if str(feature['properties'].get('tile_id')) == str(tile_id)
        ]
    if not geoms:
        raise ValueError(f"No geometry found in shapefile for tile_id={tile_id}")

    os.makedirs(output_dir, exist_ok=True)
    clipped_paths: List[str] = []
    for year in years:
        raster_path = f"{historic_full_paths}/fsm-{year}-historic-mc.tif"
        with rasterio.open(raster_path) as src:
            src_nodata = src.nodata if src.nodata is not None else nodata
            print(f"[INFO] Clipping {raster_path} (nodata={src_nodata})...")
            out_image, out_transform = mask(src, geoms, crop=True, nodata=src_nodata)
            out_meta = src.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': out_image.shape[1],
            'width' : out_image.shape[2],
            'transform': out_transform,
            'nodata': out_meta.get('nodata', nodata)
        })
        out_name = os.path.basename(raster_path).replace('.tif', '_clipped.tif')
        out_path = os.path.join(output_dir, out_name)
        with rasterio.open(out_path, 'w', **out_meta) as dest:
            dest.write(out_image)
        clipped_paths.append(out_path)
        print(f"  [DONE] Saved {out_path}")
    return clipped_paths

# ================================================================
# Main processing
# ================================================================

def compute_focal_mean_for_tile(clipped_paths: List[str], out_raster: str, nodata: int = NODATA_DEFAULT, kernel_size: int = KERNEL_SIZE):
    if len(clipped_paths) < 1:
        raise ValueError("Need at least one clipped raster to compute focal mean.")

    print(f"[INFO] Opening {len(clipped_paths)} rasters for focal mean")
    datasets = [rasterio.open(p) for p in clipped_paths]
    ref_profile = datasets[0].profile.copy()
    arrays = [ds.read(1) for ds in datasets]

    # Ensure alignment
    base_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=2):
        if arr.shape != base_shape:
            for ds in datasets:
                ds.close()
            raise ValueError(f"All clipped rasters must have identical shape. Mismatch at index {i}")

    stack = np.stack(arrays, axis=0)

    # Compute focal mean
    result2d = focal_mean_variable(stack, nodata=nodata, kernel_size=kernel_size)

    # Clip to 0–100 (preserve nodata)
    mask_valid = (result2d != nodata)
    result2d[mask_valid & (result2d < 0)] = 0
    result2d[mask_valid & (result2d > 100)] = 100

    # Write output
    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, nodata=nodata)

    print(f"[INFO] Writing output to {out_raster}")
    with rasterio.open(out_raster, 'w', **out_profile) as dst:
        dst.write(result2d.astype(np.uint8), 1)

    for ds in datasets:
        ds.close()
    print("[DONE] Output saved.")

# ================================================================
# Entry point
# ================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python weighted_variable_rasters.py <wide_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    print(f"Using CSV file: {csv_file}")

    data = read_tile_years_wide(csv_file)

    for tile_id, years in data:
        if not years:
            print(f"[WARN] No years for tile {tile_id}; skipping.")
            continue
        print(f"[INFO] Processing tile {tile_id} with years {years}")
        #create subdir based on CSV, then work there
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        wd = f"{maindir}/{csv_basename}/{tile_id}"
        os.makedirs(wd, exist_ok=True)
        os.chdir(wd)

        # Clip rasters to watershed polygon
        clipped_paths = clip_rasters(years, shapefile, tile_id, wd, nodata=NODATA_DEFAULT)

        # Compute focal mean across N rasters
        out_raster = f"output_mean{len(clipped_paths)}r_{KERNEL_SIZE}x{KERNEL_SIZE}.tif"
        compute_focal_mean_for_tile(clipped_paths, out_raster, nodata=NODATA_DEFAULT, kernel_size=KERNEL_SIZE)

        # Move final output ------------------------------------------
        new_raster_name = f"{os.path.splitext(out_raster)[0]}_{tile_id}.tif"
        output_dir = f"{csv_basename}/final_outputs-meanNr{KERNEL_SIZE}x{KERNEL_SIZE}"
        os.makedirs(output_dir, exist_ok=True)
        out_raster2 = os.path.join(output_dir, new_raster_name)

        os.rename(out_raster, out_raster2)
        print(f"[DONE] Final saved: {out_raster2}")

if __name__ == '__main__':
    main()
