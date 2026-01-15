import os
import re
import csv
import math
import gc
from pathlib import Path

import fiona
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.features import geometry_mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.validation import explain_validity
import numpy as np
import pandas as pd

# ---------------- PARAMS -----------------
raster_dir = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/whole/historic-calibrated-0to100"
raster_dir = "/gpfs/fs5/nrcan/nrcan_geobase/work/data/fs-historic"
shapefile_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/shape/nhn-land.shp"
output_dir = "/gpfs/fs5/nrcan/nrcan_geobase/work/dev/hem000/FSI2/results/newtop5"
threshold1 = 39 #because boundary is 0.383  (or 38.3.. only 39 + are 'wet')
threshold2 = 88 # wet+ = ≥ 88 (global q25)
NODATA = 255
WINDOW_SIZE = 1024
os.makedirs(output_dir, exist_ok=True)

top_n_default = 5
extreme_weight_default = 3
lt_weight_default = 0.5

# ---------------- HELPERS ----------------
def get_raster_paths_by_year(raster_dir, pattern=r'(\d{4})'):
    year_re = re.compile(pattern)
    rasters = {}
    for root, _, files in os.walk(raster_dir):
        for f in files:
            if f.lower().endswith(".tif"):
                m = year_re.search(f)
                if m:
                    year = m.group(1)
                    rasters[year] = os.path.join(root, f)
    return dict(sorted(rasters.items()))

def repair_geometry(geom):
    if not geom.is_valid:
        print(f"Repairing geometry: {explain_validity(geom)}")
        geom = geom.buffer(0)
    return geom

def _iter_windows_for_bounds(bounds, src, blocksize=WINDOW_SIZE):
    base_win = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, src.transform)
    col_off = int(math.floor(base_win.col_off))
    row_off = int(math.floor(base_win.row_off))
    win_width = int(math.ceil(base_win.width))
    win_height = int(math.ceil(base_win.height))
    for r in range(row_off, row_off + win_height, blocksize):
        h = min(blocksize, row_off + win_height - r)
        for c in range(col_off, col_off + win_width, blocksize):
            w = min(blocksize, col_off + win_width - c)
            yield Window(col_off=c, row_off=r, width=w, height=h)

# ---------------- CORE FUNCTIONS ----------------
def count_by_two_thresholds(raster_paths, threshold1, threshold2):
    results = {}
    for year, path in raster_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            valid = ~np.isnan(arr)
            lt = np.sum((arr < threshold1) & valid)
            mid = np.sum((arr >= threshold1) & (arr < threshold2) & valid)
            gt = np.sum((arr >= threshold2) & valid)
            results[year] = {"lt": int(lt), "mid": int(mid), "gt": int(gt)}
        gc.collect()
    return results
    
def count_pixels_by_threshold(raster_paths, geom, threshold1, threshold2, nodata=255):
    """
    Count pixels per raster (year) for a given polygon geometry:
      - lt: pixels < threshold1
      - gte: pixels >= threshold1
      - gt2: pixels >= threshold2
    Returns a dict: {year: {"lt": ..., "gte": ..., "gt2": ...}, ...}

    Parameters:
        raster_paths (dict): {year: raster_path}
        geom (shapely.geometry): polygon geometry
        threshold1 (int/float): wet threshold
        threshold2 (int/float): extreme wet threshold (> threshold1)
        nodata (int/float): nodata value in raster
    """
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds, transform as window_transform
    import math

    results = {}
    for year, path in raster_paths.items():
        total_lt = 0
        total_gte = 0
        total_gt2 = 0
        geoms = [mapping(geom)]

        with rasterio.open(path) as src:
            # Iterate over windows covering raster bounds
            for win in _iter_windows_for_bounds(src.bounds, src, blocksize=WINDOW_SIZE):
                arr = src.read(1, window=win, boundless=True, fill_value=nodata).astype(np.uint8)

                # Mask nodata
                arr = np.where(arr == nodata, np.nan, arr)

                # Apply polygon mask
                win_transform = window_transform(win, src.transform)
                mask_geom = geometry_mask(geoms, out_shape=arr.shape, transform=win_transform, invert=True)
                arr[~mask_geom] = np.nan

                valid = ~np.isnan(arr)
                total_lt += np.sum((arr < threshold1) & valid)
                total_gte += np.sum((arr >= threshold1) & valid)
                total_gt2 += np.sum((arr >= threshold2) & valid)

        results[year] = {"lt": int(total_lt), "gte": int(total_gte), "gt2": int(total_gt2)}

    return results


def clip_rasters_for_tile(tile_id, geom, raster_paths, output_dir):
    tile_dir = Path(output_dir) / str(tile_id)
    tile_dir.mkdir(parents=True, exist_ok=True)
    geoms = [mapping(geom)]
    clipped_paths = {}
    for year, path in raster_paths.items():
        out_tif = tile_dir / f"{year}.tif"
        if out_tif.exists():
            clipped_paths[year] = str(out_tif)
            continue
        with rasterio.open(path) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            with rasterio.open(out_tif, "w", **out_meta) as dst:
                dst.write(out_image)
        clipped_paths[year] = str(out_tif)
    return clipped_paths

def process_all_polygons(shapefile_path, raster_dir, output_dir, threshold1, threshold2, nodata=255):
    """
    Process all polygons in a shapefile:
    - Clip rasters to each polygon
    - Count pixels < threshold1 (lt), >= threshold1 (gte), >= threshold2 (gt2)
    - Compute _mid = gte - gt2
    - Write results to CSV with columns: tile_id, _lt, _mid, _gt2, _gte per year

    Parameters:
        shapefile_path (str or Path): Path to polygon shapefile
        raster_dir (str or Path): Directory containing yearly rasters
        output_dir (str or Path): Directory to save output CSV
        threshold1 (int/float): Wet threshold (gte)
        threshold2 (int/float): Extreme wet threshold (gt2 > threshold1)
        nodata (int/float): Raster nodata value (default 255 for Byte rasters)
    """
    raster_paths = get_raster_paths_by_year(raster_dir)
    print(f"[INFO] {len(raster_paths)} rasters found.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_years.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # --- Write header if file does not exist ---
        if not file_exists:
            header = ["tile_id"]
            for year in raster_paths:
                header += [f"{year}_lt", f"{year}_mid", f"{year}_gt2", f"{year}_gte"]
            writer.writerow(header)
            f.flush()

        # --- Process polygons ---
        with fiona.open(shapefile_path, "r") as shp:
            for feat in shp:
                tile_id = feat["properties"]["tile_id"]
                geom = repair_geometry(shape(feat["geometry"]))
                print(f"[INFO] Processing {tile_id} ...")

                # --- Clip rasters for polygon ---
                clipped = clip_rasters_for_tile(tile_id, geom, raster_paths, output_dir)

                # --- Count pixels per threshold ---
                counts = count_pixels_by_threshold(clipped, geom, threshold1, threshold2, nodata=nodata)

                # --- Prepare row: lt, mid, gt2, gte ---
                row = [tile_id]
                for year in raster_paths:
                    lt = counts[year]["lt"]
                    gt2 = counts[year]["gt2"]
                    gte = counts[year]["gte"]
                    mid = gte - gt2  # wet but not extreme
                    row += [lt, mid, gt2, gte]

                # --- Write row ---
                writer.writerow(row)
                f.flush()
                gc.collect()

    print(f"[DONE] All polygons processed → {csv_path}")
    return csv_path



# ---------------- WETTEST YEARS FUNCTIONS ----------------
def summarize_wettest_years(csv_path, output_dir, top_n=top_n_default, col_suffix="_gt2"):
    df = pd.read_csv(csv_path)
    year_cols = [c for c in df.columns if c.endswith(col_suffix)]
    tile_results = []

    for _, row in df.iterrows():
        tile_id = row["tile_id"]
        year_counts = {c.replace(col_suffix, ""): row[c] for c in year_cols}
        top_years = sorted(year_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result = {"tile_id": tile_id}
        for i, (year, value) in enumerate(top_years, start=1):
            result[f"top{i}"] = year
            result[f"count{i}"] = value
        tile_results.append(result)

    out_df = pd.DataFrame(tile_results)
    output_file = Path(output_dir) / f"wettest_by_tile_id{col_suffix}.csv"
    out_df.to_csv(output_file, index=False)
    print(f"[DONE] Saved results to: {output_file}")
    return out_df

import pandas as pd
from pathlib import Path

def compute_top_wet_years_custom(
    csv_path,
    output_dir,
    top_n=5,
    weight_gt2=2.0,
    weight_gte=1.0,
    weight_lt=0.5
):
    """
    Compute top N wettest years per polygon using weighted ranking,
    with fully customizable weights for each component (_gte, _gt2, _lt).

    Score = weight_gte * gte_rank + weight_gt2 * gt2_rank + weight_lt * lt_rank
    Lower score = wetter year

    Parameters
    ----------
    csv_path : str or Path
        Input CSV containing tile_id and columns like 1990_gte, 1990_gt2, 1990_lt, ...
    output_dir : str or Path
        Directory to save output CSV.
    top_n : int
        Number of top wet years to output per polygon.
    weight_gt2 : float
        Weight for extreme wet pixels (_gt2).
    weight_gte : float
        Weight for wet pixels (_gte).
    weight_lt : float
        Weight for dry pixels (_lt); higher = stronger penalty.
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify columns
    gte_cols = sorted([c for c in df.columns if c.endswith("_gte")])
    gt2_cols = sorted([c for c in df.columns if c.endswith("_gt2")])
    lt_cols = sorted([c for c in df.columns if c.endswith("_lt")])

    results = []

    for _, row in df.iterrows():
        tile_id = row["tile_id"]

        # Rank each component
        gte_rank = pd.Series(row[gte_cols]).rank(ascending=False, method='min')
        gt2_rank = pd.Series(row[gt2_cols]).rank(ascending=False, method='min')
        lt_rank = pd.Series(row[lt_cols]).rank(ascending=True, method='min')  # lower _lt = better

        # Weighted score
        score = weight_gte * gte_rank + weight_gt2 * gt2_rank + weight_lt * lt_rank

        # Map years to scores
        years = [c.replace("_gte","") for c in gte_cols]
        year_scores = dict(zip(years, score))

        # Sort ascending (lower = wetter)
        top_years = sorted(year_scores.items(), key=lambda x: x[1])[:top_n]

        # Prepare output row
        result = {"tile_id": tile_id}
        for i, (year, s) in enumerate(top_years, start=1):
            result[f"top{i}"] = year
            result[f"score{i}"] = round(s, 2)
        results.append(result)

    out_df = pd.DataFrame(results)
    output_file = output_dir / f"top{top_n}_wet_years_custom_w{weight_gte}_{weight_gt2}_{weight_lt}.csv"
    out_df.to_csv(output_file, index=False)
    print(f"[DONE] Saved top {top_n} wet years with custom weights to: {output_file}")
    return out_df


# ----------------------------- RUN ----------------
if __name__ == "__main__":
    # Step 1: process all polygons to CSV
    csv_file = process_all_polygons(shapefile_path, raster_dir, output_dir, threshold1, threshold2)

    # Step 2: summarize wettest years (example)
    #summarize_wettest_years(csv_file, output_dir, col_suffix="_gt2", top_n=top_n_default)
    #summarize_wettest_years(csv_file, output_dir, col_suffix="_gte", top_n=top_n_default)
    #summarize_wettest_years(csv_file, output_dir, col_suffix="_lt", top_n=top_n_default)

    # Step 3: compute top wet years with weighted scores
    # compute_top_wet_years_custom(
    #     csv_file,
    #     output_dir,
    #     top_n=5,
    #     weight_gt2=2.5,
    #     weight_gte=1.0,
    #     weight_lt=0.5
    # )
