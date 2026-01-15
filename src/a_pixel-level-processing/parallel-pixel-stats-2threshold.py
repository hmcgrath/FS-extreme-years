
import os
import re
import csv
import math
import gc
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.features import geometry_mask
from shapely.geometry import shape, mapping
from shapely.validation import explain_validity
import numpy as np
import pandas as pd

# ------------------ CONFIG ------------------
CONFIG_PATH = Path(__file__).parent / 'config.json'
with open(CONFIG_PATH, 'r') as cf:
    cfg = json.load(cf)

raster_dir = cfg.get('raster_dir')
shapefile_path = cfg.get('shapefile_path')
output_dir = cfg.get('output_dir')
threshold1 = cfg['thresholds'].get('wet_min_integer', 39)  # wet threshold
threshold2 = cfg['thresholds'].get('wet_plus', 88)         # wet+ threshold
NODATA = cfg.get('nodata', 255)
WINDOW_SIZE = cfg.get('window_size', 1024)
YEAR_PATTERN = cfg.get('year_filename_regex', r'(\d{4})')
TOP_N_DEFAULT = cfg.get('top_n_default', 5)
EXTREME_WEIGHT_DEFAULT = cfg.get('extreme_weight_default', 3.0)
LT_WEIGHT_DEFAULT = cfg.get('lt_weight_default', 0.5)
PARALLEL = cfg.get('parallel', True)
MAX_WORKERS = cfg.get('max_workers') or max(1, (os.cpu_count() or 8) - 1)

os.makedirs(output_dir, exist_ok=True)

# ------------------ HELPERS ------------------
def get_raster_paths_by_year(raster_dir, pattern=YEAR_PATTERN):
    year_re = re.compile(pattern)
    rasters = {}
    for root, _, files in os.walk(raster_dir):
        for f in files:
            if f.lower().endswith('.tif'):
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

# ------------------ CORE (Worker-safe) ------------------
def _count_pixels_by_threshold_worker(tile_id, geom_geojson, raster_paths, threshold1, threshold2, nodata, window_size):
    """Count pixels for a single polygon across all yearly rasters (worker function)."""
    from shapely.geometry import shape as shp_shape
    geom = shp_shape(geom_geojson)
    geoms = [mapping(geom)]
    results = {}
    for year, path in raster_paths.items():
        total_lt = 0
        total_gte = 0
        total_gt2 = 0
        with rasterio.open(path) as src:
            for win in _iter_windows_for_bounds(src.bounds, src, blocksize=window_size):
                arr = src.read(1, window=win, boundless=True, fill_value=nodata).astype(np.uint8)
                arr = np.where(arr == nodata, np.nan, arr)
                win_transform = window_transform(win, src.transform)
                mask_geom = geometry_mask(geoms, out_shape=arr.shape, transform=win_transform, invert=True)
                arr[~mask_geom] = np.nan
                valid = ~np.isnan(arr)
                total_lt += np.sum((arr < threshold1) & valid)
                total_gte += np.sum((arr >= threshold1) & valid)
                total_gt2 += np.sum((arr >= threshold2) & valid)
        results[year] = {"lt": int(total_lt), "gte": int(total_gte), "gt2": int(total_gt2)}
    return {"tile_id": tile_id, "counts": results}

# ------------------ PARALLEL PIPELINE ------------------
def process_all_polygons_parallel(shapefile_path, raster_dir, output_dir, threshold1, threshold2, nodata=NODATA, max_workers=MAX_WORKERS):
    raster_paths = get_raster_paths_by_year(raster_dir)
    print(f"[INFO] {len(raster_paths)} rasters found.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_years.csv"

    # Prepare features as tasks (GeoJSON geometries to avoid pickling Shapely objects)
    tasks = []
    with fiona.open(shapefile_path, 'r') as shp:
        for feat in shp:
            tile_id = feat['properties']['tile_id']
            geom_geojson = feat['geometry']
            # Validate/repair geometry in parent
            geom = shape(geom_geojson)
            geom = repair_geometry(geom)
            tasks.append((tile_id, mapping(geom)))

    # Write header
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['tile_id']
            for year in raster_paths:
                header += [f"{year}_lt", f"{year}_mid", f"{year}_gt2", f"{year}_gte"]
            writer.writerow(header)
            f.flush()

        print(f"[INFO] Starting parallel processing with max_workers={max_workers} ...")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for tile_id, geom_geojson in tasks:
                fut = ex.submit(
                    _count_pixels_by_threshold_worker,
                    tile_id, geom_geojson, raster_paths, threshold1, threshold2, NODATA, WINDOW_SIZE
                )
                futures.append(fut)
            for fut in as_completed(futures):
                res = fut.result()
                tile_id = res['tile_id']
                counts = res['counts']
                row = [tile_id]
                for year in raster_paths:
                    lt = counts[year]['lt']
                    gt2 = counts[year]['gt2']
                    gte = counts[year]['gte']
                    mid = gte - gt2
                    row += [lt, mid, gt2, gte]
                writer.writerow(row)
                f.flush()
    print(f"[DONE] All polygons processed → {csv_path}")
    return csv_path

# ------------------ Non-parallel fallback ------------------
def process_all_polygons(shapefile_path, raster_dir, output_dir, threshold1, threshold2, nodata=NODATA):
    raster_paths = get_raster_paths_by_year(raster_dir)
    print(f"[INFO] {len(raster_paths)} rasters found.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_years.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["tile_id"]
            for year in raster_paths:
                header += [f"{year}_lt", f"{year}_mid", f"{year}_gt2", f"{year}_gte"]
            writer.writerow(header)
            f.flush()
        with fiona.open(shapefile_path, "r") as shp:
            for feat in shp:
                tile_id = feat["properties"]["tile_id"]
                geom = repair_geometry(shape(feat["geometry"]))
                res = _count_pixels_by_threshold_worker(tile_id, mapping(geom), raster_paths, threshold1, threshold2, nodata=NODATA, window_size=WINDOW_SIZE)
                row = [tile_id]
                for year in raster_paths:
                    lt = res['counts'][year]["lt"]
                    gt2 = res['counts'][year]["gt2"]
                    gte = res['counts'][year]["gte"]
                    mid = gte - gt2
                    row += [lt, mid, gt2, gte]
                writer.writerow(row)
                f.flush()
                gc.collect()
    print(f"[DONE] All polygons processed → {csv_path}")
    return csv_path

# ------------------ WETTEST YEARS ------------------
def summarize_wettest_years(csv_path, output_dir, top_n=TOP_N_DEFAULT, col_suffix="_gt2"):
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

def compute_top_wet_years_custom(
    csv_path,
    output_dir,
    top_n=TOP_N_DEFAULT,
    weight_gt2=EXTREME_WEIGHT_DEFAULT,
    weight_gte=1.0,
    weight_lt=LT_WEIGHT_DEFAULT,
):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gte_cols = sorted([c for c in df.columns if c.endswith("_gte")])
    gt2_cols = sorted([c for c in df.columns if c.endswith("_gt2")])
    lt_cols = sorted([c for c in df.columns if c.endswith("_lt")])
    results = []
    for _, row in df.iterrows():
        tile_id = row["tile_id"]
        gte_rank = pd.Series(row[gte_cols]).rank(ascending=False, method='min')
        gt2_rank = pd.Series(row[gt2_cols]).rank(ascending=False, method='min')
        lt_rank = pd.Series(row[lt_cols]).rank(ascending=True, method='min')
        score = weight_gte * gte_rank + weight_gt2 * gt2_rank + weight_lt * lt_rank
        years = [c.replace("_gte","") for c in gte_cols]
        year_scores = dict(zip(years, score))
        top_years = sorted(year_scores.items(), key=lambda x: x[1])[:top_n]
        result = {"tile_id": tile_id}
        for i, (year, s) in enumerate(top_years, start=1):
            result[f"top{i}"] = year
            result[f"score{i}"] = round(float(s), 2)
        results.append(result)
    out_df = pd.DataFrame(results)
    output_file = output_dir / f"top{top_n}_wet_years_custom_w{weight_gte}_{weight_gt2}_{weight_lt}.csv"
    out_df.to_csv(output_file, index=False)
    print(f"[DONE] Saved top {top_n} wet years with custom weights to: {output_file}")
    return out_df

# ------------------ RUN ------------------
if __name__ == "__main__":
    if PARALLEL:
        csv_file = process_all_polygons_parallel(shapefile_path, raster_dir, output_dir, threshold1, threshold2, nodata=NODATA, max_workers=MAX_WORKERS)
    else:
        csv_file = process_all_polygons(shapefile_path, raster_dir, output_dir, threshold1, threshold2, nodata=NODATA)
    # Example post-processing (uncomment as needed):
    # summarize_wettest_years(csv_file, output_dir, col_suffix="_gt2", top_n=TOP_N_DEFAULT)
    # summarize_wettest_years(csv_file, output_dir, col_suffix="_gte", top_n=TOP_N_DEFAULT)
    # summarize_wettest_years(csv_file, output_dir, col_suffix="_lt", top_n=TOP_N_DEFAULT)
    # compute_top_wet_years_custom(csv_file, output_dir)
