
import json
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------------
# Helper: adjust lightness of a base color (bivariate logic)
# --------------------------------------------------------------------------------
def adjust_lightness(color, factor):
    rgb = np.array(to_rgb(color))
    return np.clip(rgb * factor, 0, 1)

# --------------------------------------------------------------------------------
# Helper: min-max scaling (kept if needed later)
# --------------------------------------------------------------------------------
def minmax(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

# --------------------------------------------------------------------------------
# Helper: scalebar
# --------------------------------------------------------------------------------
def add_scalebar(ax, length_fraction=1, height_fraction=0.015):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    map_width = xmax - xmin
    raw_length = map_width * length_fraction
    nice_lengths = np.array([
        100, 250, 500,
        1000, 2500, 5000,
        10000, 25000, 50000,
        100000
    ])
    length = nice_lengths[np.argmin(np.abs(nice_lengths - raw_length))]
    bar_height = (ymax - ymin) * height_fraction
    x0 = xmin + map_width * 0.05
    y0 = ymin + (ymax - ymin) * 0.05
    ax.add_patch(Rectangle((x0, y0), length / 2, bar_height,
                           facecolor="black", edgecolor="black"))
    ax.add_patch(Rectangle((x0 + length / 2, y0), length / 2, bar_height,
                           facecolor="white", edgecolor="black"))
    label = f"{int(length / 1000)} km" if length >= 1000 else f"{int(length)} m"
    ax.text(x0 + length / 2,
            y0 + bar_height * 1.6,
            label,
            ha="center",
            va="bottom",
            fontsize=9)

# --------------------------------------------------------------------------------
# Bivariate plotting into a provided Axes (with inset legend)
# - CHANGES:
#   * Removed subplot title (per request).
#   * Legend scaled by 0.80 and moved to upper-right.
# --------------------------------------------------------------------------------
def plot_bivariate_on_ax(
    gdf, primary_field, secondary_field,
    decade_colors, lightness_map,
    ax,
    legend_inset=True,
    # Previous size ~ (0.02, 0.02, 0.22, 0.22)
    # Scale by 0.80 -> width/height ≈ 0.176, move to upper-right.
    inset_rect=(0.80, 0.80, 0.18, 0.18)
):
    gdf = gdf.copy()

    # Compute decade distance (0–2, clipped) and drop rows with NaNs
    gdf["__dist__"] = (abs(gdf[primary_field] - gdf[secondary_field]) // 10).clip(0, 2)
    gdf = gdf.dropna(subset=[primary_field, secondary_field])

    # Compute color
    def compute_color(row):
        decade = int(row[primary_field])
        dist = int(row["__dist__"])
        base = decade_colors.get(str(decade), "#999999")
        factor = lightness_map[str(dist)]
        return adjust_lightness(base, factor)

    gdf["plot_color"] = gdf.apply(compute_color, axis=1)

    # Draw map
    gdf.plot(ax=ax, color=gdf["plot_color"], linewidth=0.2, edgecolor="black")
    ax.set_axis_off()
    add_scalebar(ax)

    # Inset bivariate legend (3x3 style) inside the subplot
    if legend_inset:
        decades_sorted = sorted([int(k) for k in decade_colors.keys()])
        dists_sorted = sorted([int(k) for k in lightness_map.keys()])

        # inset axes: (left, bottom, width, height) in axes coordinates
        legend_ax = ax.inset_axes(inset_rect)
        legend_ax.set_axis_off()

        # draw grid of color squares
        for i, dist in enumerate(dists_sorted):
            for j, decade in enumerate(decades_sorted):
                base = decade_colors[str(decade)]
                factor = lightness_map[str(dist)]
                col = adjust_lightness(base, factor)
                legend_ax.add_patch(
                    Rectangle((j, len(dists_sorted)-1-i), 1, 1,
                              facecolor=col, edgecolor="black", linewidth=0.5)
                )

        legend_ax.set_xlim(0, len(decades_sorted))
        legend_ax.set_ylim(0, len(dists_sorted))

        # ticks & labels
        legend_ax.set_xticks(np.arange(len(decades_sorted)) + 0.5)
        legend_ax.set_yticks(np.arange(len(dists_sorted)) + 0.5)
        legend_ax.set_xticklabels(decades_sorted, fontsize=7, rotation=0)
        legend_ax.set_yticklabels(dists_sorted, fontsize=7)
        legend_ax.set_xlabel("Primary Decade", fontsize=7, labelpad=2)
        legend_ax.set_ylabel("Decade Distance", fontsize=7, labelpad=2)

# --------------------------------------------------------------------------------
# Intensity (count) plotting into a provided Axes (with colorbar)
# - CHANGES:
#   * Removed subplot title (per request).
# --------------------------------------------------------------------------------
def plot_intensity_on_ax(gdf, field, cmap_name, ax):
    gdf = gdf.copy()

    # Draw map without GeoPandas legend
    gdf.plot(
        column=field,
        ax=ax,
        cmap=cmap_name,
        linewidth=0.2,
        edgecolor="black",
        legend=False
    )

    # Colorbar
    values = gdf[field].to_numpy()
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Attach a compact colorbar to the current ax
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(field.replace("_", " ").title(), fontsize=9)

    ax.set_axis_off()
    add_scalebar(ax)

# --------------------------------------------------------------------------------
# Main workflow: build a single 2x2 figure
# - CHANGES:
#   * Panel letters added at top-left of each subplot.
#   * Subplot titles removed entirely.
# --------------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, "..", "configs", "wet-dry.json"))
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Load data
    csv_path = cfg["csv_output"]
    shp_path = cfg["shapefile_input"]
    outdir = cfg["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(shp_path)

    # CRS check / reproject if needed
    crs_target = cfg["crs_settings"]["target_crs"]
    if cfg["crs_settings"]["enforce_projected"]:
        if (gdf.crs is None) or ("EPSG:3979" not in str(gdf.crs)):
            warnings.warn("Shapefile not in EPSG:3979. Reprojecting...")
            gdf = gdf.to_crs(crs_target)

    # Join using tile_id
    if "tile_id" not in gdf.columns:
        raise RuntimeError("Shapefile is missing 'tile_id' field.")
    merged = gdf.merge(df, on="tile_id", how="left", indicator=True)

    # Warn for missing matches, then filter
    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        warnings.warn(f"{len(missing)} features missing CSV data. They will be omitted in plots.")
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

    # Create a single 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # --- Top-left: Wet bivariate (keep wet on the left)
    plot_bivariate_on_ax(
        merged,
        primary_field="primary_wet",
        secondary_field="secondary_wet",
        decade_colors=cfg["bivariate"]["wet_decade_colors"],
        lightness_map=cfg["bivariate"]["lightness"],
        ax=axes[0, 0],
        legend_inset=True
    )

    # --- Top-right: Dry bivariate
    plot_bivariate_on_ax(
        merged,
        primary_field="primary_dry",
        secondary_field="secondary_dry",
        decade_colors=cfg["bivariate"]["dry_decade_colors"],
        lightness_map=cfg["bivariate"]["lightness"],
        ax=axes[0, 1],
        legend_inset=True
    )

    # --- Bottom-left: Wet year count + neighbors
    wet_field = "count_wet_neighbors"
    if wet_field in merged.columns:
        plot_intensity_on_ax(
            merged,
            field=wet_field,
            cmap_name=cfg["colormaps"]["wet"],
            ax=axes[1, 0]
        )
    else:
        axes[1, 0].set_axis_off()
        axes[1, 0].text(0.5, 0.5, f"Missing field: {wet_field}", ha="center", va="center")

    # --- Bottom-right: Dry year count + neighbors
    dry_field = "count_dry_neighbors"
    if dry_field in merged.columns:
        plot_intensity_on_ax(
            merged,
            field=dry_field,
            cmap_name=cfg["colormaps"]["dry"],
            ax=axes[1, 1]
        )
    else:
        axes[1, 1].set_axis_off()
        axes[1, 1].text(0.5, 0.5, f"Missing field: {dry_field}", ha="center", va="center")

    # Add panel letters (top-left of each subplot, no titles)
    panel_labels = [
        "(A) Wet Bivariate",
        "(B) Dry Bivariate",
        "(C) Wet Count",
        "(D) Dry Count",
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for label, (r, c) in zip(panel_labels, positions):
        axes[r, c].text(
            0.02, 0.98, label,
            transform=axes[r, c].transAxes,
            ha="left", va="top",
            fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2)
        )

    # Save the combined 4-panel figure
    out_png = os.path.join(outdir, "wet_dry_4panel.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved combined 4-panel figure: {out_png}")

if __name__ == "__main__":
    main()
