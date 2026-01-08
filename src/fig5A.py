
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from matplotlib.patches import Rectangle

# -----------------------------
# USER SETTINGS
# -----------------------------
INPUT_FILE = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\gisdata\\geometry_nhn.shp"
PRIMARY_FIELD = "primary"
SECONDARY_FIELD = "secondary"
OUTPUT_PNG = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\bivariate_choropleth.png"
# -----------------------------
# LOAD DATA
# -----------------------------
gdf = gpd.read_file(INPUT_FILE)

if not gdf.crs or not gdf.crs.is_projected:
    raise ValueError("⚠ Scalebar requires a projected CRS (meters).")

# -----------------------------
# COMPUTE DECADE DISTANCE
# -----------------------------
gdf["decade_dist"] = (gdf[PRIMARY_FIELD] - gdf[SECONDARY_FIELD]).abs() // 10
gdf["decade_dist"] = gdf["decade_dist"].clip(upper=2)

# -----------------------------
# BASE COLORS (HUE)
# -----------------------------
BASE_COLORS = {
    1990: "#800080",
    2000: "#0066cc",
    2010: "#00994c",
    2020: "#e67e22",
}

DECADE_LABELS = {
    1990: "1990s",
    2000: "2000s",
    2010: "2010s",
    2020: "2020s",
}

DISTANCE_LABELS = {
    0: "Same decade",
    1: "Adjacent decade",
    2: "Distant decade",
}

LIGHTNESS = {
    0: 1.0,
    1: 1.25,
    2: 0.75,
}

# -----------------------------
# COLOR TRANSFORM
# -----------------------------
def adjust_lightness(color, factor):
    rgb = np.array(to_rgb(color))
    return np.clip(rgb * factor, 0, 1)

# -----------------------------
# ASSIGN COLORS
# -----------------------------
gdf["plot_color"] = gdf.apply(
    lambda r: adjust_lightness(
        BASE_COLORS.get(r[PRIMARY_FIELD], "#999999"),
        LIGHTNESS[r["decade_dist"]],
    ),
    axis=1,
)

# -----------------------------
# FIGURE LAYOUT
# -----------------------------
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])

ax_map = fig.add_subplot(gs[0])
ax_leg = fig.add_subplot(gs[1])

# -----------------------------
# MAP
# -----------------------------
gdf.plot(
    ax=ax_map,
    color=gdf["plot_color"],
    linewidth=0.2,
    edgecolor="black",
)

ax_map.set_title(
    "Bivariate Choropleth\n"
    "Hue = Primary decade | Lightness = Secondary-decade distance",
    fontsize=14,
)
ax_map.set_axis_off()

# -----------------------------
# SCALEBAR FUNCTION
# -----------------------------
def add_scalebar(ax, length_fraction=0.25, height_fraction=0.015):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    map_width = xmax - xmin
    raw_length = map_width * length_fraction

    # Nice rounded lengths (meters)
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

    if length >= 1000:
        label = f"{int(length / 1000)} km"
    else:
        label = f"{int(length)} m"

    ax.text(
        x0 + length / 2,
        y0 + bar_height * 1.6,
        label,
        ha="center",
        va="bottom",
        fontsize=10
    )

# Add scalebar
add_scalebar(ax_map)

# -----------------------------
# BIVARIATE LEGEND
# -----------------------------
ax_leg.set_xlim(0, len(BASE_COLORS))
ax_leg.set_ylim(0, len(DISTANCE_LABELS))
ax_leg.invert_yaxis()
ax_leg.axis("off")

for col, decade in enumerate(BASE_COLORS.keys()):
    for row, dist in enumerate(DISTANCE_LABELS.keys()):
        ax_leg.add_patch(
            Rectangle(
                (col, row),
                1,
                1,
                facecolor=adjust_lightness(
                    BASE_COLORS[decade],
                    LIGHTNESS[dist]
                ),
                edgecolor="black",
            )
        )

for i, decade in enumerate(BASE_COLORS.keys()):
    ax_leg.text(i + 0.5, -0.4, DECADE_LABELS[decade],
                ha="center", va="top", fontsize=10, rotation=45)

for i, label in enumerate(DISTANCE_LABELS.values()):
    ax_leg.text(-0.1, i + 0.5, label,
                ha="right", va="center", fontsize=10)

ax_leg.text(len(BASE_COLORS) / 2, -1.1, "Primary decade",
            ha="center", va="top", fontsize=11, fontweight="bold")

ax_leg.text(-1.4, len(DISTANCE_LABELS) / 2,
            "Secondary decade\nrelationship",
            ha="center", va="center",
            fontsize=11, rotation=90, fontweight="bold")

# -----------------------------
# EXPORT
# -----------------------------
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print(f"✅ Map with bivariate legend and scalebar saved to {OUTPUT_PNG}")