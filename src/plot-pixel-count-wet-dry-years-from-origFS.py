#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "D:\\Research\\FS-2dot0\\results\\newtop5\\2000-2023\\all_years.csv"        # <-- update
OUTPUT_DIR = Path("D:\\Research\\FS-2dot0\\results\\newtop5\\2000-2023\\OrigFScounts-outputs")
GROUP_BY_PREFIX = True        # group by first 2 chars of tile_id
PREFIX_LEN = 2

OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH, sep=",", dtype={"tile_id": str})

# -------------------------
# RESHAPE TO LONG FORMAT
# -------------------------
value_cols = [c for c in df.columns if c != "tile_id"]

long = (
    df
    .melt(id_vars="tile_id", value_vars=value_cols,
          var_name="year_class", value_name="count")
)

# Split "1990_lt" → year=1990, class=lt
long[["year", "class"]] = long["year_class"].str.split("_", expand=True)
long["year"] = long["year"].astype(int)

# -------------------------
# PIVOT TO WIDE PER YEAR
# -------------------------
wide = (
    long
    .pivot_table(index=["tile_id", "year"],
                 columns="class",
                 values="count",
                 aggfunc="sum")
    .reset_index()
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_wet_points_trend_by_tile_group(wide_df, output_dir, selected_groups=None):
    """
    Plot Wet pixel count over time as points with a trend line,
    grouped by the first 2 characters of tile_id.
    Allows selection of specific tile groups.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['year', 'tile_id', 'gte', 'gt2'].
    output_dir : pathlib.Path or str
        Directory to save the figure.
    selected_groups : list of str, optional
        List of tile group IDs (first 2 chars) to include. If None, all groups are plotted.
        Example: ['01', '05', '10']
    """

    df = wide_df.copy()
    # Compute Wet pixels
    df['mid'] = df['gte'] - df['gt2']

    # Extract tile group (first 2 chars of tile_id)
    df['tile_group'] = df['tile_id'].str[:2]

    # Filter for selected groups if provided
    if selected_groups is not None:
        df = df[df['tile_group'].isin(selected_groups)]

    # Aggregate Wet pixels per year per group
    group_national = df.groupby(['year', 'tile_group'], as_index=False)['mid'].sum()
    group_national.rename(columns={'mid': 'wet_pixels'}, inplace=True)

    # Plot
    plt.figure(figsize=(14,6))
    sns.set_palette('tab10')  # up to 10 distinct groups

    # Scatter points
    sns.scatterplot(
        data=group_national,
        x='year',
        y='wet_pixels',
        hue='tile_group',
        s=60,
        palette='tab10'
    )

    # Add trend lines per group using linear regression
    for group in group_national['tile_group'].unique():
        group_data = group_national[group_national['tile_group'] == group]
        sns.regplot(
            data=group_data,
            x='year',
            y='wet_pixels',
            scatter=False,
            label=None,  # avoid duplicate legend
            line_kws={'lw':2, 'alpha':0.7},
        )

    plt.title("Wet Pixel Count Over Time by Selected Tile Groups")
    plt.xlabel("Year")
    plt.ylabel("Wet Pixels (Sum per Group)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Tile Group')
    plt.tight_layout()
    plt.savefig(output_dir / "wet_points_trend_selected_groups.png", dpi=300)
    plt.close()


def plot_national_wet_trend(wide_df, output_dir):
    """
    Plot a national Wet pixel count over time as points with a trend line.
    Wet pixels are computed as _mid = _gte - gt2, summed across all tiles per year.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['year', '_gte', 'gt2'].
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    # Compute Wet pixels per row
    df = wide_df.copy()
    df['mid'] = df['gte'] - df['gt2']

    # Aggregate nationally
    national = df.groupby('year', as_index=False)['gte'].sum()
    national.rename(columns={'gte': 'wetpixels'}, inplace=True)

    # Plot
    plt.figure(figsize=(14,5))
    sns.regplot(
        data=national,
        x='year',
        y='wetpixels',
        marker='o',
        scatter_kws={'s':50, 'color':'#00a2ff'},
        line_kws={'color':'#ff4500', 'lw':2}
    )

    plt.title("National Wet Pixel Count Over Time with Trend")
    plt.xlabel("Year")
    plt.ylabel("Wet Pixels (Sum of All Tiles)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "national_wet_trend.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_wet_trend_by_tile_group(wide_df, output_dir):
    """
    Plot Wet pixel count over time, grouped by the first 2 characters of tile_id,
    with points and trend lines for each group.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['year', 'tile_id', 'gte', 'gt2'].
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    df = wide_df.copy()
    # Compute Wet pixels
    df['mid'] = df['gte'] - df['gt2']

    # Extract tile group (first 2 chars of tile_id)
    df['tile_group'] = df['tile_id'].str[:2]

    # Aggregate Wet pixels per year per group
    group_national = df.groupby(['year', 'tile_group'], as_index=False)['mid'].sum()
    group_national.rename(columns={'mid': 'wet_pixels'}, inplace=True)

    # Plot
    plt.figure(figsize=(14,6))
    sns.set_palette('tab10')  # nice color palette for up to 10 groups
    sns.lineplot(
        data=group_national,
        x='year',
        y='wet_pixels',
        hue='tile_group',
        marker='o'
    )

    plt.title("Wet Pixel Count Over Time by Tile Group")
    plt.xlabel("Year")
    plt.ylabel("Wet Pixels (Sum per Group)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Tile Group')
    plt.tight_layout()
    plt.savefig(output_dir / "wet_trend_by_tile_group.png", dpi=300)
    plt.close()


import matplotlib.pyplot as plt
import pandas as pd

def plot_single_tile_wet_counts(wide_df, tile_id, output_dir):
    """
    Plot a yearly line chart of Wet pixel counts for a single tile.
    Computes _mid = _gte - gt2 for Wet pixels.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['tile_id', 'year', '_gte', 'gt2'].
    tile_id : str
        Tile ID to plot.
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    df_tile = wide_df[wide_df['tile_id'] == tile_id].copy()
    # Compute Wet pixels
    df_tile['mid'] = df_tile['gte'] - df_tile['gt2']

    # Ensure full year range
    years = list(range(1990, 2024))
    df_tile = df_tile.set_index('year').reindex(years, fill_value=0).reset_index()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df_tile['year'], df_tile['mid'], marker='o', color='#00a2ff', label='Wet Pixels')

    ax.set_title(f"Wet Pixel Counts Over Time — Tile {tile_id}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Wet Pixels")
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"tile_{tile_id}_wet_pixels.png", dpi=300)
    plt.close()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_tile_wet_dry_heatmap(wide_df, tile_ids, output_dir):
    """
    Plot heatmaps of Dry / Wet / Very Wet fractions over time for up to 3 tile_ids.
    
    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['tile_id', 'year', 'lt', 'gte', 'gt2'].
    tile_ids : list of str
        List of tile_ids to plot (max 3 recommended).
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    # Limit to requested tiles
    df_tiles = wide_df[wide_df['tile_id'].isin(tile_ids)].copy()
    
    # Compute fractions per tile per year
    df_tiles['total'] = df_tiles['lt'] + df_tiles['gte'] + df_tiles['gt2']
    df_tiles['dry_frac'] = df_tiles['lt'] / df_tiles['total']
    df_tiles['wet_frac'] = df_tiles['gte'] / df_tiles['total']
    df_tiles['very_wet_frac'] = df_tiles['gt2'] / df_tiles['total']

    # Prepare heatmap DataFrames
    fractions = ['dry_frac', 'wet_frac', 'very_wet_frac']
    titles = ['Dry', 'Wet', 'Very Wet']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for i, frac in enumerate(fractions):
        # Pivot: rows=tile_id, columns=year, values=fraction
        pivot = df_tiles.pivot(index='tile_id', columns='year', values=frac)
        
        sns.heatmap(
            pivot,
            ax=axes[i],
            cmap='YlGnBu' if frac != 'very_wet_frac' else 'Greens',
            annot=True,
            fmt=".2f",
            cbar=i == 2  # only show colorbar on last subplot
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Tile ID")

    plt.suptitle("Dry / Wet / Very Wet Fractions Over Time per Tile", fontsize=16)
    plt.savefig(output_dir / "tile_wet_dry_heatmap.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_tile_wet_dry_stacked_heatmap(wide_df, tile_ids, output_dir):
    """
    Plot a single stacked heatmap per tile showing Dry / Wet / Very Wet fractions over time.
    
    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['tile_id', 'year', 'lt', 'gte', 'gt2'].
    tile_ids : list of str
        List of tile_ids to plot (up to 3 recommended).
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    df_tiles = wide_df[wide_df['tile_id'].isin(tile_ids)].copy()
    df_tiles['total'] = df_tiles['lt'] + df_tiles['gte'] + df_tiles['gt2']
    df_tiles['dry_frac'] = df_tiles['lt'] / df_tiles['total']
    df_tiles['wet_frac'] = df_tiles['gte'] / df_tiles['total']
    df_tiles['very_wet_frac'] = df_tiles['gt2'] / df_tiles['total']

    fractions = ['dry_frac', 'wet_frac', 'very_wet_frac']
    colors = ['#f0a500', '#00a2ff', '#2ca02c']  # Dry, Wet, Very Wet

    # Prepare a 2D array: rows = tiles * 3 (stacked fractions), columns = years
    years = sorted(df_tiles['year'].unique())
    heatmap_array = []

    row_labels = []
    for tile in tile_ids:
        tile_data = df_tiles[df_tiles['tile_id'] == tile].set_index('year')
        for frac in fractions:
            row_labels.append(f"{tile} - {frac.split('_')[0].capitalize()}")
            heatmap_array.append([tile_data.loc[y, frac] if y in tile_data.index else np.nan for y in years])

    heatmap_array = np.array(heatmap_array)

    fig, ax = plt.subplots(figsize=(len(years) * 0.6 + 2, len(row_labels) * 0.5))
    im = ax.imshow(heatmap_array, aspect='auto', cmap='YlGnBu')

    # Color overlay per fraction
    # We'll use a custom mapping for Dry/Wet/Very Wet for clarity
    for i, frac in enumerate(fractions):
        cmap = plt.cm.get_cmap('YlOrBr' if frac=='dry_frac' else ('Blues' if frac=='wet_frac' else 'Greens'))
        start_row = i
        for j, tile in enumerate(tile_ids):
            row_idx = j * 3 + i
            ax.imshow(heatmap_array[row_idx:row_idx+1, :], aspect='auto', cmap=cmap, extent=[0, len(years), row_idx, row_idx+1])

    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_title("Stacked Dry / Wet / Very Wet Fractions per Tile")
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    plt.savefig(output_dir / "tile_wet_dry_stacked_heatmap.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_tile_wet_dry_continuous_heatmap(wide_df, tile_ids, output_dir):
    """
    Plot a compact heatmap of Dry / Wet / Very Wet fractions over time per tile.
    Each tile is a single row, with colors stacked per fraction per year.
    
    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['tile_id', 'year', 'lt', 'gte', 'gt2'].
    tile_ids : list of str
        List of tile_ids to plot (up to 3 recommended).
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    df_tiles = wide_df[wide_df['tile_id'].isin(tile_ids)].copy()
    df_tiles['total'] = df_tiles['lt'] + df_tiles['gte'] + df_tiles['gt2']
    df_tiles['dry_frac'] = df_tiles['lt'] / df_tiles['total']
    df_tiles['wet_frac'] = df_tiles['gte'] / df_tiles['total']
    df_tiles['very_wet_frac'] = df_tiles['gt2'] / df_tiles['total']

    years = sorted(df_tiles['year'].unique())
    fractions = ['dry_frac', 'wet_frac', 'very_wet_frac']
    colors = ['#f0a500', '#00a2ff', '#2ca02c']  # Dry, Wet, Very Wet

    fig, ax = plt.subplots(figsize=(len(years)*0.6 + 2, len(tile_ids)*0.8))

    for i, tile in enumerate(tile_ids):
        tile_data = df_tiles[df_tiles['tile_id'] == tile].set_index('year')
        bottom = np.zeros(len(years))
        for frac, color in zip(fractions, colors):
            values = np.array([tile_data.loc[y, frac] if y in tile_data.index else 0 for y in years])
            ax.bar(
                x=np.arange(len(years)),
                height=values,
                bottom=bottom,
                width=0.8,
                color=color,
                edgecolor='white'
            )
            bottom += values  # stack
        # label row
        ax.text(-0.5, i, tile, va='center', ha='right', fontsize=10)

    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks([])
    ax.set_xlim(-1, len(years))
    ax.set_ylim(0, len(tile_ids))
    ax.set_title("Dry / Wet / Very Wet Fractions Over Time per Tile")
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c,l in zip(colors, ['Dry','Wet','Very Wet'])]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / "tile_wet_dry_continuous_heatmap.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import pandas as pd

def plot_single_tile_stacked_bar(wide_df, tile_id, output_dir):
    """
    Plot a stacked bar chart of Dry / Wet / Very Wet fractions over time for a single tile.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Must contain columns: ['tile_id', 'year', 'lt', '_mid', 'gt2'].
    tile_id : str
        Tile ID to plot.
    output_dir : pathlib.Path or str
        Directory to save the figure.
    """

    df_tile = wide_df[wide_df['tile_id'] == tile_id].copy()
    #df_tile['_mid'] = df_tile['_gte'] - df_tile['gt2']
    # Ensure full year range
    years = list(range(1990, 2024))
    df_tile = df_tile.set_index('year').reindex(years, fill_value=0).reset_index()

    # Compute fractions
    df_tile['total'] = df_tile['lt'] + df_tile['mid'] + df_tile['gt2']
    df_tile['dry_frac'] = df_tile['lt'] / df_tile['total']
    df_tile['wet_frac'] = df_tile['mid'] / df_tile['total']
    df_tile['very_wet_frac'] = df_tile['gt2'] / df_tile['total']

    fig, ax = plt.subplots(figsize=(14,5))

    ax.bar(df_tile['year'], df_tile['dry_frac'], label='Dry', color='#f0a500')
    ax.bar(df_tile['year'], df_tile['wet_frac'], bottom=df_tile['dry_frac'], label='Wet', color='#00a2ff')
    ax.bar(df_tile['year'], df_tile['very_wet_frac'], 
           bottom=df_tile['dry_frac'] + df_tile['wet_frac'], label='Very Wet', color='#2ca02c')

    ax.set_title(f"Stacked Wet/Dry Fractions Over Time — Tile {tile_id}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of Pixels")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"tile_{tile_id}_stacked_bar.png", dpi=300)
    plt.close()


def plot_national_wet_dry(wide_df, output_dir):
    """
    Create a national stacked bar plot of dry vs wet pixels per year.
    Aggregates all tiles before computing fractions.
    """

    national = (
        wide_df
        .groupby("year", as_index=False)[["lt", "gte"]]
        .sum()
    )

    national["total"] = national["lt"] + national["gte"]
    national["dry_pct"] = national["lt"] / national["total"]
    national["wet_pct"] = national["gte"] / national["total"]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(national["year"],
           national["dry_pct"],
           label="Dry",
           alpha=0.7)

    ax.bar(national["year"],
           national["wet_pct"],
           bottom=national["dry_pct"],
           label="Wet",
           alpha=0.7)

    ax.set_title("National Wet vs Dry Fraction Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of pixels")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "bar_national_wet_dry.png", dpi=300)
    plt.close()



# Ensure all expected columns exist
for c in ["lt", "mid", "gt2", "gte"]:
    if c not in wide.columns:
        wide[c] = 0

# -------------------------
# CALCULATE TOTALS & %
# -------------------------
wide["total"] = wide["lt"] + wide["gte"]

wide["dry_pct"]       = wide["lt"]  / wide["total"]
wide["wet_pct"]       = wide["gte"] / wide["total"]
wide["very_wet_pct"]  = wide["gt2"] / wide["total"]
wide["mid_wet_pct"]   = (wide["gte"] - wide["gt2"]) / wide["total"]

# -------------------------
# OPTIONAL GROUPING BY TILE PREFIX
# -------------------------
if GROUP_BY_PREFIX:
    wide["group"] = wide["tile_id"].str[:PREFIX_LEN]
else:
    wide["group"] = wide["tile_id"]

grouped = (
    wide
    .groupby(["group", "year"], as_index=False)
    .mean(numeric_only=True)
)

# -------------------------
# BAR PLOT: WET VS DRY OVER TIME
# -------------------------
for grp, sub in grouped.groupby("group"):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(sub["year"], sub["dry_pct"], label="Dry", alpha=0.7)
    ax.bar(sub["year"], sub["wet_pct"],
           bottom=sub["dry_pct"],
           label="Wet", alpha=0.7)

    ax.set_title(f"Wet vs Dry Fraction Over Time (Group {grp})")
    ax.set_ylabel("Fraction of pixels")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"bar_wet_dry_group_{grp}.png", dpi=300)
    plt.close()

# -------------------------
# HEAT MAP: WET FRACTION
# -------------------------
heat = (
    grouped
    .pivot(index="group", columns="year", values="wet_pct")
    .sort_index()
)

fig, ax = plt.subplots(figsize=(14, 6))

im = ax.imshow(heat.values, aspect="auto")

ax.set_yticks(np.arange(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_xticks(np.arange(len(heat.columns)))
ax.set_xticklabels(heat.columns, rotation=90)

ax.set_title("Wet Fraction Heat Map")
ax.set_xlabel("Year")
ax.set_ylabel("Tile Group")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Wet fraction")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_wet_fraction.png", dpi=300)
plt.close()


plot_national_wet_dry(wide, OUTPUT_DIR)
#plot_tile_wet_dry_heatmap(wide, ["01AL000", "05OG000", "09AB000"], OUTPUT_DIR)
#plot_tile_wet_dry_continuous_heatmap(wide, ["01AL000", "05OG000", "09AB000"], OUTPUT_DIR)

plot_single_tile_stacked_bar(wide, "01AD000", OUTPUT_DIR)
plot_single_tile_stacked_bar(wide, "01AJB00", OUTPUT_DIR)
plot_single_tile_stacked_bar(wide, "01AB000", OUTPUT_DIR)
plot_single_tile_stacked_bar(wide, "05OG000", OUTPUT_DIR)
plot_single_tile_stacked_bar(wide, "01AR000", OUTPUT_DIR)
plot_single_tile_wet_counts(wide, "01AD000", OUTPUT_DIR)
plot_single_tile_wet_counts(wide, "01AD000", OUTPUT_DIR)
plot_single_tile_wet_counts(wide, "01AD000", OUTPUT_DIR)



plot_national_wet_trend(wide, OUTPUT_DIR)
plot_wet_trend_by_tile_group(wide, OUTPUT_DIR)
plot_wet_points_trend_by_tile_group(wide, OUTPUT_DIR, selected_groups=["01", "05", "09"])
print("✔ Analysis complete. Outputs written to:", OUTPUT_DIR)
