#
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import to_rgba

# Load your watershed polygons GeoDataFrame (replace with your actual file)
gdf = gpd.read_file("D:\\Research\\flood-drivers\\Data\\new_wu_clip\\new_wu_clip.gpkg")

# Load the percentile_justification.csv data
df = pd.read_csv('D:\\Research\\FS-2dot0\\results\\newtop5\\2000-2023\\all-equalweights\\percentile_justification.csv', dtype={'tile_id': str})

# Merge GeoDataFrame with CSV data on tile_id
gdf = gdf.merge(df, on='tile_id', how='left')

#atlantic_gdf = gdf

def plot_map_of_years_focus_01(gdf, focus_prefix='01'):
    
    # Filter for Atlantic Canada watersheds (tile_id starts with '01')
    atlantic_gdf = gdf[gdf['tile_id'].str.startswith(focus_prefix)].copy()
    # Reproject to Web Mercator for consistency (even without basemap)
    #atlantic_gdf = atlantic_gdf.to_crs(epsg=3857)

    # Define columns for counts of extreme years and neighbors
    wet_count_col = 'wet_years_count'  # Number of selected wet years
    wet_neigh_col = 'wet_years_exp_count'  # Number of selected wet neighbor years
    dry_count_col = 'dry_years_count'  # Number of selected dry years
    dry_neigh_col = 'dry_years_exp_count'  # Number of selected dry neighbor years

    # Calculate total counts (selected + neighbors) for wet and dry
    atlantic_gdf['wet_total'] = atlantic_gdf[wet_count_col].fillna(0) # + atlantic_gdf[wet_neigh_col].fillna(0)
    atlantic_gdf['dry_total'] = atlantic_gdf[dry_count_col].fillna(0) # + atlantic_gdf[dry_neigh_col].fillna(0)

    # Notable tiles to highlight
    notable_tiles = ['01AD000', '01AJB00', '01AB000', '01AR000']

    # Create figure and axes for side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))

    # Plot wet extreme years map
    atlantic_gdf.plot(column='wet_total', cmap='Blues', linewidth=0.5, edgecolor='gray', legend=True,
                    legend_kwds={'label': "Total Wet Extreme Years", 'shrink': 0.6}, ax=ax1, vmax=6)
    ax1.set_title(f'Wet Extreme Years in major drainage basin: {focus_prefix}' , fontsize=16)

    #Highlight notable tiles on wet map
    notable_gdf = atlantic_gdf[atlantic_gdf['tile_id'].isin(notable_tiles)]
    notable_gdf.boundary.plot(ax=ax1, color='black', linewidth=2)
    for idx, row in notable_gdf.iterrows():
       x, y = row.geometry.centroid.coords[0]
       ax1.text(x, y, row['tile_id'], fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add scale bar and north arrow to wet map
    scalebar1 = ScaleBar(1, location='lower left')
    ax1.add_artist(scalebar1)
    ax1.annotate('N', xy=(0.95, 0.1), xytext=(0.95, 0.05),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=16, xycoords='axes fraction')

    ax1.set_axis_off()

    # Plot dry extreme years map
    atlantic_gdf.plot(column='dry_total', cmap='Reds', linewidth=0.5, edgecolor='gray', legend=True,
                    legend_kwds={'label': "Total Dry Extreme Years", 'shrink': 0.6}, ax=ax2, vmax=7)
    ax2.set_title(f'Dry Extreme Years in major drainage basin: {focus_prefix} ', fontsize=16)

    # Highlight notable tiles on dry map
    notable_gdf.boundary.plot(ax=ax2, color='black', linewidth=2)
    for idx, row in notable_gdf.iterrows():
        x, y = row.geometry.centroid.coords[0]
        ax2.text(x, y, row['tile_id'], fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add scale bar and north arrow to dry map
    scalebar2 = ScaleBar(1, location='lower left')
    ax2.add_artist(scalebar2)
    ax2.annotate('N', xy=(0.95, 0.1), xytext=(0.95, 0.05),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=16, xycoords='axes fraction')

    ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(f'D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\{focus_prefix}_extreme_years_map_wet_dry_sidebyside.png', dpi=300)
    plt.show()



def plot_map_(atlantic_gdf):
    
    # Reproject to Web Mercator for consistency (even without basemap)
    #atlantic_gdf = atlantic_gdf.to_crs(epsg=3857)

    # Define columns for counts of extreme years and neighbors
    wet_count_col = 'wet_years_count'  # Number of selected wet years
    wet_neigh_col = 'wet_years_exp_count'  # Number of selected wet neighbor years
    dry_count_col = 'dry_years_count'  # Number of selected dry years
    dry_neigh_col = 'dry_years_exp_count'  # Number of selected dry neighbor years

    # Calculate total counts (selected + neighbors) for wet and dry
    atlantic_gdf['wet_total'] = atlantic_gdf[wet_count_col].fillna(0) # + atlantic_gdf[wet_neigh_col].fillna(0)
    atlantic_gdf['dry_total'] = atlantic_gdf[dry_count_col].fillna(0) # + atlantic_gdf[dry_neigh_col].fillna(0)

    # Notable tiles to highlight
    notable_tiles = ['01AD000', '01AJB00', '01AB000', '01AR000']

    # Create figure and axes for side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))

    # Plot wet extreme years map
    atlantic_gdf.plot(column='wet_total', cmap='Blues', linewidth=0.5, edgecolor='gray', legend=True,
                    legend_kwds={'label': "Total Wet Extreme Years", 'shrink': 0.6}, ax=ax1, vmax=6)
    ax1.set_title('Wet Extreme Years Canada Watersheds', fontsize=16)

    # Highlight notable tiles on wet map
    #notable_gdf = atlantic_gdf[atlantic_gdf['tile_id'].isin(notable_tiles)]
    #notable_gdf.boundary.plot(ax=ax1, color='black', linewidth=2)
    #for idx, row in notable_gdf.iterrows():
    #    x, y = row.geometry.centroid.coords[0]
    #    ax1.text(x, y, row['tile_id'], fontsize=12, fontweight='bold', ha='center', va='center',
    #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add scale bar and north arrow to wet map
    scalebar1 = ScaleBar(1, location='lower left')
    ax1.add_artist(scalebar1)
    ax1.annotate('N', xy=(0.95, 0.1), xytext=(0.95, 0.05),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=16, xycoords='axes fraction')

    ax1.set_axis_off()

    # Plot dry extreme years map
    atlantic_gdf.plot(column='dry_total', cmap='Reds', linewidth=0.5, edgecolor='gray', legend=True,
                    legend_kwds={'label': "Total Dry Extreme Years", 'shrink': 0.6}, ax=ax2, vmax=7)
    ax2.set_title('Dry Extreme Years in Atlantic Canada Watersheds', fontsize=16)

    # # Highlight notable tiles on dry map
    # notable_gdf.boundary.plot(ax=ax2, color='black', linewidth=2)
    # for idx, row in notable_gdf.iterrows():
    #     x, y = row.geometry.centroid.coords[0]
    #     ax2.text(x, y, row['tile_id'], fontsize=12, fontweight='bold', ha='center', va='center',
    #             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add scale bar and north arrow to dry map
    scalebar2 = ScaleBar(1, location='lower left')
    ax2.add_artist(scalebar2)
    ax2.annotate('N', xy=(0.95, 0.1), xytext=(0.95, 0.05),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=16, xycoords='axes fraction')

    ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig('D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\extreme_years_map_wet_dry_sidebyside.png', dpi=300)
    plt.show()

##################
############

#plto maps for differnt focus: 
plot_map_of_years_focus_01(gdf, '01')
plot_map_of_years_focus_01(gdf, '02')
plot_map_of_years_focus_01(gdf, '03')
plot_map_of_years_focus_01(gdf, '04')
plot_map_of_years_focus_01(gdf, '05')
plot_map_of_years_focus_01(gdf, '06')
plot_map_of_years_focus_01(gdf, '07')
plot_map_of_years_focus_01(gdf, '08')
plot_map_of_years_focus_01(gdf, '09')
plot_map_of_years_focus_01(gdf, '10')
plot_map_of_years_focus_01(gdf, '11')
#plot canada wide map
plot_map_(gdf)
