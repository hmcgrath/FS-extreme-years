#scoring from mktau and mk_p. 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute NetTrend and mapping-ready classifications from Mann–Kendall outputs,
and produce summary figures and tables.

Inputs:
  - A CSV with columns:
      tile_id, mk_tau_wet, mk_p_wet, mk_tau_dry, mk_p_dry
  - A watershed polygon layer (e.g., shapefile) that contains 'tile_id' for join

Outputs (to --outdir):
  - net_trend_classification.csv
  - net_trend_summary_table.csv
  - net_trend_summary.png          (scatter, hist, class counts, map)
  - net_trend_map.png              (standalone choropleth)
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# geopandas and mapclassify are commonly used for spatial plotting
try:
    import geopandas as gpd
except ImportError as e:
    gpd = None

# --------------------------
# Utilities
# --------------------------

def bh_fdr(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR adjustment.
    Returns array of q-values (adjusted p-values) in the same order.

    Reference: Benjamini & Hochberg (1995).
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    # cumulative minimum of (p_i * n / i) from the right
    prev = 1.0
    for i in range(n, 0, -1):
        val = ranked[i-1] * n / i
        if val > prev:
            val = prev
        prev = val
        q[i-1] = val
    # back to original order
    q_full = np.empty_like(q)
    q_full[order] = q
    return q_full

def weight_from_p(p, cap=3.0):
    """
    Convert p-value to a 0..1 weight using -log10(p) capped, then scaled.
    p<=0 handled by epsilon.
    """
    eps = 1e-16
    p = max(p, eps)
    w = min(cap, -math.log10(p)) / cap
    return w

def classify_row(row, q_alpha_strict=0.05, q_alpha_loose=0.10):
    """
    Classification logic using τ and FDR-adjusted p-values (q).
    - Strong/Moderate Wetting or Drying
    - Mixed significant (both wet and dry significant but opposing signals)
    - Inconclusive
    """
    tw, td = row['mk_tau_wet'], row['mk_tau_dry']
    qw, qd = row['mk_q_wet'], row['mk_q_dry']

    # Mixed significant if both are significant at 0.05 and signs conflict
    if (qw < q_alpha_strict) and (qd < q_alpha_strict) and ((tw > 0 and td > 0) or (tw < 0 and td < 0) or (tw*td < 0)):
        # Any situation where both are significant but point to different or complex directions
        return "Mixed significant"

    # Strong wetting: wet ↑ significant, and no counteracting significant dry trend
    if (tw > 0 and qw < q_alpha_strict) and not (td > 0 and qd < q_alpha_strict):
        return "Strong wetting"

    # Strong drying: dry ↓ significant (i.e., fewer dry scores over time would be wetting;
    # but here we interpret 'drying' as dry scores increasing (td > 0) or wet decreasing (tw < 0).
    # To keep interpretation intuitive, define drying as: wet τ <= 0 significant OR dry τ >= 0 significant indicating more dryness.
    # However, users often define 'drying' by wet τ < 0 or dry τ > 0. We use that pair:
    if ((tw < 0 and qw < q_alpha_strict) or (td > 0 and qd < q_alpha_strict)):
        return "Strong drying"

    # Moderate wetting/drying (looser threshold)
    if (tw > 0 and qw < q_alpha_loose) and not (td > 0 and qd < q_alpha_loose):
        return "Moderate wetting"
    if (tw < 0 and qw < q_alpha_loose) or (td > 0 and qd < q_alpha_loose):
        return "Moderate drying"

    return "Inconclusive"

def build_parser():
    p = argparse.ArgumentParser(description="Compute NetTrend and classifications from MK outputs and plot maps.")
    p.add_argument("--incsv", required=True, help="Input CSV with columns: tile_id, mk_tau_wet, mk_p_wet, mk_tau_dry, mk_p_dry")
    p.add_argument("--shapefile", required=True, help="Path to watershed polygon file (e.g., nhn_land.shp) with 'tile_id' field")
    p.add_argument("--outdir", required=True, help="Directory to write outputs")
    p.add_argument("--join_key_csv", default="tile_id", help="Join key in CSV (default: tile_id)")
    p.add_argument("--join_key_shp", default="tile_id", help="Join key in shapefile (default: tile_id)")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR alpha for significance display (default: 0.05)")
    p.add_argument("--title_prefix", default="Net Wetting", help="Title prefix for plots")
    return p

# --------------------------
# Main procedure
# --------------------------

def main():
    
    
    incsv =  "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\working\\processed\\2000-2023percentile_justification_processed.csv" 
    shapefile = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\gisdata\\geometry_nhn.shp" 
    outdir = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\net_trend_results" 
    join_key_csv = "tile_id" 
    join_key_shp = "tile_id" 
    alpha = 0.05
    title_prefix = "Canada Watersheds Net Trend" 

    os.makedirs(outdir, exist_ok=True)

    # ---- Read inputs
    df = pd.read_csv(incsv)
    needed = [join_key_csv, "mk_tau_wet", "mk_p_wet", "mk_tau_dry", "mk_p_dry"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # ---- Basic checks & cleaning
    df = df.copy()
    # coerce to numeric
    for c in ["mk_tau_wet", "mk_p_wet", "mk_tau_dry", "mk_p_dry"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows with all NaN tau or p
    df = df.dropna(subset=["mk_tau_wet", "mk_p_wet", "mk_tau_dry", "mk_p_dry"], how="any").copy()

    # ---- FDR adjustment
    df["mk_q_wet"] = bh_fdr(df["mk_p_wet"].values)
    df["mk_q_dry"] = bh_fdr(df["mk_p_dry"].values)

    # ---- NetTrend and weighted NetTrend
    df["NetTrend_tau"] = df["mk_tau_wet"] - df["mk_tau_dry"]

    # significance weights
    df["w_wet"] = df["mk_p_wet"].apply(weight_from_p)
    df["w_dry"] = df["mk_p_dry"].apply(weight_from_p)
    df["NetTrend_weighted"] = df["mk_tau_wet"] * df["w_wet"] - df["mk_tau_dry"] * df["w_dry"]

    # ---- Classification
    df["trend_class"] = df.apply(classify_row, axis=1)

    # Flag “significant overall” if either wet or dry is significant at FDR alpha
    df["is_significant_any"] = (df["mk_q_wet"] < alpha) | (df["mk_q_dry"] < alpha)

    # ---- Save mapping-ready CSV
    out_csv = os.path.join(outdir, "net_trend_classification.csv")
    cols = [
        join_key_csv, "mk_tau_wet", "mk_p_wet", "mk_q_wet",
        "mk_tau_dry", "mk_p_dry", "mk_q_dry",
        "NetTrend_tau", "NetTrend_weighted", "trend_class", "is_significant_any"
    ]
    df[cols].to_csv(out_csv, index=False)

    # ---- Supplementary table (class counts & summary stats)
    class_counts = df["trend_class"].value_counts(dropna=False).rename_axis("trend_class").reset_index(name="count")
    summary = df[["NetTrend_tau", "NetTrend_weighted"]].describe().T.reset_index().rename(columns={"index": "metric"})
    supp = {
        "class_counts": class_counts,
        "summary_stats": summary
    }
    out_table = os.path.join(outdir, "net_trend_summary_table.csv")
    # Save a simple joined table (counts then a blank row then summary)
    with open(out_table, "w", encoding="utf-8") as f:
        class_counts.to_csv(f, index=False)
        f.write("\n")
        summary.to_csv(f, index=False)


    # ---- Figures: Scatter, distributions, class counts, map
    mpl.rcParams.update({"figure.dpi": 150})
    fig = plt.figure(figsize=(13, 9))

    # Make the right column (C and E) wider AND increase the gap before it.
    # This both enlarges (E) and visually moves (C) to the right.
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.25],
        width_ratios=[1.0, 1.2, 1.40],   # <-- was [1, 1, 1]; right column widened
        hspace=0.32,
        wspace=0.35                     # <-- add a bit more gap between B and C
    )

    
    # (A) Scatter τ_wet vs τ_dry with NetTrend contours
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(df["mk_tau_wet"], df["mk_tau_dry"], s=10, c=df["NetTrend_tau"],
                cmap="BrBG", vmin=-1, vmax=1, alpha=0.8, edgecolors="none")
    axA.axhline(0, color="k", lw=0.8)
    axA.axvline(0, color="k", lw=0.8)
    axA.set_xlabel("Kendall τ (Wet)")
    axA.set_ylabel("Kendall τ (Dry)")
    axA.set_title("(A) τ Wet vs τ Dry (color = NetTrend)")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap="BrBG"), ax=axA)
    cbar.set_label("NetTrend (τ_wet − τ_dry)")

    # (B) Histograms of τ_wet and τ_dry
    axB = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-1, 1, 41)
    axB.hist(df["mk_tau_wet"], bins=bins, alpha=0.6, label="Wet τ", color="#1b9e77")
    axB.hist(df["mk_tau_dry"], bins=bins, alpha=0.6, label="Dry τ", color="#d95f02")
    axB.axvline(0, color="k", lw=0.8)
    axB.set_xlabel("Kendall τ")
    axB.set_ylabel("Count")
    axB.set_title("(B) τ distributions")
    axB.legend(frameon=False)


    # (C) Trend-class counts — horizontal orientation
    axC = fig.add_subplot(gs[0, 2])
    order = ["Strong wetting", "Moderate wetting", "Inconclusive",
            "Mixed significant", "Moderate drying", "Strong drying"]
    counts = df["trend_class"].value_counts().reindex(order).fillna(0).astype(int)

    axC.barh(
        counts.index, counts.values,
        color=["#2166ac", "#67a9cf", "#bababa", "#762a83", "#f4a582", "#b2182b"]
    )
    axC.set_xlabel("Watersheds")
    axC.set_ylabel("Trend class")
    axC.set_title("(C) Trend-class counts")
    axC.set_ylim(-0.5, len(counts.index) - 0.5)
    axC.margins(y=0.05)

    # ---- Key nudges to move C further to the RIGHT within its own cell ----
    # Hug the right-hand side of its slot
    axC.set_anchor('E')  # 'E' = east edge; keeps the axes tight to the right

    # If you want an extra shove to the right (fine-tune the position box):
    from matplotlib.transforms import Bbox
    pos = axC.get_position()
    # Shift right by 1.5% of figure width and reduce width slightly to avoid overlap
    axC.set_position(Bbox.from_bounds(pos.x0 + 0.015, pos.y0, pos.width - 0.015, pos.height))


    # (D) & (E) Map(s)
    # Read shapefile only if geopandas is available
    map_png = None
    if gpd is not None:
        gdf = gpd.read_file(shapefile)
        if join_key_shp not in gdf.columns:
            raise ValueError(f"Shapefile is missing join key '{join_key_shp}'")
        # Ensure same dtype for join keys
        gdf[join_key_shp] = gdf[join_key_shp].astype(str)
        df[join_key_csv] = df[join_key_csv].astype(str)
        gdf_merged = gdf.merge(df, left_on=join_key_shp, right_on=join_key_csv, how="left")

        # (D) Choropleth of NetTrend
        axD = fig.add_subplot(gs[1, :2])
        gdf_merged.plot(column="NetTrend_tau",
                        cmap="BrBG", vmin=-1, vmax=1, linewidth=0.1, edgecolor="0.5",
                        legend=True, ax=axD, missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No data"})
        # Overlay hatching or outlines for non-significant
        try:
            notsig = gdf_merged[~gdf_merged["is_significant_any"].fillna(False)]
            notsig.boundary.plot(ax=axD, color="k", linewidth=0.2, alpha=0.4)
        except Exception:
            pass
        axD.set_title("(D) NetTrend (τ_wet − τ_dry); outline = not significant (FDR {:.2f})".format(alpha))
        axD.set_axis_off()

                

                
        # (E) Map of trend_class (categorical) — LEGEND OUTSIDE
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        axE = fig.add_subplot(gs[1, 2])

        class_palette = {
            "Strong wetting":   "#2166ac",
            "Moderate wetting": "#67a9cf",
            "Inconclusive":     "#bababa",
            "Mixed significant":"#762a83",
            "Moderate drying":  "#f4a582",
            "Strong drying":    "#b2182b",
        }
        cats = list(class_palette.keys())
        gdf_merged["trend_class"] = pd.Categorical(gdf_merged["trend_class"], categories=cats)

        # Build a ListedColormap and plot without legend (we’ll add a manual legend)
        cmap_cat = ListedColormap([class_palette[c] for c in cats])

        gdf_merged.plot(
            column="trend_class",
            ax=axE,
            categorical=True,
            cmap=cmap_cat,
            linewidth=0.1,
            edgecolor="0.5",
            legend=False,  # IMPORTANT: turn off GeoPandas' legend to avoid overlay
            missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No data"},
        )

        # Title and aesthetics
        axE.set_title("(E) Trend class (categorical)")
        axE.set_axis_off()

        # ---------- Build a manual legend OUTSIDE the Axes ----------
        legend_handles = [Patch(facecolor=class_palette[c], edgecolor="0.3", label=c) for c in cats]
        # Add "No data" entry
        legend_handles.append(Patch(facecolor="lightgrey", edgecolor="0.3", hatch="///", label="No data"))

        # Place the legend to the RIGHT of the map (outside the axes)
        # bbox_to_anchor: (x, y, width, height) in axes coordinates for the legend's bounding box
        # loc="center left" puts the legend's left-center point at the anchor x,y
        leg = axE.legend(
            handles=legend_handles,
            title="Trend class",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),   # push slightly to the right of the axes
            frameon=True,
            borderpad=0.6,
            labelspacing=0.5,
            handlelength=1.4,
            handletextpad=0.6,
            fontsize=9,
            title_fontsize=10,
        )

        # Ensure the figure layout leaves room for the external legend
        # (tight_layout will respect the legend bbox in most cases)



        axE.set_title("(E) Trend class (categorical)")
        axE.set_axis_off()

        # Save a standalone map as well
        map_png = os.path.join(outdir, "net_trend_map.png")
        fig_map, ax = plt.subplots(figsize=(8, 6), dpi=300)
        gdf_merged.plot(column="NetTrend_tau",
                        cmap="BrBG", vmin=-1, vmax=1, linewidth=0.2, edgecolor="0.6",
                        legend=True, ax=ax, missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No data"})
        ax.set_title(f"{title_prefix}: NetTrend (τ_wet − τ_dry)")
        ax.set_axis_off()
        fig_map.tight_layout()
        fig_map.savefig(map_png, bbox_inches="tight")
        plt.close(fig_map)



    else:
        # Geopandas not available — leave map panels blank but proceed
        axD = fig.add_subplot(gs[1, :2])
        axD.text(0.5, 0.5, "GeoPandas not installed.\nMap skipped.", ha="center", va="center", fontsize=12)
        axD.axis("off")
        axE = fig.add_subplot(gs[1, 2])
        axE.axis("off")

    # Save the summary figure
    out_fig = os.path.join(outdir, "net_trend_summary.png")
    fig.suptitle(f"{title_prefix}: Summary", y=0.98, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_table}")
    print(f"[OK] Wrote: {out_fig}")
    if map_png is not None:
        print(f"[OK] Wrote: {map_png}")



    # --- q-value maps for supplement ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    gdf_merged.plot(
        column="mk_q_wet",
        cmap="viridis_r",
        vmin=0, vmax=0.2,
        linewidth=0.1,
        ax=axes[0],
        legend=True,
    )
    axes[0].set_title("Wet-season FDR-adjusted p (q)")
    axes[0].axis("off")

    gdf_merged.plot(
        column="mk_q_dry",
        cmap="viridis_r",
        vmin=0, vmax=0.2,
        linewidth=0.1,
        ax=axes[1],
        legend=True,
    )
    axes[1].set_title("Dry-season FDR-adjusted p (q)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "q_value_maps.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    #parser = build_parser()
    #args = parser.parse_args()
    main()

