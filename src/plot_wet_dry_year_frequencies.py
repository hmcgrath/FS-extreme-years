import os 
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# USER INPUT
# -----------------------------
CSV_PATH = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement\\1990-2023percentile_justification.csv"  # <-- set to your justification CSV path
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors

# -----------------------------
# USER INPUT
# -----------------------------
CSV_PATH = CSV_PATH

outfolder = r"D:\Research\FS-2dot0\results\WetDryTrendsPaper\supplement"
os.makedirs(outfolder, exist_ok=True)

WET_COL = "selected_wet_years"
DRY_COL = "selected_dry_years"
TILE_COL = "tile_id"

# -----------------------------
# WU COLOR MAP
# -----------------------------
WU_COLORS = {
    "01": "#FFD700",
    "02": "#F4C2C2",
    "03": "#2AA198",
    "04": "#FFA500",
    "05": "#98FF98",
    "06": "#F6C1CC",
    "07": "#B784A7",
    "08": "#007BFF",
    "09": "#ADD8E6",
    "10": "#A8D5BA",
    "11": "#FA8072",
}

# -----------------------------
# HELPERS
# -----------------------------
def parse_years(val):
    if pd.isna(val) or val.strip() == "":
        return []
    return [int(y) for y in val.split(";") if y.strip().isdigit()]


def darken_color(color, factor=0.85):
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)


# -----------------------------
# CORE FREQUENCY CALCULATION
# -----------------------------
def compute_year_frequencies(df, year_col):
    year_counts = defaultdict(lambda: defaultdict(int))
    wu_totals = df.groupby("WU").size().to_dict()

    for _, row in df.iterrows():
        wu = row["WU"]
        years = parse_years(row[year_col])
        for y in years:
            year_counts[y][wu] += 1

    records = []
    for year, wu_dict in year_counts.items():
        for wu, count in wu_dict.items():
            records.append({
                "year": year,
                "WU": wu,
                "percent": 100 * count / wu_totals.get(wu, 1)
            })

    return pd.DataFrame(records)


# -----------------------------
# 1️⃣ COLLAPSE INTO DECADES
# -----------------------------
def collapse_to_decades(freq_df):
    df = freq_df.copy()
    df["decade"] = (df["year"] // 10) * 10

    decade_df = (
        df.groupby(["decade", "WU"], as_index=False)
        .agg({"percent": "mean"})
    )

    return decade_df


# -----------------------------
# PLOTTING (ALL WUs)
# -----------------------------
def plot_grouped_bars(freq_df, x_col, title):
    xvals = sorted(freq_df[x_col].unique())
    wus = sorted(freq_df["WU"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))

    width = 0.9 / len(wus)

    for i, wu in enumerate(wus):
        subset = freq_df[freq_df["WU"] == wu]
        lookup = dict(zip(subset[x_col], subset["percent"]))

        values = [lookup.get(x, 0) for x in xvals]

        color = darken_color(WU_COLORS.get(wu, "#999999"))

        ax.bar(
            [j + i * width for j in range(len(xvals))],
            values,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=f"WU {wu}"
        )

    ax.set_xticks([j + width * (len(wus) / 2) for j in range(len(xvals))])
    ax.set_xticklabels(xvals, rotation=45)
    ax.set_ylabel("Percent of work units (%)")
    ax.set_xlabel(x_col.capitalize())
    ax.set_title(title)

    # Subtle grid
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)

    ax.legend(ncol=3, title="Work Unit")
    plt.tight_layout()

    plt.savefig(
        os.path.join(outfolder, f"{title.replace(' ', '_').lower()}.png"),
        dpi=300
    )
    plt.show()


# -----------------------------
# 4️⃣ ONE-PLOT-PER-WU
# -----------------------------
def plot_per_wu(freq_df, x_col, title_prefix):
    for wu in sorted(freq_df["WU"].unique()):
        subset = freq_df[freq_df["WU"] == wu].sort_values(x_col)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            subset[x_col],
            subset["percent"],
            width=0.8,
            color=darken_color(WU_COLORS.get(wu, "#999999")),
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_title(f"{title_prefix} – WU {wu}")
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel("Percent of work units (%)")

        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)

        plt.tight_layout()

        fname = f"{title_prefix}_wu_{wu}".replace(" ", "_").lower() + ".png"
        plt.savefig(os.path.join(outfolder, fname), dpi=300)
        plt.show()


# -----------------------------
# LOAD & RUN
# -----------------------------
df = pd.read_csv(CSV_PATH, dtype=str)
df["WU"] = df[TILE_COL].str[:2]

wet_years = compute_year_frequencies(df, WET_COL)
dry_years = compute_year_frequencies(df, DRY_COL)

wet_decades = collapse_to_decades(wet_years)
dry_decades = collapse_to_decades(dry_years)

# Combined plots
plot_grouped_bars(wet_decades, "decade", "Wet Years by Decade")
plot_grouped_bars(dry_decades, "decade", "Dry Years by Decade")

# Per-WU plots
plot_per_wu(wet_decades, "decade", "Wet Years by Decade")
plot_per_wu(dry_decades, "decade", "Dry Years by Decade")
