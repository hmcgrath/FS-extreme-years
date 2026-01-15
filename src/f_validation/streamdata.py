import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gauge_record_with_percentiles(
    xlsx_path,
    station_number,
    start_year,
    end_year,
    p_high=90,
    p_mid=65,
    outfolder=None
):
    """
    Plot daily gauge observations over time with percentile thresholds.
    """

    # -----------------------------
    # EXTRACT DATA
    # -----------------------------
    df, ylabel = extract_daily_observations(
        xlsx_path,
        station_number,
        start_year,
        end_year
    )

    df["year"] = df["date"].dt.year

    # -----------------------------
    # PERCENTILES
    # -----------------------------
    q_high = df["VALUE"].quantile(p_high / 100)
    q_mid = df["VALUE"].quantile(p_mid / 100)

    # -----------------------------
    # X-TICKS (adaptive)
    # -----------------------------
    year_span = end_year - start_year

    if year_span <= 15:
        step = 1
    elif year_span <= 40:
        step = 5
    else:
        step = 10

    xticks = np.arange(start_year, end_year + 1, step)

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    # Line + points (hydrograph style)
    ax.plot(
        df["year"],
        df["VALUE"],
        color="firebrick",
        linewidth=0.6,
        alpha=0.8,
        zorder=2
    )

    ax.scatter(
        df["year"],
        df["VALUE"],
        color="firebrick",
        s=8,
        alpha=0.8,
        zorder=3
    )

    # Percentile thresholds
    ax.axhline(
        q_high,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"{p_high}th percentile"
    )

    ax.axhline(
        q_mid,
        linestyle="--",
        color="grey",
        linewidth=1.2,
        label=f"{p_mid}th percentile"
    )

    ax.set_xlim(start_year, end_year)
    ax.set_xticks(xticks)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Gauge Record – Station {station_number}\n"
        f"{start_year}–{end_year}"
    )

    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        outfolder if outfolder else ".",
        f"gauge_record_{station_number}_{start_year}_{end_year}.png"
    ), dpi=300)
    plt.show()

    return {
        "station": station_number,
        "years_present": df["year"].nunique(),
        "n_observations": len(df),
        f"p{p_high}": q_high,
        f"p{p_mid}": q_mid
    }



def extract_daily_observations(
    xlsx_path,
    station_number,
    start_year,
    end_year=None
):
    if end_year is None:
        end_year = start_year

    xlsx_lower = xlsx_path.lower()
    if "levels" in xlsx_lower:
        value_col = "LEVEL"
        sheet_name = "DLY_LEVELS"
        ylabel = "Level (m)"
    elif "flows" in xlsx_lower:
        value_col = "FLOWS"
        sheet_name = "DLY_FLOWS"
        ylabel = "Discharge (m³/s)"
    else:
        raise ValueError("Cannot determine data type from XLSX filename.")

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    df = df[
        (df["STATION_NUMBER"] == station_number) &
        (df["YEAR"].between(start_year, end_year))
    ].copy()

    if df.empty:
        raise ValueError("No data found for given station and year range.")

    day_cols = [
        c for c in df.columns
        if c.startswith(value_col) and c[len(value_col):].isdigit()
    ]

    long_df = df.melt(
        id_vars=["STATION_NUMBER", "YEAR", "MONTH", "NO_DAYS"],
        value_vars=day_cols,
        var_name="DAY_COL",
        value_name=value_col
    )

    long_df["DAY"] = long_df["DAY_COL"].str.replace(
        value_col, "", regex=False
    ).astype(int)

    long_df = long_df[
        (long_df["DAY"] <= long_df["NO_DAYS"]) &
        (~long_df[value_col].isna())
    ]

    long_df["date"] = pd.to_datetime(dict(
        year=long_df["YEAR"],
        month=long_df["MONTH"],
        day=long_df["DAY"]
    ))

    out = (
        long_df[["date", value_col]]
        .rename(columns={value_col: "VALUE"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return out, ylabel


xlssheet = "D:\\Research\\FS-2dot0\\hydat\\DLY_LEVELS.xlsx"   # or DLY_FLOWS.xlsx
station_number = "01AL002" #01AL004 
#station_number = "05OG001" 
#station_number = "09AB001"
year_pre = 1990
year_post = 2023

stats = plot_gauge_record_with_percentiles(
    xlssheet,
    "09AB001",
    start_year=year_pre,
    end_year=year_post,
    p_high=99,
    p_mid=95,
    outfolder="D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement"
)

stats = plot_gauge_record_with_percentiles(
    xlssheet,
    "05OG002" ,
    start_year=year_pre,
    end_year=year_post,
    p_high=99,
    p_mid=95,
    outfolder="D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement"
)
stats = plot_gauge_record_with_percentiles(
    xlssheet,
    "05OG005" ,
    start_year=year_pre,
    end_year=year_post,
    p_high=99,
    p_mid=95,
    outfolder="D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement"
)
stats = plot_gauge_record_with_percentiles(
    xlssheet,
    "05OG008" ,
    start_year=year_pre,
    end_year=year_post,
    p_high=99,
    p_mid=95,
    outfolder="D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement"
)

stats = plot_gauge_record_with_percentiles(
    xlssheet,
    "09AB004" ,
    start_year=year_pre,
    end_year=year_post,
    p_high=99,
    p_mid=95,
    outfolder="D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement"
)

print(stats)

stats = plot_gauge_record_with_percentiles(
    xlssheet,
    "09AB010",
    start_year=year_pre,
    end_year=year_post,
    p_high=99,
    p_mid=95,
    outfolder="D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\supplement"
)

print(stats)
 