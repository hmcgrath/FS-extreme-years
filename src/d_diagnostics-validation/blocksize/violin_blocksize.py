
import os
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts\diagnostics-validation\blocksize -> up 2 -> scripts -> configs
    default_cfg = os.path.abspath(
        os.path.join(script_dir, "..", "..", "configs", "violin_blocksize.json")
    )
    cfg_path = os.environ.get("VIOLIN_BLOCKSIZE_CONFIG", default_cfg)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f), cfg_path

# ---------------------------------------------------------------------
# Helper: build uncertainty ranges and long-format data for plotting
# Uses templates from config to compute (hi - lo) for each block size.
# ---------------------------------------------------------------------
def build_plot_data(df, cfg):
    conds = cfg["logic"]["conditions"]
    sizes = cfg["logic"]["block_sizes"]
    cols = cfg["logic"]["columns"]
    labels = cfg["logic"]["labels"]

    pieces = []
    cats = []
    meta = []  # (condition, block_size, widths_series) for later summaries

    missing_msgs = []

    for cond in conds:
        for b in sizes:
            if cond.lower() == "wet":
                hi_col = cols["wet_hi_template"].format(b=b)
                lo_col = cols["wet_lo_template"].format(b=b)
                label = f"{labels['wet']} - {b}"
            elif cond.lower() == "dry":
                hi_col = cols["dry_hi_template"].format(b=b)
                lo_col = cols["dry_lo_template"].format(b=b)
                label = f"{labels['dry']} - {b}"
            else:
                raise ValueError(f"Unknown condition: {cond}")

            if hi_col not in df.columns or lo_col not in df.columns:
                missing_msgs.append(f"Missing columns for {label}: "
                                    f"{hi_col if hi_col not in df.columns else ''} "
                                    f"{lo_col if lo_col not in df.columns else ''}".strip())
                continue

            widths = (df[hi_col] - df[lo_col]).rename("Uncertainty Range")
            pieces.append(widths)
            cats.extend([label] * len(widths))
            meta.append((cond.lower(), b, widths))

    if missing_msgs:
        warnings.warn("Some block sizes/conditions skipped due to missing columns:\n - " +
                      "\n - ".join(missing_msgs))

    if not pieces:
        raise RuntimeError("No data available to plot after column checks.")

    long_df = pd.DataFrame({
        "Uncertainty Range": pd.concat(pieces, ignore_index=True),
        "Block Size": cats
    })

    return long_df, meta

# ---------------------------------------------------------------------
# Summaries: print to screen
# ---------------------------------------------------------------------
def print_summaries(df_raw, long_df, meta, cfg):
    print("\n" + "="*72)
    print("SUMMARY: Median Uncertainty Ranges by Condition & Block Size")
    print("="*72)

    # 1) Table of median uncertainty ranges across block sizes (wet & dry)
    #    Use the long_df labels ("Wet - 3", "Dry - 5", ...)
    med_table = (
        long_df.groupby("Block Size")["Uncertainty Range"].median()
        .rename("Median Uncertainty Range")
        .sort_index()
    )
    # Pretty print
    for label, med_val in med_table.items():
        print(f"{label:<10}  median_range = {med_val:.4f}")

    # 2) Median quantiles q_(med,wet) and q_(med,dry) from justification table
    #    If recommended_q_* columns exist, compute their medians across tiles.
    print("\n" + "="*72)
    print("SUMMARY: Median Quantiles (q_med)")
    print("="*72)
    q_wet = df_raw["recommended_q_wet"].median() if "recommended_q_wet" in df_raw.columns else np.nan
    q_dry = df_raw["recommended_q_dry"].median() if "recommended_q_dry" in df_raw.columns else np.nan
    print(f"q_(med,wet) = {q_wet:.4f}" if np.isfinite(q_wet) else "q_(med,wet) = NA")
    print(f"q_(med,dry) = {q_dry:.4f}" if np.isfinite(q_dry) else "q_(med,dry) = NA")

    # 3) 95% bootstrap CIs: classify as narrow/wide using config threshold
    #    For each condition & block size, compute CI widths and classify.
    print("\n" + "="*72)
    print("SUMMARY: 95% Bootstrap CI Widths — Narrow/Wide Classification")
    print("="*72)
    thr = cfg["logic"].get("ci_width_threshold", 0.15)  # default threshold
    print(f"(Classification threshold: width <= {thr:.3f} => 'narrow')")

    # Build count & median per condition/block
    for cond, b, widths in meta:
        label = f"{cond.capitalize()} - {b}"
        widths_clean = widths.replace([np.inf, -np.inf], np.nan).dropna()
        if widths_clean.empty:
            print(f"{label:<10}  (no data)")
            continue
        narrow = (widths_clean <= thr).sum()
        wide = (widths_clean > thr).sum()
        med_width = widths_clean.median()
        p_narrow = 100.0 * narrow / len(widths_clean)
        print(f"{label:<10}  median_width = {med_width:.4f} | narrow={narrow} ({p_narrow:.1f}%) | wide={wide}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg, cfg_path = load_config()

    # Seaborn / Matplotlib theming
    sns.set_theme(style=cfg["plot"].get("style", "whitegrid"),
                  context=cfg["plot"].get("context", "notebook"))

    # Load data
    csv_path = cfg["data"]["csv_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}\n(Check config at {cfg_path})")
    df = pd.read_csv(csv_path)

    # Prepare long-form data for violin plot + meta for summaries
    plot_data, meta = build_plot_data(df, cfg)

    # ---- NEW: print requested summaries to screen ----
    print_summaries(df, plot_data, meta, cfg)

    # Figure
    fig_w, fig_h = cfg["plot"].get("figure_size", [12, 6])
    plt.figure(figsize=(fig_w, fig_h))

    # Category order (optional)
    order = cfg["plot"].get("category_order", None)

    sns.violinplot(
        x="Block Size",
        y="Uncertainty Range",
        data=plot_data,
        inner=cfg["plot"].get("inner", "quartile"),
        palette=cfg["plot"].get("palette", "Set2"),
        order=order
    )

    # Titles / labels
    plt.title(cfg["plot"].get("title", ""), pad=10)
    plt.ylabel(cfg["plot"].get("y_label", "Uncertainty Range"))
    plt.xlabel(cfg["plot"].get("x_label", "Condition and Block Size"))

    # Grid
    grid_cfg = cfg["plot"].get("grid", {})
    if grid_cfg.get("enabled", True):
        plt.grid(True,
                 linestyle=grid_cfg.get("linestyle", "--"),
                 alpha=grid_cfg.get("alpha", 0.5))
    else:
        plt.grid(False)

    plt.tight_layout()

    # Output
    out_png = cfg["output"]["png_path"]
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=cfg["output"].get("dpi", 300))

    if cfg["output"].get("show", False):
        plt.show()
    else:
        plt.close()

    print("\n" + "="*72)
    print(f"Config used: {cfg_path}")
    print(f"Saved violin plot: {out_png}")
    print("="*72)

if __name__ == "__main__":
    main()
