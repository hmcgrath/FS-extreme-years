
import numpy as np
import pandas as pd

def best_change_point_year(years, values, min_seg=3, penalty=None):
    """
    One-change-point detector using SSE split.
    Returns (cp_year or None, sse_gain), where cp_year is the right-edge of the left segment.
    """
    y = np.asarray(values, dtype=float)
    t = np.asarray(years, dtype=int)
    n = y.size
    if n < 2*min_seg:
        return None, 0.0

    # Precompute cumulative sums for fast SSE
    c1 = np.cumsum(y)
    c2 = np.cumsum(y**2)

    def seg_sse(a, b):
        # SSE on interval [a, b) zero-based, b exclusive
        m = b - a
        if m <= 0:
            return np.inf
        s = c1[b-1] - (c1[a-1] if a > 0 else 0.0)
        q = c2[b-1] - (c2[a-1] if a > 0 else 0.0)
        mu = s / m
        return q - 2*mu*s + m*(mu**2)

    sse_total = seg_sse(0, n)

    # Evaluate all admissible splits
    ks = range(min_seg, n - min_seg + 1)
    sse_left = np.array([seg_sse(0, k) for k in ks])
    sse_right = np.array([seg_sse(k, n) for k in ks])
    sse_split = sse_left + sse_right

    # Best improvement
    idx = int(np.argmin(sse_split))
    k_best = list(ks)[idx]
    gain = sse_total - sse_split[idx]

    # Penalty: BIC-like default if not provided
    if penalty is None:
        # simple, conservative penalty scaled by variance and log n
        sigma2 = np.var(y, ddof=1) if n > 1 else 0.0
        penalty = sigma2 * np.log(n)  # tweakable

    if gain > penalty:
        return int(t[k_best-1]), float(gain)
    return None, 0.0


# Example: apply per WU (expects a long-form table with columns: tile_id, year, S_wet)
def compute_cp_per_wu(df_long, score_col="S_wet"):
    out = []
    for wu, g in df_long.sort_values(["tile_id", "year"]).groupby("tile_id"):
        years = g["year"].to_numpy()
        vals = g[score_col].to_numpy()
        cp, gain = best_change_point_year(years, vals, min_seg=3, penalty=None)
        out.append({"tile_id": wu, "cp_year": cp, "cp_gain": gain})
    return pd.DataFrame(out)

# Example usage
incsvscores = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\NonSupplement\\results\\wet_dry_scores.csv"
df_long = pd.read_csv(incsvscores)
df_cp = compute_cp_per_wu(df_long, score_col="wet_score")
df_cp.to_csv("D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\NonSupplement\\results\\wet_score_change_points.csv", index=False)