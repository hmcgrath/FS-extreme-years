
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple
from dataclasses import dataclass
from scipy.stats import genextreme as gev
from scipy.optimize import minimize


@dataclass
class GEVFit:
    model: str                      # 'stationary' | 'mu' | 'mu+sigma'
    theta: np.ndarray               # fitted parameter vector
    loglik: float                   # maximized log-likelihood
    aic: float
    bic: float
    xi: float                       # constant shape parameter
    mu_t: np.ndarray                # mu at each time t (vector of len n)
    sigma_t: np.ndarray             # sigma at each time t (vector of len n)
    rl_t: np.ndarray                # return level at prob p for each t (len n)
    selected_years: np.ndarray      # boolean mask (y >= rl_t)


def _center_time(t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center time to reduce collinearity."""
    t0 = float(np.mean(t))
    return (t - t0), t0


def _negloglik(theta: np.ndarray, y: np.ndarray, t: np.ndarray, model: str) -> float:
    """
    Negative log-likelihood for GEV with time-varying location (and optionally scale).
    Shape xi is constant. Scale is >0 via log-link. Time is centered.
    Parameterizations:
      stationary : theta = [a0, log_sigma, xi]
      mu         : theta = [a0, a1, log_sigma, xi]
      mu+sigma   : theta = [a0, a1, log_sigma, b1, xi]
    """
    # unpack & build mu_t, sigma_t, xi
    if model == "stationary":
        a0, log_sig, xi = theta
        mu_t = np.full_like(y, fill_value=a0, dtype=float)
        sigma_t = np.exp(log_sig) * np.ones_like(y, dtype=float)
    elif model == "mu":
        a0, a1, log_sig, xi = theta
        mu_t = a0 + a1 * t
        sigma_t = np.exp(log_sig) * np.ones_like(y, dtype=float)
    elif model == "mu+sigma":
        a0, a1, log_sig, b1, xi = theta
        mu_t = a0 + a1 * t
        sigma_t = np.exp(log_sig + b1 * t)
    else:
        raise ValueError("Unknown model")

    # invalid if any sigma <= 0
    if np.any(~np.isfinite(sigma_t)) or np.any(sigma_t <= 0):
        return np.inf

    # domain check for GEV: z = 1 + xi*(y - mu)/sigma must be > 0
    # Let SciPy handle logpdf (it returns -inf outside support), but guard for nan/inf
    c = -xi  # SciPy's c = -xi
    ll = np.sum(gev.logpdf(y, c=c, loc=mu_t, scale=sigma_t))
    if not np.isfinite(ll):
        return np.inf
    return -ll


def _fit_model(y: np.ndarray, t: np.ndarray, p: float, model: str) -> GEVFit:
    """
    Fit one of: 'stationary', 'mu', 'mu+sigma'; return AIC/BIC and RL curve at quantile p.
    """
    # initial guesses from stationary fit
    c0, loc0, scale0 = gev.fit(y)               # SciPy: c = -xi
    xi0 = -c0
    log_sig0 = np.log(scale0)

    if model == "stationary":
        theta0 = np.array([loc0, log_sig0, xi0], dtype=float)
        k = 3
        bounds = [(-np.inf, np.inf),  # a0
                  (np.log(1e-6), np.log(1e6)),  # log_sigma
                  (-0.9, 0.9)]  # xi (practical box to aid convergence)
    elif model == "mu":
        theta0 = np.array([loc0, 0.0, log_sig0, xi0], dtype=float)
        k = 4
        bounds = [(-np.inf, np.inf),  # a0
                  (-10, 10),          # a1
                  (np.log(1e-6), np.log(1e6)),  # log_sigma
                  (-0.9, 0.9)]        # xi
    elif model == "mu+sigma":
        theta0 = np.array([loc0, 0.0, log_sig0, 0.0, xi0], dtype=float)
        k = 5
        bounds = [(-np.inf, np.inf),  # a0
                  (-10, 10),          # a1
                  (np.log(1e-6), np.log(1e6)),  # log_sigma
                  (-10, 10),          # b1
                  (-0.9, 0.9)]        # xi
    else:
        raise ValueError("Unknown model")

    res = minimize(_negloglik, theta0, args=(y, t, model), method="L-BFGS-B", bounds=bounds)
    if not res.success:
        # Try a gentle fallback: jitter init slightly
        theta0_alt = theta0 + 1e-3 * np.random.default_rng(123).standard_normal(len(theta0))
        res = minimize(_negloglik, theta0_alt, args=(y, t, model), method="L-BFGS-B", bounds=bounds)

    nll = _negloglik(res.x, y, t, model)
    ll = -nll
    n = len(y)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    # build mu_t, sigma_t, xi and the time-varying RL curve
    theta = res.x
    if model == "stationary":
        a0, log_sig, xi = theta
        mu_t = np.full_like(y, a0, dtype=float)
        sigma_t = np.exp(log_sig) * np.ones_like(y, dtype=float)
    elif model == "mu":
        a0, a1, log_sig, xi = theta
        mu_t = a0 + a1 * t
        sigma_t = np.exp(log_sig) * np.ones_like(y, dtype=float)
    else:  # 'mu+sigma'
        a0, a1, log_sig, b1, xi = theta
        mu_t = a0 + a1 * t
        sigma_t = np.exp(log_sig + b1 * t)

    rl_t = gev.ppf(p, c=-xi, loc=mu_t, scale=sigma_t)
    selected = (y >= rl_t)

    return GEVFit(model=model, theta=theta, loglik=ll, aic=aic, bic=bic,
                  xi=xi, mu_t=mu_t, sigma_t=sigma_t, rl_t=rl_t, selected_years=selected)


def gev_stationarity_sensitivity(
    df: pd.DataFrame,
    wu_name: str,
    score_cols=("S_wet", "S_dry"),
    year_col="year",
    q=0.917,
    fit_models=("stationary", "mu", "mu+sigma"),
    make_plot=True
) -> Dict[str, Dict[str, Any]]:
    """
    Run GEV stationarity vs. non-stationarity sensitivity for the given WU.

    Parameters
    ----------
    df : DataFrame with columns [year_col, *score_cols]
         (scores are unitless wet/dry scores per year).
    wu_name : str
         Work Unit name (used for titling).
    score_cols : tuple
         Column names for scores, e.g., ('S_wet', 'S_dry'). Any subset is OK.
    year_col : str
         Year column.
    q : float
         Probability for the RL quantile cutoff (e.g., 0.917).
    fit_models : tuple
         Models to fit among {'stationary','mu','mu+sigma'}.
    make_plot : bool
         If True, draws a 3-panel figure per score showing:
         (1) Time series with RL overlays, (2) AIC/BIC bars, (3) Selected-year changes.

    Returns
    -------
    results : dict
        results[score] = {
            'fits' : dict(model -> GEVFit),
            'table': DataFrame with AIC/BIC and delta vs. stationary,
            'years': np.ndarray of years,
            'y'    : np.ndarray of score
        }

    Notes
    -----
    • The “return levels” produced here are **quantile cutoffs on unitless scores**,
      not hydrologic return periods; they are used to flag high-score “extreme” years.
    • Shape ξ is held constant across time within each model for identifiability on short series.
    • Time is centered internally to stabilize optimization.
    """
    out = {}
    df = df.sort_values(year_col).copy()
    years = df[year_col].to_numpy().astype(float)
    ycenter, t0 = _center_time(years)

    for col in score_cols:
        if col not in df.columns:
            continue
        y = df[col].to_numpy().astype(float)

        # Fit models
        fits = {}
        for m in fit_models:
            fits[m] = _fit_model(y, ycenter, q, m)

        # Assemble model comparison table
        base = fits["stationary"]
        rows = []
        for m, f in fits.items():
            rows.append({
                "model": m,
                "loglik": f.loglik,
                "AIC": f.aic,
                "BIC": f.bic,
                "ΔAIC vs stat": f.aic - base.aic,
                "ΔBIC vs stat": f.bic - base.bic
            })
        table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)

        out[col] = {"fits": fits, "table": table, "years": years, "y": y}

        if make_plot:
            # --- Plot per score ---
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"wspace": 0.28})
            ax0, ax1, ax2 = axes

            # (1) Time series + RL overlays + selections
            ax0.plot(years, y, color="#444", lw=1.5, label=f"{col} score")
            colors = {"stationary": "#1f77b4", "mu": "#2ca02c", "mu+sigma": "#d62728"}
            for m, f in fits.items():
                ax0.plot(years, f.rl_t, color=colors[m], lw=1.6, label=f"RL (q={q}, {m})")

            # Highlight selected years under each model
            yy = np.max(y) + 0.02 * (np.max(y) - np.min(y) + 1e-9)
            y_off = yy
            for m, f in fits.items():
                sel_years = years[f.selected_years]
                ax0.scatter(sel_years, np.interp(sel_years, years, y),
                            s=30, color=colors[m], marker="*", zorder=5, label=f"Selected ({m})")

            ax0.set_title(f"{wu_name} — {col}: scores & RL overlays")
            ax0.set_xlabel("Year")
            ax0.set_ylabel("Score (unitless)")
            ax0.legend(fontsize=8, ncols=2, loc="best")

            # (2) AIC/BIC bar comparison
            idx = np.arange(len(fits))
            labels = list(fits.keys())
            aics = [fits[m].aic for m in labels]
            bics = [fits[m].bic for m in labels]
            w = 0.36
            ax1.bar(idx - w/2, aics, width=w, color="#8da0cb", label="AIC")
            ax1.bar(idx + w/2, bics, width=w, color="#fc8d62", label="BIC")
            ax1.set_xticks(idx)
            ax1.set_xticklabels(labels, rotation=0)
            ax1.set_title("Model selection (lower is better)")
            ax1.legend()

            # (3) Selected-year differences (Venn-like counts)
            base_sel = fits["stationary"].selected_years
            ns_sel = {m: f.selected_years for m, f in fits.items() if m != "stationary"}
            lines = []
            for m, sel in ns_sel.items():
                add = np.sum((~base_sel) & sel)
                drop = np.sum(base_sel & (~sel))
                keep = np.sum(base_sel & sel)
                lines.append(f"{m}: keep={keep}, add={add}, drop={drop}")
            ax2.axis("off")
            txt = "\n".join(["Change in selected years (vs stationary):"] + lines)
            ax2.text(0.02, 0.95, txt, va="top", ha="left", family="monospace")

            fig.suptitle(f"GEV stationarity sensitivity — {wu_name} — {col}", y=1.05, fontsize=12)
            outdir = "D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\NonSupplement\\results\\gev_stationarity_sensitivity"
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(f"{outdir}\\{wu_name}_{col}_gev_stationarity_sensitivity.png", bbox_inches="tight", dpi=300)
            plt.show()

    return out

df_wu = pd.read_csv("D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\NonSupplement\\results\\wet_dry_scores.csv")


# Example input per WU (one row per year)
# df_wu columns: ['year', 'wet_score', 'dry_score']
results = gev_stationarity_sensitivity(
    df=df_wu,
    wu_name="05OG000",
    score_cols=("wet_score", "dry_score"),
    year_col="year",
    q=0.917,  # same quantile as in your manuscript
    fit_models=("stationary", "mu", "mu+sigma"),
    make_plot=True
)

# Inspect numeric comparison table for, say, wet:
print(results["S_wet"]["table"])
#   model   loglik     AIC     BIC  ΔAIC vs stat  ΔBIC vs stat
# 0  mu+sigma  ...      ...     ...       ...           ...
# 1  mu        ...      ...     ...       ...           ...
# 2  stationary...

# Extract selected years under non-stationary mu+sigma:
years = results["S_wet"]["years"]
sel_ns = results["S_wet"]["fits"]["mu+sigma"].selected_years
years_selected = years[sel_ns]
print(years_selected)
