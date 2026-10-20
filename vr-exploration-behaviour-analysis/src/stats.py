"""Paired comparisons with assumption checks, effect sizes, bootstrap CIs and Holm correction."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05
N_BOOT = 10_000


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014): (R+ - R-) / (R+ + R-), zeros dropped."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cohens_dz(diff: np.ndarray) -> float:
    """Standardised mean of paired differences."""
    return float(diff.mean() / diff.std(ddof=1))


def bootstrap_ci(diff: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of fn(diff) resampling pairs with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    vals = np.array([fn(diff[i]) for i in idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare paired samples b vs a (difference b - a).

    Shapiro-Wilk on the differences decides the test: paired t-test (effect size Cohen's dz) if
    p >= ALPHA, else Wilcoxon signed-rank (effect size matched-pairs rank-biserial r).
    """
    diff = np.asarray(b, float) - np.asarray(a, float)
    shapiro_p = stats.shapiro(diff).pvalue
    if shapiro_p >= ALPHA:
        test, res, es_name, es_fn = "paired t", stats.ttest_rel(b, a), "dz", cohens_dz
        centre_name, centre_fn = "mean_diff", np.mean
    else:
        test, res, es_name, es_fn = "Wilcoxon", stats.wilcoxon(b, a), "r_rb", rank_biserial
        centre_name, centre_fn = "median_diff", np.median
    c_lo, c_hi = bootstrap_ci(diff, centre_fn)
    e_lo, e_hi = bootstrap_ci(diff, es_fn, seed=1)
    return {
        "n": len(diff), "shapiro_p": shapiro_p, "test": test, "statistic": float(res.statistic),
        "p": float(res.pvalue), "centre": centre_name, "diff": float(centre_fn(diff)),
        "diff_ci_lo": c_lo, "diff_ci_hi": c_hi, "effect": es_name, "effect_size": es_fn(diff),
        "es_ci_lo": e_lo, "es_ci_hi": e_hi,
    }


def friedman(wide: pd.DataFrame) -> dict:
    """Friedman test across columns (conditions) of a complete subject x condition table,
    with Kendall's W = chi2 / (n (k - 1))."""
    res = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    n, k = wide.shape
    return {"n": n, "k": k, "chi2": float(res.statistic), "p": float(res.pvalue),
            "kendall_w": float(res.statistic / (n * (k - 1)))}


def holm(results: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    """Add Holm-adjusted p-values and a significance flag across the rows of one test family."""
    reject, p_adj, _, _ = multipletests(results[p_col], alpha=ALPHA, method="holm")
    return results.assign(p_holm=p_adj, significant=reject)
