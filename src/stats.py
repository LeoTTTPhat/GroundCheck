"""Statistical inference for GroundCheck (numpy/scipy only, no statsmodels).

Everything the audit claims about a difference between two conditions is a
PAIRED comparison on the same items, so the default test is an exact McNemar
(binomial on the discordant pairs).  Rates get Wilson score intervals; derived
quantities (gaps, pooled rates over heterogeneous strata) get item-level
bootstrap intervals.  Family-level test batteries are Holm-corrected.

All functions are deterministic: the bootstrap uses an explicit seed.
"""
import numpy as np
from scipy import stats as sps

BOOT = 10000
SEED = 12345


# ------------------------------------------------------------- intervals --

def wilson(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = (c - h) / d, (c + h) / d
    return (p, float(max(0.0, lo)), float(min(1.0, hi)))


def wilson_from_flags(flags, alpha=0.05):
    f = [x for x in flags if x is not None]
    return wilson(int(sum(f)), len(f), alpha)


def boot_ci(values, stat=np.mean, n_boot=BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI of `stat` over a 1-D sample."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = stat(v[idx], axis=1)
    return (float(stat(v)), float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


def boot_diff_ci(a, b, paired=True, n_boot=BOOT, alpha=0.05, seed=SEED):
    """Bootstrap CI for mean(a) - mean(b).

    paired=True resamples ITEMS (a[i], b[i] together), which is the right
    thing for condition-vs-condition comparisons on a fixed item set.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    if paired:
        assert len(a) == len(b), "paired bootstrap needs equal lengths"
        if len(a) == 0:
            return (float("nan"), float("nan"), float("nan"))
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        bs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        return (float(a.mean() - b.mean()),
                float(np.percentile(bs, 100 * alpha / 2)),
                float(np.percentile(bs, 100 * (1 - alpha / 2))))
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"), float("nan"))
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    bs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


# ----------------------------------------------------------------- tests --

def mcnemar(a, b):
    """Exact McNemar test on paired binary outcomes a[i], b[i] in {0,1}.

    Returns dict(n, n01, n10, p) where n10 = #(a=1,b=0) (a better) and
    n01 = #(a=0,b=1).  Two-sided exact binomial on the discordant pairs.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    assert len(a) == len(b)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    d = n10 + n01
    if d == 0:
        p = 1.0
    else:
        p = float(sps.binomtest(n10, d, 0.5, alternative="two-sided").pvalue)
    return {"n": int(len(a)), "n10": n10, "n01": n01, "p": p}


def tost_equivalence(a, b, margin=0.05, alpha=0.05):
    """Two one-sided tests for EQUIVALENCE of two paired binary rates.

    Establishes 'no meaningful difference' rather than merely failing to
    reject.  Equivalent iff the (1-2*alpha) CI of the paired difference lies
    inside +-margin.  Returns dict(diff, lo, hi, margin, equivalent).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d, lo, hi = boot_diff_ci(a, b, paired=True, alpha=2 * alpha)
    return {"diff": d, "lo": lo, "hi": hi, "margin": margin,
            "equivalent": bool(lo > -margin and hi < margin)}


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns (adjusted p array, reject array)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p, np.zeros(0, dtype=bool)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj, adj <= alpha


def prop_test_unpaired(k1, n1, k2, n2):
    """Fisher exact test for two independent proportions (small-n safe)."""
    if min(n1, n2) == 0:
        return 1.0
    tbl = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(sps.fisher_exact(tbl, alternative="two-sided")[1])


def spearman_trend(x, y):
    """Monotone trend test (e.g. accuracy vs log model size)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(len(x))}
    r = sps.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


# ------------------------------------------------------------ degeneracy --

def response_diversity(preds):
    """How many distinct answers does a condition actually produce?

    Returns dict(distinct, n, distinct_rate, norm_entropy, modal_share).
    A model emitting one constant answer scores distinct_rate ~ 1/n,
    norm_entropy 0 and modal_share 1 -- which distinguishes 'answers, but not
    from the signal' from 'does not answer at all'.
    """
    vals = [p for p in preds if p is not None]
    n = len(vals)
    if n == 0:
        return {"distinct": 0, "n": 0, "distinct_rate": float("nan"),
                "norm_entropy": float("nan"), "modal_share": float("nan")}
    # bucket floats at 2 decimals so 1.9000001 and 1.9 are one answer
    keys = [round(float(v), 2) if not isinstance(v, str) else v for v in vals]
    uniq, counts = np.unique(np.array([str(k) for k in keys]),
                             return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    max_ent = float(np.log(len(uniq))) if len(uniq) > 1 else 0.0
    return {"distinct": int(len(uniq)), "n": int(n),
            "distinct_rate": float(len(uniq) / n),
            "norm_entropy": float(ent / max_ent) if max_ent > 0 else 0.0,
            "modal_share": float(counts.max() / counts.sum())}


def fmt_ci(triple, dec=3):
    """'0.196 [0.161, 0.237]'"""
    p, lo, hi = triple
    if p != p:  # nan
        return "--"
    return f"{p:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"


def fmt_pm(triple, dec=3):
    """'0.196 (+-0.038)' -- half-width form, for space-tight tables."""
    p, lo, hi = triple
    if p != p:
        return "--"
    return f"{p:.{dec}f}$\\pm${(hi - lo) / 2:.{dec}f}"
