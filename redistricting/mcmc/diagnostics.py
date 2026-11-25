"""Convergence diagnostics and two-sample comparisons for scalar plan statistics."""
import numpy as np
from scipy import stats as sps


def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time (Sokal's automatic window)."""
    x = np.asarray(x, float); n = len(x)
    if n < 4 or x.std() == 0:
        return 1.0
    f = np.fft.rfft(x - x.mean(), 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]
    tau = 2 * np.cumsum(acf) - 1
    for m in range(1, n):
        if m >= c * tau[m]:
            return max(1.0, float(tau[m]))
    return max(1.0, float(tau[-1]))


def ess(chains):
    """Effective sample size summed over chains (list of 1-D arrays)."""
    return float(sum(len(x) / autocorr_time(x) for x in chains))


def rhat(chains):
    """Split-R-hat (Gelman et al.); ~1.00 when chains agree."""
    halves = []
    for x in chains:
        x = np.asarray(x, float); h = len(x) // 2
        halves += [x[:h], x[h:2 * h]]
    m = len(halves); n = min(len(h) for h in halves)
    X = np.array([h[:n] for h in halves])
    W = X.var(axis=1, ddof=1).mean()
    if W == 0:
        return 1.0
    B = n * X.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def compare(ref, x, ess_x=None, qs=(0.05, 0.25, 0.5, 0.75, 0.95), tails=(0.01, 0.99)):
    """Compare sample x against reference ref (both 1-D).

    KS p-values assume independent draws, which MCMC output is not; with ess_x the
    KS test is re-run on x thinned to its effective size (`ks_p_ess`).  Tail
    probabilities are P(x <= ref's 1% quantile) and P(x >= ref's 99% quantile).
    """
    ref = np.asarray(ref, float); x = np.asarray(x, float)
    out = {"mean_ref": ref.mean(), "mean_x": x.mean(), "dmean": x.mean() - ref.mean(),
           "dmedian": np.median(x) - np.median(ref)}
    for q in qs:
        out[f"dq{int(q*100)}"] = np.quantile(x, q) - np.quantile(ref, q)
    lo, hi = np.quantile(ref, tails[0]), np.quantile(ref, tails[1])
    out["tail_lo_ref"] = (ref <= lo).mean(); out["tail_lo_x"] = (x <= lo).mean()
    out["tail_hi_ref"] = (ref >= hi).mean(); out["tail_hi_x"] = (x >= hi).mean()
    ks = sps.ks_2samp(ref, x)
    out["ks"] = ks.statistic; out["ks_p_naive"] = ks.pvalue
    if ess_x is not None:
        k = max(2, int(ess_x)); idx = np.linspace(0, len(x) - 1, k).astype(int)
        out["ks_p_ess"] = sps.ks_2samp(ref, x[idx]).pvalue
        out["ess"] = ess_x
    return out
