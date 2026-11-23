"""Spectral start: recursive bisection on the border-length-weighted Laplacian."""
import numpy as np
from .repair import repair


def init_spectral(g, rng):
    """Recursive spectral bisection on the border-length-weighted Laplacian."""
    W = np.zeros((g.N, g.N))
    for (i, j), L in g.border.items():
        w = L * (1 + 0.05 * rng.uniform(-1, 1))
        W[i, j] = W[j, i] = w
    a = np.full(g.N, -1, dtype=int)
    nxt = [0]

    def bisect(S, k):
        if k == 1 or len(S) <= 1:
            for i in S:
                a[i] = nxt[0]
            nxt[0] += 1
            return
        S = sorted(S); idx = {n: t for t, n in enumerate(S)}
        sub = W[np.ix_(S, S)]
        L = np.diag(sub.sum(1)) - sub
        val, vec = np.linalg.eigh(L)
        f = vec[:, 1] if len(S) > 1 else np.zeros(len(S))
        order = sorted(S, key=lambda n: f[idx[n]])
        k1 = k // 2
        want = g.pop[order].sum() * k1 / k
        run, cut = 0.0, 0
        for t, n in enumerate(order):
            run += g.pop[n]
            if run >= want:
                cut = t + 1
                break
        cut = max(1, min(len(order) - 1, cut))
        bisect(order[:cut], k1); bisect(order[cut:], k - k1)

    bisect(list(range(g.N)), g.D)
    return repair(g, a, rng)
