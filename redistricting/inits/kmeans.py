"""Population-weighted k-means start on county centroids."""
import numpy as np
from .repair import repair


def init_kmeans(g, rng):
    """Population-weighted k-means on county centroids (km-projected)."""
    xy = g.xy
    cent = xy[rng.sample(range(g.N), g.D)].copy()
    a = np.zeros(g.N, dtype=int)
    for _ in range(60):
        d2 = ((xy[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(1)
        for d in range(g.D):
            m = a == d
            if m.any():
                w = g.pop[m]
                cent[d] = (xy[m] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    return repair(g, a, rng)
