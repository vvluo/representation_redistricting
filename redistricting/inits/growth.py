"""Contiguous-by-construction starts: multi-source region growth."""
import numpy as np
from ..geo import Geo


def seeded_growth(g: Geo, rng, pop_aware=False):
    """Multi-source random growth -> guaranteed contiguous seed plan."""
    seeds = rng.sample(range(g.N), g.D)
    a = np.full(g.N, -1, dtype=int)
    frontier = {d: set() for d in range(g.D)}
    popd = np.zeros(g.D)
    for d, s in enumerate(seeds):
        a[s] = d; popd[d] = g.pop[s]
        frontier[d] = {j for j in g.nbr[s]}
    left = g.N - g.D
    while left:
        order = (sorted(range(g.D), key=lambda d: popd[d]) if pop_aware
                 else rng.sample(range(g.D), g.D))
        moved = False
        for d in order:
            opts = [j for j in frontier[d] if a[j] == -1]
            if not opts:
                continue
            j = min(opts, key=lambda x: g.pop[x]) if pop_aware else rng.choice(opts)
            a[j] = d; popd[d] += g.pop[j]; left -= 1; moved = True
            frontier[d].update(g.nbr[j])
            break
        if not moved:                       # stranded: attach to any neighbour
            for j in range(g.N):
                if a[j] == -1:
                    nb = [a[k] for k in g.nbr[j] if a[k] >= 0]
                    if nb:
                        a[j] = nb[0]; left -= 1; moved = True; break
            if not moved:
                raise RuntimeError("cannot complete contiguous seed plan")
    return a


def init_random(g, rng):
    """Grow D regions from random seeds, one random county at a time."""
    return seeded_growth(g, rng, pop_aware=False)


def init_pop_greedy(g, rng):
    """Grow D regions from random seeds, always extending the least-populous district."""
    return seeded_growth(g, rng, pop_aware=True)
