"""Sweep start: fill one district at a time along a random direction."""
import math
import numpy as np
from .repair import repair


def init_sweep(g, rng):
    """Grow one district at a time along a random sweep direction."""
    xy = g.xy
    th = rng.uniform(0, math.pi)
    proj = xy[:, 0] * math.cos(th) + xy[:, 1] * math.sin(th)
    a = np.full(g.N, -1, dtype=int)
    for d in range(g.D):
        un = [i for i in range(g.N) if a[i] < 0]
        if not un:
            break
        start = min(un, key=lambda i: proj[i])
        a[start] = d; tot = g.pop[start]
        front = {j for j in g.nbr[start] if a[j] < 0}
        target = g.ideal if d < g.D - 1 else float("inf")
        while tot < target and front:
            j = min(front, key=lambda x: proj[x])
            front.discard(j)
            if a[j] >= 0:
                continue
            a[j] = d; tot += g.pop[j]
            front |= {k for k in g.nbr[j] if a[k] < 0}
    for i in range(g.N):                         # anything the sweep missed
        if a[i] < 0:
            a[i] = -1
    return repair(g, a, rng)
