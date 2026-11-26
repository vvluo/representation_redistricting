"""Starter maps for environment resets.

Each map: an initialisation strategy (random, pop_greedy, kmeans, spectral, sweep, in
turn), the ReCom population balancer, then a flip repair that greedily takes the
valid flip most reducing sum_d r_d^2 (ties broken at random) until every district is
within tolerance.  Every kept map is contiguous and within tolerance; maps that do
not get there in the step budget are dropped.  Strategies are mixed so resets vary
in shape, and none is optimised for any objective beyond population.

    P = build_pool(F, n=50, seed=0)          # (n, N) int array, plus the strategy names
"""
import random
import numpy as np
from ..inits import make_init
from ..inits.balance import balance

STRATEGIES = ("random", "pop_greedy", "kmeans", "spectral", "sweep")


def pop_repair(F, st, rng, max_steps=None):
    """Greedy flips on population until max |r_d| <= tol.  Returns True on success."""
    max_steps = max_steps or 4 * F.g.N
    for _ in range(max_steps):
        r = st.sums[:, 0] / F.ideal - 1
        if np.abs(r).max() <= F.tol:
            return True
        acts, mask = F.actions(st)
        if not mask.any():
            return False
        i, t = acts[:, 0], acts[:, 1]; s = st.a[i]; pi = F.g.pop[i]
        rs, rt = (st.sums[s, 0] - pi) / F.ideal - 1, (st.sums[t, 0] + pi) / F.ideal - 1
        d = rs ** 2 + rt ** 2 - r[s] ** 2 - r[t] ** 2
        d = np.where(mask, d, np.inf)
        best = d.min()
        if best >= 0:
            return False
        cand = np.flatnonzero(d <= best + 1e-12)
        k = cand[rng.integers(len(cand))]
        st.apply(int(acts[k, 0]), int(acts[k, 1]))
    return bool(np.abs(st.sums[:, 0] / F.ideal - 1).max() <= F.tol)


def build_pool(F, n=50, seed=0, max_tries=None, strategies=STRATEGIES, strict=True, verbose=False):
    """strict=False keeps maps the repair could not bring within tolerance (when the
    units are too coarse, e.g. state houses on tracts); the env's mask still forbids
    any flip that moves an out-of-tolerance district further out."""
    g = F.g; out, names = [], []
    rng = np.random.default_rng(seed); tries = 0
    max_tries = max_tries or 6 * n
    while len(out) < n and tries < max_tries:
        name = strategies[tries % len(strategies)]; tries += 1
        r = random.Random(int(rng.integers(2 ** 31)))
        try:
            a = make_init(name, g, r)
            a, _, _ = balance(g, a, r, tol=F.tol, max_steps=4000, trees_per_step=6)
        except Exception:
            continue
        st = F.state(a)
        if pop_repair(F, st, rng) or not strict:
            out.append(st.a.copy()); names.append(name)
        if verbose:
            print(f"try {tries} {name}: kept {len(out)}/{n}", flush=True)
    if not out:
        raise RuntimeError("no starter map reached the tolerance")
    return np.array(out), names
