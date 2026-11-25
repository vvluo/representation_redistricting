"""Plain ReCom as a sampler: a NEGATIVE CONTROL for the benchmarks.

Each step merges a uniformly chosen pair of adjacent districts, draws a uniform
spanning tree of the union (Wilson), and splits it at a uniformly chosen tree edge
whose halves both meet the population tolerance (staying put if none does).  Every
proposal is accepted, so the chain does not target the uniform distribution on
plans: its stationary law favours districts with many spanning trees, i.e. compact
ones (DeFord, Duchin & Solomon 2021).  A benchmark that cannot tell this chain from
the uniform ZDD sample has no power.
"""
import random
from collections import defaultdict
import numpy as np
from ..search.recom import induced, tree_ust


class ReComChain:
    def __init__(self, g, labels, p, walk_tol, seed):
        self.g, self.p = g, p
        self.rng = random.Random(seed)
        self.a = [int(x) for x in labels]
        ideal = g.pop.sum() / p
        self.lo, self.hi = ideal * (1 - walk_tol), ideal * (1 + walk_tol)
        self.steps = self.accepted = 0

    def _pairs(self):
        s = set()
        for (i, j) in self.g.border:
            x, y = self.a[i], self.a[j]
            if x != y:
                s.add((min(x, y), max(x, y)))
        return sorted(s)

    def step(self):
        self.steps += 1
        d1, d2 = self.rng.choice(self._pairs())
        nodes = [u for u, d in enumerate(self.a) if d in (d1, d2)]
        parent = tree_ust(self.g, induced(self.g, nodes), self.rng)
        children = defaultdict(list); root = None
        for v, q in parent.items():
            if q is None: root = v
            else: children[q].append(v)
        order = []; st = [root]
        while st:
            u = st.pop(); order.append(u); st.extend(children[u])
        sub = {}
        for u in reversed(order):
            sub[u] = self.g.pop[u] + sum(sub[c] for c in children[u])
        tot = sub[root]
        ok = [v for v in order if v != root
              and self.lo <= sub[v] <= self.hi and self.lo <= tot - sub[v] <= self.hi]
        if not ok:
            return False
        v = self.rng.choice(ok)
        A = set(); st = [v]
        while st:
            u = st.pop(); A.add(u); st.extend(children[u])
        for u in nodes:
            self.a[u] = d1 if u in A else d2
        self.accepted += 1
        return True

    def run(self, steps, thin=1, keep_tol=None):
        out = []; ideal = self.g.pop.sum() / self.p; P = self.g.pop
        for t in range(1, steps + 1):
            self.step()
            if t % thin == 0:
                a = np.array(self.a)
                if keep_tol is None or np.abs(np.bincount(a, P, self.p) / ideal - 1).max() <= keep_tol:
                    out.append(list(self.a))
        return np.array(out, dtype=np.int8).reshape(-1, self.g.N)
