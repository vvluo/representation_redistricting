"""Incremental districting state for single-tract flips.

Keeps, for an assignment a (unit -> district 0..p-1):
  sums[d, k]   district sums of every count column the features need
  perim[d]     district perimeter = sum of unit perimeters - 2 * internal borders
  size[d]      units per district
  artic[d]     articulation units of district d's induced subgraph: moving one out
               would disconnect d (cached, recomputed only for the two districts a
               flip touches)
  ccount/cpop  for every protected-group cluster, its units / population per district

apply(i, t) moves unit i to district t in O(deg(i) + |district|) time.
"""
import numpy as np


def articulation(nodes, nbr, a, d):
    """Articulation points of the subgraph induced by `nodes` (all with a[u] == d).
    Iterative Tarjan; returns a set."""
    if len(nodes) < 3:
        return set()
    disc, low, out = {}, {}, set()
    t = 0
    for root in nodes:
        if root in disc:
            continue
        disc[root] = low[root] = t; t += 1
        stack = [(root, -1, iter(nbr[root]))]; children = 0
        while stack:
            u, parent, it = stack[-1]
            advanced = False
            for v in it:
                if a[v] != d:
                    continue
                if v not in disc:
                    disc[v] = low[v] = t; t += 1
                    stack.append((v, u, iter(nbr[v])))
                    if u == root:
                        children += 1
                    advanced = True
                    break
                elif v != parent:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    w = stack[-1][0]
                    low[w] = min(low[w], low[u])
                    if w != root and low[u] >= disc[w]:
                        out.add(w)
        if children > 1:
            out.add(root)
    return out


class DistrictState:
    def __init__(self, g, a, p, C, clusters):
        """g: Geo;  a: assignment;  C: (N, K) count matrix;  clusters: list of int arrays
        (unit -> cluster id or -1), one per protected group, ids global across groups."""
        self.g, self.p = g, p
        self.a = np.asarray(a, dtype=np.int64).copy()
        self.C = C
        self.nbr = [list(v) for v in g.nbr]
        self.sums = np.zeros((p, C.shape[1]))
        np.add.at(self.sums, self.a, C)
        self.size = np.bincount(self.a, minlength=p)
        if self.size.min() == 0:
            raise ValueError("every district needs at least one unit")
        self.perim = np.bincount(self.a, g.perim, p)
        for (i, j), L in g.border.items():
            if self.a[i] == self.a[j]:
                self.perim[self.a[i]] -= 2 * L
        self.clusters = clusters
        self.ncl = 1 + max((int(c.max()) for c in clusters if c.size and c.max() >= 0), default=-1)
        self.ccount = np.zeros((max(self.ncl, 1), p)); self.cpop = np.zeros((max(self.ncl, 1), p))
        self.unit_cl = [np.array([c[u] for c in clusters if c[u] >= 0], dtype=np.int64)
                        for u in range(g.N)]
        for u in range(g.N):
            for c in self.unit_cl[u]:
                self.ccount[c, self.a[u]] += 1; self.cpop[c, self.a[u]] += g.pop[u]
        self.members = [set(np.flatnonzero(self.a == d).tolist()) for d in range(p)]
        self.artic = [articulation(list(self.members[d]), self.nbr, self.a, d) for d in range(p)]

    def shared(self, i, d):
        return sum(self.g.bl(i, j) for j in self.nbr[i] if self.a[j] == d)

    def apply(self, i, t):
        s = int(self.a[i]); t = int(t)
        if s == t:
            return
        Ls, Lt = self.shared(i, s), self.shared(i, t)
        self.perim[s] -= self.g.perim[i] - 2 * Ls
        self.perim[t] += self.g.perim[i] - 2 * Lt
        self.sums[s] -= self.C[i]; self.sums[t] += self.C[i]
        self.size[s] -= 1; self.size[t] += 1
        for c in self.unit_cl[i]:
            self.ccount[c, s] -= 1; self.ccount[c, t] += 1
            self.cpop[c, s] -= self.g.pop[i]; self.cpop[c, t] += self.g.pop[i]
        self.a[i] = t
        self.members[s].discard(i); self.members[t].add(i)
        for d in (s, t):
            self.artic[d] = articulation(list(self.members[d]), self.nbr, self.a, d)
