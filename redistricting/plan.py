"""A district assignment with cached per-district population, area and perimeter,
the exact incremental update rules for flips and splits, and an independent audit.
"""
import math
from collections import deque
import numpy as np
from .geo import Geo
from .objective import objective


class Plan:
    """Assignment plus cached per-district population, area and perimeter."""

    def __init__(self, g: Geo, assign, lam=2.0, mode="tvd"):
        self.g, self.lam, self.mode = g, lam, mode
        self.a = np.asarray(assign, dtype=int).copy()
        self.members = [set() for _ in range(g.D)]
        for i, d in enumerate(self.a):
            self.members[d].add(i)
        self.pop   = np.zeros(g.D); self.area = np.zeros(g.D)
        self.perim = np.zeros(g.D)
        for d in range(g.D):
            self._recompute(d)

    def _recompute(self, d):
        g, m = self.g, self.members[d]
        self.pop[d]  = g.pop[list(m)].sum()  if m else 0.0
        self.area[d] = g.area[list(m)].sum() if m else 0.0
        p = sum(g.perim[i] for i in m)
        for i in m:
            for j in g.nbr[i]:
                if j in m and j > i:
                    p -= 2 * g.bl(i, j)
        self.perim[d] = p

    # ---------------- metrics ----------------
    def pp(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 4 * math.pi * self.area / (self.perim ** 2)
        return np.where(self.perim > 0, v, 0.0)

    def rel_dev(self):
        return (self.pop - self.g.ideal) / self.g.ideal

    def rmsd(self):
        return float(np.sqrt((self.rel_dev() ** 2).mean()))

    def tvd(self):
        """Total variation distance of population shares from uniform."""
        p = self.pop / self.pop.sum()
        return float(0.5 * np.abs(p - 1.0 / self.g.D).sum())

    def entropy(self):
        p = self.pop / self.pop.sum()
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum() / math.log(self.g.D))

    def penalty(self):
        if self.mode == "tvd":  return self.tvd()
        if self.mode == "rmsd": return self.rmsd()
        return 1.0 - self.entropy()

    def score(self):
        return objective(self.pop, self.area, self.perim, self.g.ideal,
                         self.g.D, self.lam, self.mode)

    def stats_of(self, m):
        """(pop, area, perimeter) of an arbitrary set of units."""
        g = self.g; m = set(m)
        pop = sum(g.pop[i] for i in m); area = sum(g.area[i] for i in m)
        per = sum(g.perim[i] for i in m)
        for i in m:
            for j in g.nbr[i]:
                if j in m and j > i:
                    per -= 2 * g.bl(i, j)
        return pop, area, per

    def score_split(self, d1, d2, A, B):
        """Objective if districts d1,d2 were replaced by the unit sets A,B."""
        pop = self.pop.copy(); area = self.area.copy(); per = self.perim.copy()
        for d, m in ((d1, A), (d2, B)):
            pop[d], area[d], per[d] = self.stats_of(m)
        return objective(pop, area, per, self.g.ideal, self.g.D, self.lam, self.mode)

    def apply_split(self, d1, d2, A, B):
        for d, m in ((d1, A), (d2, B)):
            self.members[d] = set(m)
            for i in m: self.a[i] = d
            self._recompute(d)

    def report(self):
        r = self.rel_dev()
        return dict(score=self.score(), compactness=float(self.pp().mean()),
                    tvd=self.tvd(), rmsd=self.rmsd(), entropy=self.entropy(),
                    max_dev_pct=float(np.abs(r).max() * 100),
                    range_pct=float((self.pop.max() - self.pop.min())
                                    / self.g.ideal * 100),
                    min_pp=float(self.pp().min()))

    # ---------------- flip machinery ----------------
    def connected_without(self, d, drop):
        """Is district d still connected after removing county `drop`?"""
        m = self.members[d] - {drop}
        if len(m) <= 1:
            return True
        start = next(iter(m))
        seen = {start}; q = deque([start])
        while q:
            u = q.popleft()
            for v in self.g.nbr[u]:
                if v in m and v not in seen:
                    seen.add(v); q.append(v)
        return len(seen) == len(m)

    def valid(self, i, d):
        old = self.a[i]
        if d == old:                        return False
        if len(self.members[old]) <= 1:     return False          # no empty district
        if not any(self.a[j] == d for j in self.g.nbr[i]):
            return False                                          # must touch target
        return self.connected_without(old, i)

    def candidates(self):
        seen = set()
        for i in range(self.g.N):
            for j in self.g.nbr[i]:
                d = self.a[j]
                if d != self.a[i] and (i, d) not in seen:
                    seen.add((i, d)); yield i, d

    def delta(self, i, d):
        """Exact change in (pop, area, perim) for the two affected districts."""
        g, old = self.g, self.a[i]
        b_old = sum(g.bl(i, j) for j in g.nbr[i] if self.a[j] == old)
        b_new = sum(g.bl(i, j) for j in g.nbr[i] if self.a[j] == d)
        return (-g.perim[i] + 2 * b_old, g.perim[i] - 2 * b_new), old

    def score_after(self, i, d):
        (dp_old, dp_new), old = self.delta(i, d)
        pop = self.pop.copy(); area = self.area.copy(); per = self.perim.copy()
        pop[old]  -= self.g.pop[i];  pop[d]  += self.g.pop[i]
        area[old] -= self.g.area[i]; area[d] += self.g.area[i]
        per[old]  += dp_old;         per[d]  += dp_new
        return objective(pop, area, per, self.g.ideal, self.g.D,
                         self.lam, self.mode)

    def apply(self, i, d):
        old = self.a[i]
        self.members[old].discard(i); self.members[d].add(i); self.a[i] = d
        self._recompute(old); self._recompute(d)


def verify(p: Plan):
    """Independent audit of the finished plan."""
    g = p.g
    errs = []
    if set(p.a) != set(range(g.D)):
        errs.append("a district is empty")
    for d in range(g.D):                    # contiguity, from scratch
        m = p.members[d]
        s = next(iter(m)); seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in g.nbr[u]:
                if v in m and v not in seen:
                    seen.add(v); q.append(v)
        if len(seen) != len(m):
            errs.append(f"district {d} disconnected")
    if abs(p.pop.sum() - g.pop.sum()) > 1e-6:
        errs.append("population not conserved")
    for d in range(g.D):                    # cached stats vs fresh recompute
        m = list(p.members[d])
        pr = sum(g.perim[i] for i in m) - 2 * sum(
            g.bl(i, j) for i in m for j in g.nbr[i] if j in p.members[d] and j > i)
        if abs(pr - p.perim[d]) > 1e-6:
            errs.append(f"district {d} perimeter cache drift")
    tot = sum(p.perim) + 2 * sum(
        L for (i, j), L in g.border.items() if p.a[i] == p.a[j])
    if abs(tot - g.perim.sum()) > 1e-6:
        errs.append("perimeter decomposition inconsistent")
    return errs
