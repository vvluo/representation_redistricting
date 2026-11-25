"""Single-unit flip Metropolis-Hastings, targeting the UNIFORM distribution over plans.

A plan is valid when it has p non-empty contiguous districts, each within `tol` of
the ideal population.  The chain:

  propose  a uniformly random boundary pair (u, d): unit u, district d != a(u)
           that touches u.  B(x) is the set of such pairs, so q(x -> x') = 1/|B(x)|.
  reject   if moving u to d breaks a constraint (population, emptiness,
           contiguity of u's old district; d stays contiguous since u touches it).
  accept   otherwise with probability min(1, |B(x)| / |B(x')|).

The reverse move (u, old district) is always in B(x'): the old district stays
connected with at least two units, so u had a neighbour in it.  Hence the chain is
reversible with respect to the uniform distribution on every valid plan it can
reach.  It is exact only if the valid plans are connected under single flips; when
a tight tolerance disconnects them, run the walk at a looser `walk_tol` and keep the
states within `tol` -- uniform on the larger set restricted to the smaller set is
uniform on the smaller set.
"""
import numpy as np


class FlipChain:
    def __init__(self, g, labels, p, walk_tol, rng):
        self.g, self.p, self.rng = g, p, rng
        self.nbr = [list(v) for v in g.nbr]
        self.pop = g.pop.tolist()
        self.a = [int(x) for x in labels]
        ideal = g.pop.sum() / p
        self.lo, self.hi = ideal * (1 - walk_tol), ideal * (1 + walk_tol)
        self.dpop = [0.0] * p; self.size = [0] * p
        for u, d in enumerate(self.a):
            self.dpop[d] += self.pop[u]; self.size[d] += 1
        if min(self.size) == 0 or not all(self.lo <= x <= self.hi for x in self.dpop):
            raise ValueError("starting plan violates walk_tol or has an empty district")
        self.b = [self._b(u) for u in range(g.N)]
        self.B = sum(self.b)
        self.bmax = max(min(len(v), p - 1) for v in self.nbr)
        self.steps = self.accepted = self.invalid = 0

    def _b(self, u):
        au = self.a[u]
        return len({self.a[w] for w in self.nbr[u]} - {au})

    def _old_stays_connected(self, u):
        old = self.a[u]
        if self.size[old] == 1:
            return False
        start = next((w for w in self.nbr[u] if self.a[w] == old), None)
        if start is None:
            return False
        a, nbr = self.a, self.nbr
        seen = {start, u}; st = [start]; reached = 1
        while st:
            x = st.pop()
            for y in nbr[x]:
                if y not in seen and a[y] == old:
                    seen.add(y); st.append(y); reached += 1
        return reached == self.size[old] - 1

    def step(self):
        self.steps += 1
        rng = self.rng; n = self.g.N
        while True:                                   # u with prob b(u)/B by rejection
            u = int(rng.integers(n))
            if self.b[u] and rng.random() * self.bmax < self.b[u]:
                break
        au = self.a[u]
        choices = sorted({self.a[w] for w in self.nbr[u]} - {au})
        d = choices[int(rng.integers(len(choices)))]
        pu = self.pop[u]
        if (self.dpop[au] - pu < self.lo or self.dpop[d] + pu > self.hi
                or not self._old_stays_connected(u)):
            self.invalid += 1
            return False
        touched = [u] + self.nbr[u]
        old_b = [self.b[w] for w in touched]
        self.a[u] = d
        new_b = [self._b(w) for w in touched]
        B_new = self.B - sum(old_b) + sum(new_b)
        if rng.random() * B_new < self.B:              # accept w.p. min(1, B/B')
            for w, v in zip(touched, new_b):
                self.b[w] = v
            self.B = B_new
            self.dpop[au] -= pu; self.dpop[d] += pu
            self.size[au] -= 1; self.size[d] += 1
            self.accepted += 1
            return True
        self.a[u] = au
        return False

    def run(self, steps, thin=1, keep_tol=None):
        """Run `steps` steps; return (k, N) int8 labels of every `thin`-th state
        (only those within keep_tol of ideal, if given)."""
        out = []; ideal = self.g.pop.sum() / self.p
        for t in range(1, steps + 1):
            self.step()
            if t % thin == 0:
                if keep_tol is None or max(abs(x / ideal - 1) for x in self.dpop) <= keep_tol:
                    out.append(list(self.a))
        return np.array(out, dtype=np.int8).reshape(-1, self.g.N)
