"""Independent reference enumerators and small test graphs for validating the ZDD.

`brute_partitions` shares no code with the ZDD: it walks restricted-growth strings
(each set partition exactly once) and keeps those with exactly p connected blocks.
Feasible up to roughly n = 12-14 depending on p.
"""
import math
from collections import deque
import numpy as np


def _connected(block, A):
    block = set(block); s = next(iter(block)); seen = {s}; q = deque([s])
    while q:
        x = q.popleft()
        for y in A[x]:
            if y in block and y not in seen:
                seen.add(y); q.append(y)
    return len(seen) == len(block)


def brute_partitions(n, edges, p):
    """Set of canonical label tuples (first-occurrence labels 0..p-1)."""
    A = [[] for _ in range(n)]
    for u, v in edges:
        A[u].append(v); A[v].append(u)
    out = set()
    lab = [0] * n

    def rec(i, k):
        if n - i < p - k:                       # not enough vertices left to open blocks
            return
        if i == n:
            if k == p and all(_connected([v for v in range(n) if lab[v] == b], A)
                              for b in range(p)):
                out.add(tuple(lab))
            return
        for b in range(min(k + 1, p)):
            lab[i] = b
            rec(i + 1, max(k, b + 1))
    rec(0, 0)
    return out


def canonical(labels):
    """Relabel each row to first-occurrence order (tuple form)."""
    res = []
    for row in np.asarray(labels):
        m = {}
        res.append(tuple(m.setdefault(int(x), len(m)) for x in row))
    return res


# ------------------------------------------------------------- test graphs ----

def grid(r, c, diagonals=False):
    """r x c grid (rook); diagonals=True adds one diagonal per cell (still planar)."""
    idx = lambda i, j: i * c + j
    E = []
    for i in range(r):
        for j in range(c):
            if j + 1 < c: E.append((idx(i, j), idx(i, j + 1)))
            if i + 1 < r: E.append((idx(i, j), idx(i + 1, j)))
            if diagonals and i + 1 < r and j + 1 < c: E.append((idx(i, j), idx(i + 1, j + 1)))
    return r * c, E


def path(n):    return n, [(i, i + 1) for i in range(n - 1)]
def cycle(n):   return n, [(i, (i + 1) % n) for i in range(n)]
def wheel(n):   # hub 0 plus rim 1..n-1
    return n, [(0, i) for i in range(1, n)] + [(i, i % (n - 1) + 1) for i in range(1, n)]
def complete(n):
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def random_tree(n, rng):
    return n, [(i, int(rng.integers(0, i))) for i in range(1, n)]


def delaunay(n, rng):
    """Random planar triangulation of n points in the unit square."""
    from scipy.spatial import Delaunay
    T = Delaunay(rng.random((n, 2)))
    E = set()
    for a, b, c in T.simplices:
        for u, v in ((a, b), (b, c), (a, c)):
            E.add((int(min(u, v)), int(max(u, v))))
    return n, sorted(E)


def stirling2(n, k):
    return sum((-1) ** j * math.comb(k, j) * (k - j) ** n for j in range(k + 1)) // math.factorial(k)


def closed_form(kind, n, p):
    """Known counts of connected p-partitions: path/tree C(n-1,p-1); cycle C(n,p) for
    p >= 2 (choose p cut edges); complete graph S(n,p)."""
    if kind in ("path", "tree"): return math.comb(n - 1, p - 1)
    if kind == "cycle":          return math.comb(n, p) if p >= 2 else 1
    if kind == "complete":       return stirling2(n, p)
    raise ValueError(kind)
