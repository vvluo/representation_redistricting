"""Zero-suppressed decision diagram of all p-graph partitions (frontier-based search).

Follows Fifield, Imai, Kawahara & Kenny (2020), sec. 2.2-2.4, after Kawahara et al.
(2017).  The ZDD has one variable per edge of the unit graph; a root-to-1 path
selects the set of *retained* edges, and the partition is the connected components
of the retained edges.  The induced-subgraph condition (every edge inside a district
is retained, every edge between districts is cut) makes the map from paths to
partitions one-to-one.

A node at level l stores, for the vertices on frontier F_{l-1}:
    comp  canonical component labels (relabelled in frontier order, so that two
          paths with the same frontier connectivity merge -- sec. 2.3.3)
    forb  forbidden pairs: components joined by a cut edge, never to be merged
    dcc   number of determined (closed) components
No population is stored.  As in the paper, storing component populations stops
nodes from merging; population limits are applied afterwards to enumerated or
sampled plans (`population_ok`).  Uniform samples filtered by a population limit
are exactly uniform on the plans meeting that limit.

The diagram is built level by level ("quasi-reduced": every path visits every
level), which is what counting, uniform sampling and unranking need.  `zdd_size`
reports the node count after the ZDD reduction rules as well.
"""
import time
import numpy as np
from .ordering import best_order, edge_order, frontiers


class BuildAborted(RuntimeError):
    pass


class PartitionZDD:
    """All partitions of a connected graph into exactly p connected parts.

    PartitionZDD(n, edges, p, order="best", xy=None, backend="python")
        backend: "python", or "cpp" (cpp/zdd_build.cpp; same diagram, node for node)
        order: "best" (smallest max frontier among cut / BFS / coordinate sweeps;
               sweeps need planar coordinates xy), "cut", "bfs", or an explicit list
    .count            exact number of partitions (Python int)
    .sample(M, rng)   (M, n) int8 labels, iid uniform over all partitions
    .unrank(idx)      (len(idx), n) labels of partitions number idx (0 <= idx < count)
    .all_labels()     every partition (only when count is small)
    """

    def __init__(self, n, edges, p, order="best", xy=None, max_nodes=None, max_seconds=None,
                 verbose=False, backend="python"):
        self.n, self.p = n, p
        t0 = time.time()
        if isinstance(order, str):
            if order == "best":
                self.order, self.order_name, _, self.order_tried = best_order(n, edges, xy=xy)
            else:
                self.order = edge_order(n, edges, order); self.order_name = order
                self.order_tried = {}
        else:
            self.order = [(min(u, v), max(u, v)) for u, v in order]; self.order_name = "given"
            self.order_tried = {}
        if sorted(self.order) != sorted((min(u, v), max(u, v)) for u, v in edges):
            raise ValueError("order must be a permutation of the edges")
        self.m = len(self.order)
        self.frontier_sizes = [len(f) for f in frontiers(n, self.order)]
        self.max_frontier = max(self.frontier_sizes, default=0)
        self.t_order = time.time() - t0
        t1 = time.time()
        self.backend = backend
        if backend == "cpp":
            self._build_cpp(max_nodes, max_seconds)
        elif backend == "python":
            self._build(max_nodes, max_seconds, verbose)
        else:
            raise ValueError(f"backend must be 'python' or 'cpp', not {backend!r}")
        self.t_build = time.time() - t1
        t2 = time.time()
        self._count()
        self.t_count = time.time() - t2

    # ------------------------------------------------------------- construction --
    def _build_cpp(self, max_nodes, max_seconds):
        """Same diagram as _build, built by cpp/zdd_build.cpp (identical node numbering)."""
        from . import _cpp
        if self.n == 1:
            raise ValueError("need at least one edge")
        lo, hi, width, nodes, st, nl = _cpp.build(self.n, self.order, self.p, max_nodes, max_seconds)
        if st in (1, 2):
            raise BuildAborted(f"{_cpp.STATUS[st]} exceeded at level {nl}/{self.m}")
        if st == -2:
            raise ValueError("every vertex needs an edge (graph must be connected)")
        if st != 0:
            raise ValueError(_cpp.STATUS.get(st, f"C++ builder status {st}"))
        self.lo, self.hi, self.width, self.nodes = lo, hi, width, nodes

    def _build(self, max_nodes, max_seconds, verbose):
        n, p, order, m = self.n, self.p, self.order, self.m
        if n == 1:
            raise ValueError("need at least one edge")
        first, last = {}, {}
        for l, (u, v) in enumerate(order):
            for w in (u, v):
                first.setdefault(w, l); last[w] = l
        if len(first) != n:
            raise ValueError("every vertex needs an edge (graph must be connected)")

        self.lo, self.hi = [], []                  # per level: child codes, 0=⊥ 1=⊤ k>=2 -> node k-2 of next level
        self.width = []
        Fprev = []
        entered = 0
        cur = {((), frozenset(), 0): 0}
        total = 0; t0 = time.time()
        for l, (u, v) in enumerate(order):
            enter = [w for w in (u, v) if first[w] == l]
            leave = [w for w in (u, v) if last[w] == l]
            entered += len(enter)
            unentered = n - entered
            Fext = Fprev + enter
            pos = {w: i for i, w in enumerate(Fext)}
            iu, iv = pos[u], pos[v]
            lset = set(leave)
            keep_idx = [i for i, w in enumerate(Fext) if w not in lset]
            leave_idx = [pos[w] for w in leave]
            is_last = (l == m - 1)
            nxt = {}
            nlev = len(cur)
            lo = [0] * nlev; hi = [0] * nlev
            ne = len(enter)

            for (comp, forb, dcc), sid in cur.items():
                k0 = (max(comp) + 1) if comp else 0
                lab0 = list(comp) + list(range(k0, k0 + ne))
                a, b = lab0[iu], lab0[iv]
                for take in (0, 1):
                    lab = lab0; fb = forb; d = dcc
                    if take:
                        if a != b:
                            x, y = (a, b) if a < b else (b, a)
                            if (x, y) in fb:
                                continue                     # child stays ⊥ (0)
                            lab = [x if c == y else c for c in lab0]
                            if fb:
                                fb = frozenset((min(x if c1 == y else c1, x if c2 == y else c2),
                                                max(x if c1 == y else c1, x if c2 == y else c2))
                                               for c1, c2 in fb)
                    else:
                        if a == b:
                            continue
                        fb = fb | {(a, b) if a < b else (b, a)}
                    newlab = [lab[i] for i in keep_idx]
                    if leave_idx:
                        alive = set(newlab); closed = set()
                        for i in leave_idx:
                            c = lab[i]
                            if c not in alive and c not in closed:
                                closed.add(c)
                        if closed:
                            d += len(closed)
                            if d > p:
                                continue
                            fb = frozenset(q for q in fb if q[0] not in closed and q[1] not in closed)
                    if is_last:
                        code = 1 if d == p else 0
                    else:
                        if d == p:                              # vertices remain -> too many parts
                            continue
                        if d + len(set(newlab)) + unentered < p:  # cannot reach p parts
                            continue
                        rel = {}
                        canon = tuple(rel.setdefault(c, len(rel)) for c in newlab)
                        cf = frozenset((min(rel[c1], rel[c2]), max(rel[c1], rel[c2]))
                                       for c1, c2 in fb) if fb else frozenset()
                        key = (canon, cf, d)
                        j = nxt.get(key)
                        if j is None:
                            j = nxt[key] = len(nxt)
                        code = j + 2
                    if take: hi[sid] = code
                    else:    lo[sid] = code
            self.lo.append(np.array(lo, dtype=np.int32))      # widths stay far below 2^31
            self.hi.append(np.array(hi, dtype=np.int32))
            self.width.append(nlev)
            total += nlev
            if verbose and (l % 20 == 0 or is_last):
                print(f"  level {l+1}/{m}  |F|={len(Fext)}  nodes {nlev:,}  total {total:,}"
                      f"  {time.time()-t0:.1f}s", flush=True)
            if max_nodes and total + len(nxt) > max_nodes:
                raise BuildAborted(f"node budget {max_nodes:,} exceeded at level {l+1}/{m}")
            if max_seconds and time.time() - t0 > max_seconds:
                raise BuildAborted(f"time budget {max_seconds}s exceeded at level {l+1}/{m}")
            Fprev = [Fext[i] for i in keep_idx]
            cur = nxt
        self.nodes = total

    # ---------------------------------------------------------------- counting --
    def _count(self):
        """Exact path counts, streamed one level at a time (only two levels of Python
        ints are alive at once); keeps per-node float probabilities for sampling."""
        m = self.m
        self.phi = [None] * m             # probability of taking the 1-arc
        below = None
        for l in range(m - 1, -1, -1):
            def val(codes):
                out = np.zeros(len(codes), dtype=object)
                out[codes == 1] = 1
                k = codes >= 2
                if k.any():
                    out[k] = below[codes[k] - 2]
                return out
            c1 = val(self.hi[l])
            c = val(self.lo[l]) + c1
            self.phi[l] = np.array([float(x) / float(y) if y else 0.0
                                    for x, y in zip(c1, c)], dtype=np.float64)
            below = c
        self.count = int(below[0])

    def _int64_counts(self):
        """Exact per-node counts as int64 (valid because every node count is at most
        the root count, which unranking requires to be < 2**62)."""
        if self.count >= 2**62:
            raise ValueError("count too large for int64 unranking; use sample()")
        cnt = [None] * self.m; below = None
        for l in range(self.m - 1, -1, -1):
            def val(codes):
                out = (codes == 1).astype(np.int64)
                k = codes >= 2
                if k.any():
                    out[k] = below[codes[k] - 2]
                return out
            c0 = val(self.lo[l]); cnt[l] = c0
            below = c0 + val(self.hi[l])
        return cnt                        # cnt[l] = number of paths through the 0-arc

    # ----------------------------------------------------------------- reading --
    def _labels_from_bits(self, bits):
        """(M, m) retained-edge bits -> (M, n) canonical labels 0..p-1 (first-occurrence)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        M, n, p = bits.shape[0], self.n, self.p
        E = np.array(self.order, dtype=np.int64)
        r, e = np.nonzero(bits)
        rows = r * n + E[e, 0]; cols = r * n + E[e, 1]
        A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(M * n, M * n))
        k, lab = connected_components(A, directed=False)
        if k != M * p:
            raise AssertionError(f"expected {M*p} components, got {k}")
        lab = lab.reshape(M, n) - (np.arange(M) * p)[:, None]
        if lab.min() < 0 or lab.max() >= p:
            raise AssertionError("component labels not contiguous per sample")
        return lab.astype(np.int8)

    def sample_bits(self, M, rng):
        node = np.zeros(M, dtype=np.int64)
        bits = np.zeros((M, self.m), dtype=bool)
        for l in range(self.m):
            take = rng.random(M) < self.phi[l][node]
            bits[:, l] = take
            code = np.where(take, self.hi[l][node], self.lo[l][node])
            if (code == 0).any():
                raise AssertionError("sampler reached the 0-terminal")
            node = code - 2
        if not (code == 1).all():
            raise AssertionError("sampler did not reach the 1-terminal")
        return bits

    def sample(self, M, rng, batch=200_000):
        """iid uniform partitions as (M, n) labels."""
        if self.count == 0:
            raise ValueError("no partitions")
        out = []
        for s in range(0, M, batch):
            out.append(self._labels_from_bits(self.sample_bits(min(batch, M - s), rng)))
        return np.concatenate(out)

    def unrank(self, idx):
        """Partitions number idx (array of ints in [0, count)), in a fixed order."""
        idx = np.asarray(idx, dtype=np.int64).copy()
        if len(idx) and (idx.min() < 0 or idx.max() >= self.count):
            raise IndexError("rank out of range")
        if not hasattr(self, "_c0"):
            self._c0 = self._int64_counts()
        M = len(idx); node = np.zeros(M, dtype=np.int64)
        bits = np.zeros((M, self.m), dtype=bool)
        for l in range(self.m):
            c0 = self._c0[l][node]
            take = idx >= c0
            idx = np.where(take, idx - c0, idx)
            bits[:, l] = take
            node = np.where(take, self.hi[l][node], self.lo[l][node]) - 2
        return self._labels_from_bits(bits)

    def all_labels(self, limit=5_000_000, batch=200_000):
        if self.count > limit:
            raise ValueError(f"{self.count:,} partitions exceeds limit {limit:,}")
        return np.concatenate([self.unrank(np.arange(s, min(s + batch, self.count)))
                               for s in range(0, self.count, batch)]) if self.count else \
            np.zeros((0, self.n), dtype=np.int8)

    # ----------------------------------------------------------------- reports --
    def zdd_size(self):
        """Node count after the ZDD rules (drop nodes with no path to ⊤; skip nodes
        whose 1-arc goes to ⊥; share nodes with equal (variable, lo, hi))."""
        m = self.m
        ids = {}
        below = None
        for l in range(m - 1, -1, -1):
            def red(codes):
                out = np.where(codes == 1, 1, 0).astype(np.int64)
                k = codes >= 2
                out[k] = below[codes[k] - 2]
                return out
            lo, hi = red(self.lo[l]), red(self.hi[l])
            mine = np.empty(len(lo), dtype=np.int64)
            for i in range(len(lo)):
                if hi[i] == 0:
                    mine[i] = lo[i]                     # zero-suppression rule
                else:
                    key = (l, int(lo[i]), int(hi[i]))
                    mine[i] = ids.setdefault(key, len(ids) + 2)
            below = mine
        return len(ids)

    def summary(self):
        return {"n": self.n, "m": self.m, "p": self.p, "order": self.order_name,
                "backend": getattr(self, "backend", "python"),
                "max_frontier": self.max_frontier, "nodes": self.nodes,
                "max_width": max(self.width), "count": self.count,
                "t_order": round(self.t_order, 3), "t_build": round(self.t_build, 3),
                "t_count": round(self.t_count, 3)}


# --------------------------------------------------------------- population ----

def district_pops(labels, pop, p):
    M, n = labels.shape
    flat = labels.astype(np.int64) + (np.arange(M) * p)[:, None]
    return np.bincount(flat.ravel(), weights=np.tile(pop, M), minlength=M * p).reshape(M, p)


def max_deviation(labels, pop, p):
    P = district_pops(labels, pop, p)
    ideal = pop.sum() / p
    return np.abs(P / ideal - 1).max(axis=1)


def population_ok(labels, pop, p, tol):
    """Mask of plans whose every district is within tol (e.g. 0.05) of the ideal."""
    return max_deviation(labels, pop, p) <= tol
