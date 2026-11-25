"""Edge orderings for the frontier-based ZDD, and the frontiers they induce.

The ZDD's size is driven by the largest frontier: the set of vertices touched both
by an already-processed edge and by a still-unprocessed one (Fifield, Imai,
Kawahara & Kenny 2020, sec. 2.3.2).  Section 2.3.4 of that paper orders edges by
recursive minimum vertex cuts:

  1. pick s, t at maximum shortest-path distance (the paper uses Floyd-Warshall);
  2. take a minimum s-t vertex cut S (max-flow) and split G - S into components;
  3. give each component C its edge set E_C = edges inside C and edges C-S, and
     place the block containing s first, the one containing t last;
  4. contract S to one vertex; it becomes t inside the s-block and s inside the
     t-block, and recurse until a block has <= `base` edges.

Every cut S is (close to) one of the frontiers, so small cuts give small frontiers.

On step 1: Floyd-Warshall is O(V^3), but on an unweighted graph the same all-pairs
distances come from one BFS per vertex in O(V*E), which for a planar map is O(V^2).
Both are provided (`farthest_pair(method=...)`) and must agree; neither is the
bottleneck of building the ZDD for any graph the ZDD itself can handle.
"""
import itertools
from collections import deque
import numpy as np
import networkx as nx


# ---------------------------------------------------------------- distances ----

def _adj(n, edges):
    A = [[] for _ in range(n)]
    for u, v in edges:
        A[u].append(v); A[v].append(u)
    return A


def bfs_dist(A, src):
    d = [-1] * len(A); d[src] = 0; q = deque([src])
    while q:
        x = q.popleft()
        for y in A[x]:
            if d[y] < 0:
                d[y] = d[x] + 1; q.append(y)
    return d


def farthest_pair(n, edges, method="bfs"):
    """(s, t, distance) maximising the shortest-path distance; ties -> smallest (s, t).

    method="floyd" runs scipy's Floyd-Warshall (O(V^3), as in the paper);
    method="bfs" runs one BFS per vertex (O(V*E)).  Same answer on unweighted graphs.
    """
    if method == "floyd":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import floyd_warshall
        r = [u for u, v in edges] + [v for u, v in edges]
        c = [v for u, v in edges] + [u for u, v in edges]
        D = floyd_warshall(csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)),
                           directed=False, unweighted=True)
        if np.isinf(D).any():
            raise ValueError("graph is disconnected")
        D = D.astype(np.int64)
    elif method == "bfs":
        A = _adj(n, edges)
        D = np.array([bfs_dist(A, s) for s in range(n)], dtype=np.int64)
        if (D < 0).any():
            raise ValueError("graph is disconnected")
    else:
        raise ValueError(method)
    k = int(D.argmax()); s, t = divmod(k, n)
    return s, t, int(D[s, t])


# ---------------------------------------------------------------- frontiers ----

def frontiers(n, order):
    """List of frontiers F_1..F_m (sets) for an edge order, per the paper's definition."""
    first, last = {}, {}
    for l, (u, v) in enumerate(order):
        for w in (u, v):
            first.setdefault(w, l); last[w] = l
    F, cur = [], set()
    for l, (u, v) in enumerate(order):
        for w in (u, v):
            if first[w] == l: cur.add(w)
        for w in (u, v):
            if last[w] == l: cur.discard(w)
        F.append(set(cur))
    return F


def max_frontier(n, order):
    return max((len(f) for f in frontiers(n, order)), default=0)


# ---------------------------------------------------------------- orderings ----

def order_bfs(n, edges, s=None):
    """Baseline: rank vertices by BFS from s (a farthest-pair end if s is None);
    sort edges by (later endpoint rank, earlier endpoint rank)."""
    if s is None:
        s = farthest_pair(n, edges)[0]
    A = _adj(n, edges)
    rank = {}; q = deque([s]); rank[s] = 0
    while q:
        x = q.popleft()
        for y in sorted(A[x]):
            if y not in rank:
                rank[y] = len(rank); q.append(y)
    return sorted(((min(u, v), max(u, v)) for u, v in edges),
                  key=lambda e: (max(rank[e[0]], rank[e[1]]), min(rank[e[0]], rank[e[1]])))


def order_cut(n, edges, base=5, method="bfs", seed=0):
    """Paper sec. 2.3.4: recursive minimum s-t vertex cuts."""
    s, t, _ = farthest_pair(n, edges, method)
    H = nx.Graph()
    H.add_nodes_from(range(n))
    for u, v in edges:
        H.add_edge(u, v, orig=[(min(u, v), max(u, v))])
    counter = itertools.count()
    out = _cut_rec(H, s, t, base, counter)
    assert sorted(out) == sorted((min(u, v), max(u, v)) for u, v in edges), "lost edges"
    return out


def _orig_edges(H, nodes=None):
    E = H.edges(data="orig") if nodes is None else H.subgraph(nodes).edges(data="orig")
    return [e for _, _, o in E for e in o]


def _small_order(H, s):
    """Base case: BFS from s over the block, edges sorted by later-endpoint rank."""
    rank = {}; q = deque([s]); rank[s] = 0
    while q:
        x = q.popleft()
        for y in sorted(H[x], key=str):
            if y not in rank:
                rank[y] = len(rank); q.append(y)
    E = sorted(H.edges(data="orig"),
               key=lambda e: (max(rank[e[0]], rank[e[1]]), min(rank[e[0]], rank[e[1]])))
    return [x for _, _, o in E for x in o]


def _cut_rec(H, s, t, base, counter):
    if H.number_of_edges() <= base or s == t:
        return _small_order(H, s)
    st = []
    if H.has_edge(s, t):
        # No vertex cut separates adjacent s, t.  Set the s-t edge aside, cut the
        # rest, and place that edge between the first block and the next.  (Falling
        # back to a plain BFS order here instead lets large blocks through and was
        # measured to double the max frontier on the Iowa county map: 24 vs 11.)
        st = list(H[s][t]["orig"]); H = H.copy(); H.remove_edge(s, t)
        if not nx.has_path(H, s, t):                 # the s-t edge was a bridge
            return _small_order(H.subgraph(nx.node_connected_component(H, s)).copy(), s) \
                + st + _small_order(H.subgraph(nx.node_connected_component(H, t)).copy(), t)
    S = nx.minimum_node_cut(H, s, t)
    R = H.copy(); R.remove_nodes_from(S)
    comps = list(nx.connected_components(R))
    cs = next(c for c in comps if s in c); ct = next(c for c in comps if t in c)
    mid = [c for c in comps if c is not cs and c is not ct]
    dist = nx.single_source_shortest_path_length(H, s)
    mid.sort(key=lambda c: min(dist[x] for x in c))           # "arbitrary" order, made reproducible
    sid = ("cut", next(counter))
    out = []
    for k, C in enumerate([cs] + mid + [ct]):
        Hk = nx.Graph(); Hk.add_nodes_from(C); Hk.add_node(sid)
        for x, y, o in H.subgraph(C).edges(data="orig"):
            Hk.add_edge(x, y, orig=list(o))
        for x in C:                                            # edges C - S collapse onto sid
            for y in H[x]:
                if y in S:
                    if Hk.has_edge(x, sid): Hk[x][sid]["orig"] += H[x][y]["orig"]
                    else:                   Hk.add_edge(x, sid, orig=list(H[x][y]["orig"]))
        if s in C:   sk, tk = s, sid
        elif t in C: sk, tk = sid, t
        else:
            d = nx.single_source_shortest_path_length(Hk, sid)
            sk, tk = sid, max(d, key=lambda x: (d[x], str(x)))
        out += _cut_rec(Hk, sk, tk, base, counter)
        if k == 0:                                             # edges inside S sit between blocks
            out += _orig_edges(H, S) + st
    return out


def order_sweep(n, edges, xy, angle_deg):
    """Rank vertices by projection of their planar coordinates on a direction;
    sort edges by (later endpoint rank, earlier endpoint rank)."""
    import math
    xy = np.asarray(xy, float)
    a = math.radians(angle_deg)
    rank = np.argsort(np.argsort(xy @ np.array([math.cos(a), math.sin(a)]), kind="stable"))
    return sorted(((min(u, v), max(u, v)) for u, v in edges),
                  key=lambda e: (max(rank[e[0]], rank[e[1]]), min(rank[e[0]], rank[e[1]])))


def best_order(n, edges, xy=None, candidates=("cut5", "cut10", "bfs_s", "bfs_t", "sweep"),
               method="bfs"):
    """Try several orderings; keep the one with the smallest summed frontier cost
    (`frontier_cost`), ties broken by max frontier.

    Returns (order, name, max_frontier, {name: max_frontier}).  No single ordering
    wins everywhere (measured max frontiers):
                    cut   BFS(s)  best sweep
        IA counties  12     11      10
        AL counties  10      9       9
        IA tracts    46     51       -
        AL tracts    47     81       -
    "sweep" (36 directions, 5 degrees apart) needs planar coordinates `xy`.
    """
    s, t, _ = farthest_pair(n, edges, method)
    make = {"cut5":  lambda: order_cut(n, edges, base=5, method=method),
            "cut10": lambda: order_cut(n, edges, base=10, method=method),
            "bfs_s": lambda: order_bfs(n, edges, s),
            "bfs_t": lambda: order_bfs(n, edges, t)}
    if xy is not None:
        for ang in range(0, 180, 5):
            make[f"sweep{ang}"] = (lambda a=ang: order_sweep(n, edges, xy, a))
    candidates = [c for c in candidates if c != "sweep"] + \
                 ([k for k in make if k.startswith("sweep")] if "sweep" in candidates else [])
    tried = {}
    best = None
    for c in candidates:
        o = make[c](); f, cost = frontier_cost(n, o); tried[c] = f
        if best is None or (cost, f) < (best[3], best[2]):
            best = (o, c, f, cost)
    return best[:3] + (tried,)


def frontier_cost(n, order, base=8.0):
    """(max frontier, sum over levels of base**|F_l|).

    The summed cost is the selection rule: node counts grow about 8x per extra
    frontier vertex (measured on grids), and how many levels sit at the peak matters
    as well as the peak.  On a 60-tract Iowa subset two orderings with the same max
    frontier (8) built 510,593 vs 71,385 nodes; the summed cost picks the second."""
    F = [len(f) for f in frontiers(n, order)]
    return max(F, default=0), sum(base ** k for k in F)


ORDERS = {"cut": order_cut, "bfs": order_bfs, "sweep": order_sweep,
          "best": lambda n, e, **kw: best_order(n, e, **kw)[0]}


def edge_order(n, edges, how="cut", **kw):
    if how not in ORDERS:
        raise ValueError(f"unknown ordering {how!r}; choose from {sorted(ORDERS)}")
    return ORDERS[how](n, edges, **kw)
