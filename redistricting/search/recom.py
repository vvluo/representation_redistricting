"""ReCom (recombination) moves for the districting search.

One move: take two adjacent districts, union them, draw a spanning tree of that
union, and cut one tree edge.  Both halves are subtrees, so both are connected by
construction -- no contiguity repair is ever needed.  The cut edge is chosen by
population, so a single move can shift a large, coordinated block of units, which
is what single-unit border flips cannot do.

SPANNING TREE SAMPLING -- three options, because the tree distribution *is* the
proposal distribution:

  ust      Wilson's algorithm (loop-erased random walk).  Samples uniformly from
           all spanning trees of the union.  Pursues no minimum at all; it is the
           unbiased choice, and the one that makes ReCom's stationary distribution
           well defined if this is ever run as an MCMC sampler rather than an
           optimiser.

  rmst     Minimum spanning tree under i.i.d. U(0,1) edge weights.  The "minimum"
           it pursues is the sum of meaningless random numbers -- it is purely a
           randomisation device, not a geometric objective.  Cheaper than Wilson
           and the common practical default, but it is NOT uniform: it over-samples
           trees built from locally-cheap edges.

  compact  Minimum spanning tree under weight U(0,1) / shared_border_length.
           Here the minimum is meaningful: cheap edges are long shared borders, so
           the tree preferentially keeps strongly-joined units together and the
           bipartitions it offers tend to be separated by little boundary.  This
           biases proposals toward compact districts -- useful for an optimiser,
           but it deliberately distorts the proposal distribution, so it should not
           be used if the goal is sampling.

For a greedy optimiser the choice only affects which splits get proposed, never
correctness: every cut is contiguous and is accepted only if it improves f.
"""
from collections import defaultdict, deque

TREE_MODES = ("ust", "rmst", "compact")


def induced(g, nodes):
    ns = set(nodes)
    return {u: [v for v in g.nbr[u] if v in ns] for u in ns}


def tree_ust(g, sub, rng):
    """Wilson's algorithm: uniform over spanning trees."""
    nodes = list(sub); root = rng.choice(nodes)
    parent = {root: None}; intree = {root}
    for s in nodes:
        if s in intree: continue
        nxt = {}; u = s
        while u not in intree:                 # loop-erased random walk
            v = rng.choice(sub[u]); nxt[u] = v; u = v
        u = s
        while u not in intree:                 # retrace, erasing loops
            intree.add(u); parent[u] = nxt[u]; u = nxt[u]
    return parent


def tree_mst(g, sub, rng, weighted):
    """Kruskal MST under random (or border-scaled random) edge weights."""
    edges = []
    for u in sub:
        for v in sub[u]:
            if v > u:
                w = rng.random()
                if weighted:
                    w /= max(g.bl(u, v), 1e-9)   # long shared border -> cheap edge
                edges.append((w, u, v))
    edges.sort()
    par = {u: u for u in sub}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    adj = defaultdict(list); taken = 0
    for w, u, v in edges:
        a, b = find(u), find(v)
        if a == b: continue
        par[a] = b; adj[u].append(v); adj[v].append(u); taken += 1
        if taken == len(sub) - 1: break
    root = next(iter(sub)); parent = {root: None}; q = deque([root]); seen = {root}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); parent[v] = u; q.append(v)
    return parent


def spanning_tree(g, nodes, rng, mode):
    sub = induced(g, nodes)
    if mode == "ust":     return tree_ust(g, sub, rng)
    if mode == "rmst":    return tree_mst(g, sub, rng, False)
    if mode == "compact": return tree_mst(g, sub, rng, True)
    raise ValueError(mode)


def cuts(g, parent, tol, ideal):
    """Every tree edge whose removal leaves both halves within tol of ideal,
    plus the single most balanced edge as a fallback.  Yields (A, B)."""
    children = defaultdict(list); root = None
    for v, p in parent.items():
        (children[p].append(v) if p is not None else None)
        if p is None: root = v
    order = []; st = [root]
    while st:
        u = st.pop(); order.append(u); st.extend(children[u])
    sp = {}
    for u in reversed(order):
        sp[u] = g.pop[u] + sum(sp[c] for c in children[u])
    total = sp[root]
    cand = []; best = None
    for v in order:
        if v == root: continue
        a, b = sp[v], total - sp[v]
        dev = max(abs(a - ideal), abs(b - ideal)) / ideal
        if dev <= tol: cand.append(v)
        if best is None or dev < best[0]: best = (dev, v)
    if not cand and best is not None: cand = [best[1]]
    out = []
    for v in cand:
        A = set(); st = [v]
        while st:
            u = st.pop(); A.add(u); st.extend(children[u])
        out.append((A, set(parent) - A))
    return out


def adjacent_pairs(p):
    s = set()
    for (i, j) in p.g.border:
        a, b = int(p.a[i]), int(p.a[j])
        if a != b: s.add((min(a, b), max(a, b)))
    return sorted(s)


def recom_climb(p, rng, tree="compact", tol=0.02, trees=2,
                patience=60, max_steps=4000):
    """Greedy ReCom: accept a proposed split only when it raises f."""
    cur = p.score(); steps = 0; fails = 0; props = 0
    while steps < max_steps and fails < patience:
        pairs = adjacent_pairs(p)
        if not pairs: break
        d1, d2 = pairs[rng.randrange(len(pairs))]
        union = p.members[d1] | p.members[d2]
        best = None
        for _ in range(trees):
            props += 1
            par = spanning_tree(p.g, union, rng, tree)
            for A, B in cuts(p.g, par, tol, p.g.ideal):
                s = p.score_split(d1, d2, A, B)
                if s > cur + 1e-12 and (best is None or s > best[0]):
                    best = (s, A, B)
        if best:
            p.apply_split(d1, d2, best[1], best[2]); cur = best[0]
            steps += 1; fails = 0
        else:
            fails += 1
    return p, steps, props
