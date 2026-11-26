"""Population balancing for starter maps: ReCom moves that only ever reduce imbalance.

Most initialisations give contiguous but unequal districts (median max deviation
13-91% on Iowa and Alabama).  `balance` repairs population without redrawing the
whole map: each step takes the district farthest from the ideal and one of its
neighbours, draws a uniform spanning tree (Wilson) of their union, and cuts the tree
edge whose two halves are closest to ideal.  The move is kept only if the two
districts' worst deviation goes down, so districts the step does not touch keep their
shape and the start's character survives.  Both halves of a tree cut are connected,
so contiguity is preserved by construction.

    a = balance(g, a, rng, tol=0.05)      # rng: random.Random
"""
from collections import defaultdict
import numpy as np
from ..search.recom import induced, tree_ust


def _dev(P, ideal):
    return np.abs(P / ideal - 1)


def balance(g, a, rng, tol=0.05, max_steps=5000, trees_per_step=4):
    """Return (assignment, steps, final max deviation).  a is not modified."""
    a = np.array(a, dtype=np.int64)
    D = g.D; ideal = g.pop.sum() / D
    P = np.bincount(a, g.pop, D)
    nbr_d = lambda d: sorted({int(a[j]) for i in np.flatnonzero(a == d) for j in g.nbr[i]} - {d})
    for step in range(max_steps):
        dev = _dev(P, ideal)
        if dev.max() <= tol:
            return a, step, float(dev.max())
        order = np.argsort(-dev)
        moved = False
        for d1 in order[:2]:                       # worst district, then second worst
            d1 = int(d1)
            cand = nbr_d(d1)
            rng.shuffle(cand)
            cand.sort(key=lambda d: -dev[d])       # prefer the neighbour also far from ideal
            for d2 in cand[:3]:
                nodes = list(np.flatnonzero((a == d1) | (a == d2)))
                sub = induced(g, nodes)
                best = None
                for _ in range(trees_per_step):
                    parent = tree_ust(g, sub, rng)
                    ch = defaultdict(list); root = None
                    for v, q in parent.items():
                        if q is None: root = v
                        else: ch[q].append(v)
                    order_t = []; st = [root]
                    while st:
                        u = st.pop(); order_t.append(u); st.extend(ch[u])
                    sp = {}
                    for u in reversed(order_t):
                        sp[u] = g.pop[u] + sum(sp[c] for c in ch[u])
                    tot = sp[root]
                    for v in order_t[1:]:
                        w = max(abs(sp[v] / ideal - 1), abs((tot - sp[v]) / ideal - 1))
                        if best is None or w < best[0]:
                            best = (w, v, parent, ch)
                if best and best[0] < max(dev[d1], dev[d2]) - 1e-12:
                    w, v, parent, ch = best
                    A = set(); st = [v]
                    while st:
                        u = st.pop(); A.add(u); st.extend(ch[u])
                    for u in nodes:
                        a[u] = d1 if u in A else d2
                    P = np.bincount(a, g.pop, D)
                    moved = True
                    break
            if moved:
                break
        if not moved:                              # stuck: no improving split found
            return a, step, float(_dev(P, ideal).max())
    return a, max_steps, float(_dev(P, ideal).max())
