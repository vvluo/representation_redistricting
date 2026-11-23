"""Force an arbitrary labelling into D non-empty contiguous districts."""
from collections import deque, defaultdict
import numpy as np


def components(g, nodes):
    nodes = set(nodes); out = []
    while nodes:
        s = next(iter(nodes)); seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in g.nbr[u]:
                if v in nodes and v not in seen:
                    seen.add(v); q.append(v)
        out.append(seen); nodes -= seen
    return out


def repair(g, a, rng):
    """Force an arbitrary labelling into D non-empty contiguous districts."""
    a = np.asarray(a, dtype=int).copy()
    # 1. each district keeps only its largest component; rest are freed
    free = set()
    for d in range(g.D):
        mem = [i for i in range(g.N) if a[i] == d]
        if not mem:
            continue
        comps = components(g, mem)
        comps.sort(key=len, reverse=True)
        for c in comps[1:]:
            free |= c
    for i in free:
        a[i] = -1
    # 2. absorb freed counties into whichever neighbour shares the most border
    while True:
        pend = [i for i in range(g.N) if a[i] == -1]
        if not pend:
            break
        progressed = False
        for i in pend:
            cand = defaultdict(float)
            for j in g.nbr[i]:
                if a[j] >= 0:
                    cand[a[j]] += g.bl(i, j)
            if cand:
                a[i] = max(cand.items(), key=lambda kv: kv[1])[0]
                progressed = True
        if not progressed:                       # fully detached remainder
            a[pend[0]] = rng.randrange(g.D)
    # 3. refill any empty district by donating a non-articulation county
    for d in range(g.D):
        if (a == d).any():
            continue
        donor = max((x for x in range(g.D) if (a == x).sum() > 1),
                    key=lambda x: (a == x).sum())
        mem = [i for i in range(g.N) if a[i] == donor]
        for i in sorted(mem, key=lambda i: g.pop[i]):
            if len(components(g, [k for k in mem if k != i])) == 1:
                a[i] = d
                break
    return a
