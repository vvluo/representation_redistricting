"""Invariant audit for the districting machinery, at any unit level.

Checks, independently of the code under test wherever possible:

  raw-geometry layer (needs the TIGER shapefile; skipped without --shp)
  P0 segment hygiene      : no segment repeated in one unit, none owned by >2 units
  P1 per-unit perimeter   : segment sum == stored perimeter; every singleton union too
  P2 outer boundary       : union of all units == known state boundary (if given)
  P3 union identity       : formula perimeter == boundary rebuilt from raw segments

  always on (uses only the unit graph)
  P4 merge                : P(A u B) == P(A) + P(B) - 2*shared(A,B)
  P5 split                : splitting a unit set and re-merging restores its perimeter
  P6 incremental delta    : score_after(i,d) == score of a freshly built post-flip plan
  C1 valid() vs brute     : valid(i,d) agrees with actually applying the flip and testing
  C2 union-find           : contiguity agrees with a different algorithm than BFS
  R1 ReCom trees & cuts   : trees span the union with n-1 real edges; both halves of
                            every cut are connected, disjoint and cover the union
  R2 ReCom incremental    : score_split == rebuilt plan; perimeters match the formula
  C3 fuzz                 : long random walk of valid flips keeps every invariant

    python -m redistricting.audit --unit AL_tract \\
        --shp $REDISTRICTING_TIGER/tl_2026_01_tract/tl_2026_01_tract --state 01
"""
import argparse, random, sys
from collections import defaultdict, deque, Counter
from .geo import Geo
from .plan import Plan
from .inits import seeded_growth
from .search.recom import spanning_tree, cuts, adjacent_pairs, TREE_MODES
from .borders import build_borders
from .shapefile import seg_len


def uf_components(g, nodes):
    """Connected components by union-find -- a different algorithm from the BFS used
    in Plan.connected_without / verify."""
    nodes = set(nodes); par = {n: n for n in nodes}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in nodes:
        for j in g.nbr[i]:
            if j in nodes:
                a, b = find(i), find(j)
                if a != b: par[a] = b
    return len({find(n) for n in nodes})


def bfs_connected(g, nodes):
    nodes = set(nodes)
    if not nodes: return True
    s = next(iter(nodes)); seen = {s}; q = deque([s])
    while q:
        u = q.popleft()
        for v in g.nbr[u]:
            if v in nodes and v not in seen: seen.add(v); q.append(v)
    return len(seen) == len(nodes)


def run(name, D=7, seed=20261122, nsub=60, nplans=4, nfuzz=600, shp_base=None,
        state=None, expected_outer_km=None, data_dir=None, verbose=True):
    """Run the audit on unit set `name`.  Returns the list of failed checks."""
    fails = []

    def chk(label, ok, detail=""):
        if not ok: fails.append(label)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))

    def say(*a):
        if verbose: print(*a)

    say(f"\n{'='*74}\nAUDIT  unit set={name}  D={D}  "
        f"raw-geometry checks={'on' if shp_base else 'off'}\n{'='*74}")
    rng = random.Random(seed)
    g = Geo.load(name, D=D, data_dir=data_dir)
    N = g.N
    say(f"units {N}   adjacent pairs {len(g.border)}")

    def formula(S):
        """Perimeter of the union of S, by the decomposition the code relies on."""
        S = set(S)
        p = sum(g.perim[i] for i in S)
        for (i, j), L in g.border.items():
            if i in S and j in S: p -= 2 * L
        return p

    if shp_base:
        gid, _, perim_m, border_m, owners, dup = build_borders(shp_base, state)
        # the shapefile is not in GEOID order but Geo sorts by it: remap to Geo's indices
        assert sorted(gid) == sorted(g.gid), "unit sets differ between features and shapefile"
        pos = {h: t for t, h in enumerate(g.gid)}
        remap = [pos[h] for h in gid]                   # shapefile index -> Geo index
        perim_m = [x for _, x in sorted(zip(remap, perim_m))]
        border_m = {(min(remap[i], remap[j]), max(remap[i], remap[j])): L
                    for (i, j), L in border_m.items()}
        owners = {k: {remap[u] for u in v} for k, v in owners.items()}
        assert set(border_m) == set(g.border), \
            f"border pair set differs from the features/borders CSVs: " \
            f"{len(set(border_m) ^ set(g.border))} pairs"
        unit_segs = defaultdict(list)
        for k, own in owners.items():
            for u in own: unit_segs[u].append(k)
        seglen = {k: seg_len(*k) for k in owners}

        def boundary_from_scratch(S):
            """Perimeter of the union of S, built only from raw segment ownership."""
            S = set(S); c = Counter()
            for u in S:
                for k in unit_segs[u]: c[k] += 1
            return sum(seglen[k] for k, n in c.items() if n == 1) / 1000.0

        say("\n-- perimeter (raw geometry) --")
        chk("P0 no segment duplicated inside one unit", dup == 0, f"dup={dup}")
        chk("P0 no segment owned by >2 units", all(len(v) <= 2 for v in owners.values()))
        err = max(abs(perim_m[k] / 1000 - g.perim[k]) for k in range(N))
        chk("P1 per-unit perimeter == stored perimeter", err < 1e-9, f"max abs err {err:.3e} km")
        for k in range(N):
            e = abs(boundary_from_scratch([k]) - g.perim[k])
            if e > 1e-9: chk(f"P1b singleton {g.gid[k]}", False, f"{e:.3e}"); break
        else: chk("P1b every singleton union == its own perimeter", True, f"{N} units")
        full = boundary_from_scratch(range(N))
        if expected_outer_km is not None:
            chk("P2 union of all units == known outer boundary",
                abs(full - expected_outer_km) < 0.1, f"{full:,.3f} km")
        chk("P2b formula agrees on the full union", abs(formula(range(N)) - full) < 1e-6,
            f"{formula(range(N)):,.3f} km")

        worst = 0.0
        sizes = [1, 2, 3, 5, 10, 25, 60, 150, 400, max(1, N // 2), N - 1]
        for _ in range(nsub):
            m = rng.choice([s for s in sizes if s <= N])
            S = rng.sample(range(N), m)
            worst = max(worst, abs(formula(S) - boundary_from_scratch(S)))
        chk("P3 formula == from-scratch boundary, arbitrary subsets",
            worst < 1e-6, f"{nsub} subsets, max abs err {worst:.3e} km")

        worstc = 0.0                      # connected subsets (districts are connected)
        for _ in range(nsub // 2):
            s = rng.randrange(N); S = {s}
            tgt = rng.choice([2, 5, 20, 80, min(300, N)])
            front = list(g.nbr[s])
            while len(S) < tgt and front:
                j = front.pop(rng.randrange(len(front)))
                if j not in S: S.add(j); front += list(g.nbr[j])
            worstc = max(worstc, abs(formula(S) - boundary_from_scratch(S)))
        chk("P3b same, for connected (district-shaped) subsets", worstc < 1e-6,
            f"max abs err {worstc:.3e} km")

    say("\n-- merge / split --")
    wm = ws = 0.0
    for _ in range(nsub // 2):
        A = set(rng.sample(range(N), rng.choice([2, 8, 40, min(200, N)])))
        B = set(rng.sample([x for x in range(N) if x not in A], min(len(A), N - len(A))))
        shared = sum(L for (i, j), L in g.border.items()
                     if (i in A and j in B) or (i in B and j in A))
        wm = max(wm, abs(formula(A | B) - (formula(A) + formula(B) - 2 * shared)))
        parts = [set(), set()]
        for x in A: parts[rng.randrange(2)].add(x)
        if parts[0] and parts[1]:
            sh = sum(L for (i, j), L in g.border.items()
                     if (i in parts[0] and j in parts[1]) or (i in parts[1] and j in parts[0]))
            ws = max(ws, abs(formula(A) - (formula(parts[0]) + formula(parts[1]) - 2 * sh)))
    chk("P4 merge: P(A u B) == P(A)+P(B)-2*shared", wm < 1e-6, f"max abs err {wm:.3e} km")
    chk("P5 split: pieces re-merge to the whole", ws < 1e-6, f"max abs err {ws:.3e} km")

    say("\n-- incremental update --")
    wd = 0.0; nflip = 0
    for t in range(nplans):
        p = Plan(g, seeded_growth(g, rng), 6.0, "tvd")
        for i, d in list(p.candidates()):
            if not p.valid(i, d): continue
            pred = p.score_after(i, d)
            q = Plan(g, p.a.copy(), 6.0, "tvd"); q.apply(i, d)
            wd = max(wd, abs(pred - q.score())); nflip += 1
            for dd in (d, int(p.a[i])):
                wd = max(wd, abs(q.perim[dd] - formula(q.members[dd])))
    chk("P6 score_after == rebuilt plan's score, and perimeters match the formula",
        wd < 1e-9, f"{nflip} flips over {nplans} plans, max abs err {wd:.3e}")

    say("\n-- contiguity --")
    bad = badnc = 0; ncand = 0; nnon = 0
    for t in range(nplans):
        p = Plan(g, seeded_growth(g, rng), 6.0, "tvd")
        for dd in range(D):
            if uf_components(g, p.members[dd]) != 1: bad += 1
        cand = set(p.candidates())
        for i, d in cand:
            ncand += 1
            old = int(p.a[i])
            truth = (len(p.members[old]) > 1
                     and bfs_connected(g, p.members[old] - {i})
                     and bfs_connected(g, p.members[d] | {i}))
            if p.valid(i, d) != truth: badnc += 1
        allp = [(i, d) for i in range(N) for d in range(D) if (i, d) not in cand]
        for i, d in rng.sample(allp, min(400, len(allp))):
            nnon += 1
            if d == p.a[i]: continue
            old = int(p.a[i])
            truth = (len(p.members[old]) > 1
                     and bfs_connected(g, p.members[d] | {i}))
            if p.valid(i, d) or truth: badnc += 1
    chk("C1 valid() agrees with brute-force apply-and-test", badnc == 0,
        f"{ncand} candidate + {nnon} non-candidate pairs, {badnc} disagreements")
    chk("C2 union-find agrees with BFS on every district", bad == 0, f"{nplans * D} districts")

    say("\n-- recom --")
    for tm in TREE_MODES:
        p = Plan(g, seeded_growth(g, rng), 6.0, "tvd")
        bad = 0; nsplit = 0; wsc = 0.0; ntree = 0
        for _ in range(60):
            pr = adjacent_pairs(p)
            if not pr: break
            d1, d2 = pr[rng.randrange(len(pr))]
            union = p.members[d1] | p.members[d2]
            par = spanning_tree(g, union, rng, tm); ntree += 1
            # a spanning tree must cover the union exactly and have n-1 edges
            if set(par) != union: bad += 1
            if sum(1 for v in par.values() if v is not None) != len(union) - 1: bad += 1
            if sum(1 for v in par.values() if v is None) != 1: bad += 1
            for v, q in par.items():
                if q is not None and q not in g.nbr[v]: bad += 1   # real graph edge
            cl = cuts(g, par, 0.02, g.ideal)
            for A, B in cl:
                nsplit += 1
                if A & B or (A | B) != union or not A or not B: bad += 1
                if not bfs_connected(g, A) or not bfs_connected(g, B): bad += 1
                pred = p.score_split(d1, d2, A, B)
                q2 = Plan(g, p.a.copy(), 6.0, "tvd"); q2.apply_split(d1, d2, A, B)
                wsc = max(wsc, abs(pred - q2.score()))
                for dd in (d1, d2):
                    wsc = max(wsc, abs(q2.perim[dd] - formula(q2.members[dd])))
            if cl:
                A, B = cl[rng.randrange(len(cl))]
                p.apply_split(d1, d2, A, B)
                for dd in range(D):
                    if not p.members[dd] or uf_components(g, p.members[dd]) != 1: bad += 1
                    if abs(p.perim[dd] - formula(p.members[dd])) > 1e-9: bad += 1
                if abs(p.pop.sum() - g.pop.sum()) > 1e-6: bad += 1
        chk(f"R1 {tm}: trees valid, halves connected & disjoint, splits sound",
            bad == 0, f"{ntree} trees, {nsplit} cuts, {bad} violations")
        chk(f"R2 {tm}: score_split == rebuilt plan, perimeters match formula",
            wsc < 1e-9, f"max abs err {wsc:.3e}")

    say("\n-- fuzz --")
    p = Plan(g, seeded_growth(g, rng), 6.0, "tvd")
    pop0 = p.pop.sum(); moves = 0; bad = 0
    for _ in range(nfuzz):
        cand = [c for c in p.candidates()]
        rng.shuffle(cand)
        for i, d in cand:
            if p.valid(i, d): p.apply(i, d); moves += 1; break
        else: break
        if moves % max(1, nfuzz // 12) == 0:
            for dd in range(D):
                if not p.members[dd] or uf_components(g, p.members[dd]) != 1: bad += 1
                if abs(p.perim[dd] - formula(p.members[dd])) > 1e-9: bad += 1
            if abs(p.pop.sum() - pop0) > 1e-6: bad += 1
    chk("C3 random walk of valid flips preserves every invariant", bad == 0,
        f"{moves} flips applied, {bad} violations")
    say(f"\n  all {D} districts non-empty at end: {all(len(p.members[d]) for d in range(D))}")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description="Invariant audit for the districting machinery.")
    ap.add_argument("--unit", default="AL_tract", help="unit set name, e.g. AL_county, AL_tract")
    ap.add_argument("--districts", type=int, default=7)
    ap.add_argument("--subsets", type=int, default=60)
    ap.add_argument("--plans", type=int, default=4)
    ap.add_argument("--fuzz", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--shp", help="TIGER shapefile path without extension: enables the "
                                  "raw-geometry checks")
    ap.add_argument("--state", help="STATEFP to keep from the shapefile, e.g. 01")
    ap.add_argument("--expect-outer-km", type=float, help="known outer boundary length")
    ap.add_argument("--data-dir")
    A = ap.parse_args(argv)
    fails = run(A.unit, A.districts, A.seed, A.subsets, A.plans, A.fuzz, A.shp, A.state,
                A.expect_outer_km, A.data_dir)
    print(f"\n{'='*74}")
    print("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
