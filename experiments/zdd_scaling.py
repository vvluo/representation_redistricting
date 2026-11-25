"""How far does the partition ZDD go?  (cf. Fifield et al. 2020, sec. 3)

Builds the ZDD of all p-partitions for
  * random contiguous subsets of a unit map, grown as in the paper: start at a random
    unit, keep a queue of neighbours, add a random queued unit until the size is hit;
  * optionally the whole map.
Every build runs in a fresh process with a node and a time budget; the table records
the max frontier, nodes, widest level, build time, peak memory and the count, or
where the build was stopped.

    python -m experiments.zdd_scaling --units IA_tract --sizes 30 40 50 60 70 85 100 \\
        --p 2 4 --seeds 2 --max-nodes 8000000 --max-seconds 180 --jobs 3
"""
import argparse, json, resource, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd


def grow_subset(nbr, size, rng):
    start = int(rng.integers(len(nbr)))
    S = {start}; queue = list(nbr[start])
    while len(S) < size:
        if not queue:
            return None
        k = int(rng.integers(len(queue))); v = queue.pop(k)
        if v in S:
            continue
        S.add(v); queue.extend(w for w in nbr[v] if w not in S)
    return sorted(S)


def _job(args):
    units, idx, p, max_nodes, max_seconds, tag, backend = args
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from redistricting.geo import Geo
    from redistricting.enumerate.zdd import PartitionZDD, BuildAborted
    g = Geo.load(units, p)
    pos = {v: i for i, v in enumerate(idx)}
    E = [(pos[a], pos[b]) for (a, b) in g.border if a in pos and b in pos]
    xy = g.xy[idx] if g._lat is not None else None
    row = {"tag": tag, "units": units, "n": len(idx), "m": len(E), "p": p, "backend": backend}
    t0 = time.time()
    try:
        z = PartitionZDD(len(idx), E, p, order="best", xy=xy, backend=backend,
                         max_nodes=max_nodes, max_seconds=max_seconds)
        s = z.summary()
        row.update(status="done", order=s["order"], max_frontier=s["max_frontier"],
                   nodes=s["nodes"], max_width=s["max_width"],
                   t_build=s["t_build"], t_total=round(time.time() - t0, 2),
                   count=str(z.count), log10_count=round(float(np.log10(float(z.count))), 2)
                   if z.count else None)
    except BuildAborted as e:
        from redistricting.enumerate.ordering import best_order, frontiers
        o, name, f, _ = best_order(len(idx), E, xy=xy)
        row.update(status="stopped", order=name, max_frontier=f, note=str(e),
                   t_total=round(time.time() - t0, 2))
    row["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                               / (1e9 if sys.platform == "darwin" else 1e6), 2)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", default="IA_tract")
    ap.add_argument("--sizes", type=int, nargs="+", default=[30, 40, 50, 60, 70, 85, 100])
    ap.add_argument("--p", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--whole", action="store_true", help="also build the whole map")
    ap.add_argument("--max-nodes", type=int, default=8_000_000)
    ap.add_argument("--max-seconds", type=float, default=180)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--backend", choices=["python", "cpp"], default="python")
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", default="outputs/zdd_scaling.csv")
    A = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from redistricting.geo import Geo
    g = Geo.load(A.units, 2)
    rng = np.random.default_rng(A.seed)
    tasks = []
    for k in range(A.seeds):
        for size in A.sizes:
            idx = grow_subset(g.nbr, size, rng)
            for p in A.p:
                tasks.append((A.units, idx, p, A.max_nodes, A.max_seconds, f"subset{k}", A.backend))
    if A.whole:
        for p in A.p:
            tasks.append((A.units, list(range(g.N)), p, A.max_nodes, A.max_seconds, "whole", A.backend))

    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(A.jobs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, tasks):
            rows.append(r)
            print(json.dumps(r), flush=True)
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(rows).sort_values(["tag", "n", "p"])
    D.to_csv(out, index=False)
    print(D.drop(columns=[c for c in ("note", "count", "units") if c in D]).to_string(index=False))


if __name__ == "__main__":
    main()
