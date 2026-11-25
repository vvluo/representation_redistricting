"""Build the partition ZDD of a whole unit map, count it, and sample it uniformly.

Reports the exact number of contiguous p-partitions and, from iid uniform samples,
the share meeting each population tolerance (cf. Fifield et al. 2020, Fig. 10 for
Iowa).  Plans meeting a tolerance are saved, since uniform samples filtered by a
population limit are uniform on the plans meeting that limit.

    python -m experiments.zdd_sample --units IA_county --p 4 --samples 5000000 \\
        --tol 0.01 0.05 0.10 --max-nodes 150000000 --max-seconds 3000
"""
import argparse, json, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redistricting.geo import Geo
from redistricting.enumerate.zdd import PartitionZDD, max_deviation


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", default="IA_county")
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--tol", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--max-nodes", type=int, default=150_000_000)
    ap.add_argument("--max-seconds", type=float, default=3000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out-dir", default="outputs")
    A = ap.parse_args()

    g = Geo.load(A.units, A.p)
    z = PartitionZDD(g.N, list(g.border), A.p, order="best", xy=g.xy,
                     max_nodes=A.max_nodes, max_seconds=A.max_seconds, verbose=True)
    s = z.summary()
    print(json.dumps(s, default=str))
    print(f"exact count: {z.count}  (~10^{np.log10(float(z.count)):.2f})", flush=True)

    rng = np.random.default_rng(A.seed)
    out = Path(A.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); hits = {t: 0 for t in A.tol}; keep = []
    batch = 250_000
    for b in range(0, A.samples, batch):
        L = z.sample(min(batch, A.samples - b), rng)
        dev = max_deviation(L, g.pop, A.p)
        for t in A.tol:
            hits[t] += int((dev <= t).sum())
        m = dev <= max(A.tol)
        keep.append(np.column_stack([L[m], dev[m]]))
    ts = time.time() - t0
    K = np.concatenate(keep)
    cols = [g.gid[i] for i in range(g.N)] + ["max_dev"]
    pd.DataFrame(K, columns=cols).to_csv(out / f"zdd_{A.units}_p{A.p}_valid_samples.csv", index=False)
    rows = [{"tol": t, "hits": hits[t], "share": hits[t] / A.samples,
             "est_plans": hits[t] / A.samples * z.count} for t in A.tol]
    print(f"{A.samples:,} uniform samples in {ts:.1f}s")
    print(pd.DataFrame(rows).to_string(index=False))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    print(f"peak RSS {rss:.2f} GB")


if __name__ == "__main__":
    main()
