"""Validate MCMC samplers against the ZDD's exact uniform distribution.

Benchmarks (all on Iowa, population limit --tol):
  burl2   70 tracts around Burlington (city + rural), 2 districts.  The ZDD lists
          every plan, so the reference is the exact distribution.
  burl3   same region, 3 districts.  Reference = iid uniform ZDD samples within tol.
  ia4     99 counties, 4 districts (the paper's Iowa test).  Reference = iid uniform
          ZDD samples within tol.

Sampler under test: `flip` (FlipChain, single-flip Metropolis-Hastings with a uniform
target).  `--recom-steps N` optionally adds plain ReCom (ReComChain), whose target is
not uniform, as a check that the comparison can detect a mismatch; it is off by default.
Chains start from deliberately extreme reference plans (least / most compact,
lowest / highest D_race) so that disagreement between chains shows poor mixing.

    python -m experiments.mcmc_benchmark --bench burl2 --tol 0.05 --walk-tol 0.10 \\
        --flip-steps 4000000
"""
import argparse, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting.geo import Geo
from redistricting import stats
from redistricting.enumerate.zdd import PartitionZDD, max_deviation

BENCH = {"burl2": ("IA_tract", "IA_tract_burlington70.txt", 2),
         "burl3": ("IA_tract", "IA_tract_burlington70.txt", 3),
         "ia4":   ("IA_county", None, 4)}


def load(bench):
    units, sub, p = BENCH[bench]
    g = Geo.load(units, p)
    if sub:
        ids = (ROOT / "data" / sub).read_text().split()
        g = g.subset([g.idx[x] for x in ids], p)
    return g, p


def reference(bench, tol, n_ref, out_dir, seed=0):
    """Uniform reference plans within tol: all of them if the ZDD can list them,
    else n_ref iid uniform samples.  Cached in out_dir."""
    f = out_dir / f"ref_{bench}_tol{tol}.npz"
    if f.exists():
        d = np.load(f); return d["L"], bool(d["exact"]), int(d["count"])
    g, p = load(bench)
    z = PartitionZDD(g.N, list(g.border), p, order="best", xy=g.xy, verbose=g.N > 90)
    print(f"[{bench}] ZDD: {z.summary()}", flush=True)
    if z.count <= 20_000_000:
        L = z.all_labels(limit=20_000_000); L = L[max_deviation(L, g.pop, p) <= tol]; exact = True
    else:
        rng = np.random.default_rng(seed); keep = []; got = drawn = 0
        while got < n_ref:
            S = z.sample(500_000, rng); drawn += len(S)
            S = S[max_deviation(S, g.pop, p) <= tol]; keep.append(S); got += len(S)
            print(f"  sampled {drawn:,}: {got:,} within {tol}", flush=True)
        L = np.concatenate(keep)[:n_ref]; exact = False
    np.savez_compressed(f, L=L, exact=exact, count=np.array(z.count, dtype=object).astype(str))
    return L, exact, z.count


def _chain(args):
    kind, bench, start, walk_tol, tol, steps, thin, seed = args
    sys.path.insert(0, str(ROOT))
    from redistricting.mcmc import FlipChain, ReComChain
    g, p = load(bench)
    ch = (FlipChain(g, start, p, walk_tol, np.random.default_rng(seed)) if kind == "flip"
          else ReComChain(g, start, p, walk_tol, seed))
    t = time.time()
    S = ch.run(steps, thin=thin, keep_tol=tol)
    return {"kind": kind, "L": S, "secs": time.time() - t,
            "accept": ch.accepted / ch.steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", choices=sorted(BENCH), default="burl2")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--walk-tol", type=float, default=0.10)
    ap.add_argument("--flip-steps", type=int, default=4_000_000)
    ap.add_argument("--flip-thin", type=int, default=100)
    ap.add_argument("--recom-steps", type=int, default=0, help="0 = skip the ReCom check")
    ap.add_argument("--recom-thin", type=int, default=1)
    ap.add_argument("--burn", type=float, default=0.2, help="fraction of each chain dropped")
    ap.add_argument("--n-ref", type=int, default=20_000)
    ap.add_argument("--vi-pairs", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out-dir", default="outputs")
    A = ap.parse_args()
    out = Path(A.out_dir); out.mkdir(parents=True, exist_ok=True)
    from redistricting.mcmc import rhat, ess, compare

    g, p = load(A.bench)
    R, exact, count = reference(A.bench, A.tol, A.n_ref, out)
    print(f"[{A.bench}] reference: {len(R):,} plans within {A.tol} "
          f"({'every valid plan' if exact else 'iid uniform ZDD samples'}); "
          f"all contiguous {p}-partitions: {count}", flush=True)
    RS = stats.all_stats(R, g, p)
    starts = [R[RS["mean_pp"].argmin()], R[RS["mean_pp"].argmax()],
              R[RS["D_race"].argmin()], R[RS["D_race"].argmax()]]

    tasks = [("flip", A.bench, s, A.walk_tol, A.tol, A.flip_steps, A.flip_thin, A.seed + k)
             for k, s in enumerate(starts)]
    if A.recom_steps:
        tasks += [("recom", A.bench, s, A.walk_tol, A.tol, A.recom_steps, A.recom_thin, A.seed + 10 + k)
                  for k, s in enumerate(starts)]
    with get_context("spawn").Pool(min(8, len(tasks))) as pool:
        res = pool.map(_chain, tasks)

    rng = np.random.default_rng(A.seed)
    rows = []; dump = {f"ref_{k}": v for k, v in RS.items()}
    ref_vi = stats.pairwise_vi(R, g.pop, p, A.vi_pairs, rng); dump["ref_vi"] = ref_vi
    for kind in [k for k in ("flip", "recom") if any(r["kind"] == k for r in res)]:
        chains = [r["L"][int(A.burn * len(r["L"])):] for r in res if r["kind"] == kind]
        secs = sum(r["secs"] for r in res if r["kind"] == kind)
        acc = np.mean([r["accept"] for r in res if r["kind"] == kind])
        CS = [stats.all_stats(c, g, p) for c in chains]
        pooled = np.concatenate(chains)
        print(f"[{A.bench}] {kind}: {len(pooled):,} kept states from {len(chains)} chains, "
              f"accept {acc:.2f}, {secs:.0f} CPU-s", flush=True)
        for k in RS:
            per = [c[k] for c in CS]
            x = np.concatenate(per); e = ess(per)
            row = {"bench": A.bench, "tol": A.tol, "sampler": kind, "stat": k,
                   "n": len(x), "rhat": rhat(per)}
            row.update(compare(RS[k], x, ess_x=e)); rows.append(row)
            dump[f"{kind}_{k}"] = x
        # pairwise VI: pairs drawn from post-burn-in states, thinned to ESS spacing
        vi = stats.pairwise_vi(pooled, g.pop, p, A.vi_pairs, rng); dump[f"{kind}_vi"] = vi
        row = {"bench": A.bench, "tol": A.tol, "sampler": kind, "stat": "pairwise_VI",
               "n": len(vi), "rhat": np.nan}
        row.update(compare(ref_vi, vi)); rows.append(row)

    D = pd.DataFrame(rows)
    tag = f"{A.bench}_tol{A.tol}"
    D.to_csv(out / f"mcmc_{tag}_summary.csv", index=False)
    np.savez_compressed(out / f"mcmc_{tag}_stats.npz", **dump)
    show = ["sampler", "stat", "n", "rhat", "ess", "mean_ref", "dmean", "dq5", "dq50", "dq95",
            "tail_lo_ref", "tail_lo_x", "tail_hi_ref", "tail_hi_x", "ks", "ks_p_ess"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(D[[c for c in show if c in D]].to_string(index=False))


if __name__ == "__main__":
    main()
