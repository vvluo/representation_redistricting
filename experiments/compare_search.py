"""Compare move sets from the same random starts: flips vs ReCom vs the hybrid.

    python -m experiments.compare_search --unit AL_county AL_tract --restarts 5

Methods: steepest flips; ReCom with each spanning-tree sampler (ust, rmst, compact);
and the hybrid (compact ReCom, then flips to polish the boundary, until neither helps).
"""
import argparse, random, time
import numpy as np
from redistricting import (Geo, Plan, verify, hill_climb, recom_climb, hybrid)
from redistricting.inits import seeded_growth

METHODS = [("flips only", None), ("recom ust", "ust"), ("recom rmst", "rmst"),
           ("recom compact", "compact"), ("recom compact+flips", "hybrid")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unit", nargs="+", default=["AL_county", "AL_tract"])
    ap.add_argument("--districts", type=int, default=7)
    ap.add_argument("--lam", type=float, default=6.0,
                    help="weight on the population term relative to compactness")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="per-cut population tolerance for ReCom")
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--data-dir")
    A = ap.parse_args(argv)

    for unit in A.unit:
        g = Geo.load(unit, D=A.districts, data_dir=A.data_dir)
        print(f"\n{'='*86}\n{unit}  N={g.N}  D={A.districts}  "
              f"lambda={A.lam}  tol={A.tol:.0%}  {A.restarts} restarts\n{'='*86}")
        print(f"{'method':22} {'medPP':>7} {'med maxdev%':>12} {'med moved%':>11} "
              f"{'med f':>8} {'best f':>8} {'moves':>7} {'sec':>6}")
        for label, fn in METHODS:
            rng = random.Random(A.seed); res = []; mv = []; t0 = time.time()
            for t in range(A.restarts):
                p0 = Plan(g, seeded_growth(g, rng), A.lam, "tvd")
                if fn is None:
                    p, it = hill_climb(p0); m = it
                elif fn == "hybrid":
                    p, r_, f_ = hybrid(p0, rng, tree="compact", tol=A.tol); m = r_ + f_
                else:
                    p, s_, _ = recom_climb(p0, rng, tree=fn, tol=A.tol); m = s_
                e = verify(p)
                assert not e, f"{label}: {e}"
                r = p.report(); res.append(r); mv.append(m)
            f = [x["score"] for x in res]
            print(f"{label:22} {np.median([x['compactness'] for x in res]):>7.3f} "
                  f"{np.median([x['max_dev_pct'] for x in res]):>12.2f} "
                  f"{np.median([200*x['tvd'] for x in res]):>11.2f} "
                  f"{np.median(f):>8.4f} {max(f):>8.4f} {np.median(mv):>7.0f} "
                  f"{time.time()-t0:>6.1f}")


if __name__ == "__main__":
    main()
