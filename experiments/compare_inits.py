"""Compare the five initialisation strategies under the steepest-ascent flip climb.

    python -m experiments.compare_inits --unit AL_county --restarts 20

Every init hands the climber a valid plan: D non-empty contiguous districts.  Random
and population-greedy growth are contiguous by construction; sweep, k-means and
spectral bisection pass through repair().  Writes the per-run table and the best plan
to --out-dir.
"""
import argparse, random
from pathlib import Path
import numpy as np, pandas as pd
from redistricting import Geo, Plan, verify, INITS, make_init, hill_climb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unit", default="AL_county")
    ap.add_argument("--districts", type=int, default=7)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--lam", type=float, default=2.0,
                    help="weight on the population term relative to compactness")
    ap.add_argument("--uniformity", choices=["tvd", "rmsd", "entropy"], default="tvd")
    ap.add_argument("--seed", type=int, default=6792)
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir", default="outputs")
    A = ap.parse_args(argv)

    g = Geo.load(A.unit, D=A.districts, data_dir=A.data_dir)
    print(f"N={g.N}  D={g.D}  ideal={g.ideal:,.0f}  "
          f"objective = mean PP - {A.lam} * {A.uniformity}")
    print(f"{A.restarts} restarts per initialisation\n")

    rows, best = [], {}
    for nm in INITS:
        rng = random.Random(A.seed)
        for t in range(A.restarts):
            a0 = make_init(nm, g, rng)
            p0 = Plan(g, a0, A.lam, A.uniformity)
            e0 = verify(p0)
            assert not e0, f"{nm} produced an invalid start: {e0}"
            r0 = p0.report()
            p, it = hill_climb(Plan(g, a0, A.lam, A.uniformity))
            r1 = p.report()
            assert not verify(p), verify(p)
            rows.append(dict(init=nm, restart=t, flips=it,
                             score0=r0["score"], dev0=r0["max_dev_pct"],
                             pp0=r0["compactness"],
                             score=r1["score"], compactness=r1["compactness"],
                             max_dev_pct=r1["max_dev_pct"],
                             range_pct=r1["range_pct"], tvd=r1["tvd"],
                             entropy=r1["entropy"], min_pp=r1["min_pp"]))
            if nm not in best or r1["score"] > best[nm][1]["score"]:
                best[nm] = (p, r1)

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)

    print("=== START QUALITY (before any flip) ===")
    s = R.groupby("init").agg(score0=("score0", "median"),
                              max_dev0=("dev0", "median"),
                              compact0=("pp0", "median")).round(4)
    print(s.sort_values("score0", ascending=False).to_string())

    print("\n=== AFTER HILL CLIMB (median over restarts) ===")
    m = R.groupby("init").agg(score_med=("score", "median"),
                              score_best=("score", "max"),
                              score_worst=("score", "min"),
                              compact=("compactness", "median"),
                              max_dev=("max_dev_pct", "median"),
                              dev_best=("max_dev_pct", "min"),
                              tvd=("tvd", "median"),
                              range_pct=("range_pct", "median"),
                              flips=("flips", "median")).round(4)
    print(m.sort_values("score_med", ascending=False).to_string())

    print("\n=== SPREAD OF FINAL SCORES (how trapped is each basin?) ===")
    for nm in INITS:
        v = R[R.init == nm].score
        print(f"  {nm:11s} n={len(v)}  mean {v.mean():.4f}  sd {v.std():.4f}  "
              f"[{v.min():.4f}, {v.max():.4f}]  distinct optima ~{v.round(6).nunique()}")

    print("\n=== BEST PLAN FOUND BY EACH INIT ===")
    print(f"  {'init':11s} {'score':>8} {'meanPP':>7} {'worstPP':>8} {'maxdev%':>8} "
          f"{'range%':>7} {'entropy':>12}")
    for nm, (p, r) in sorted(best.items(), key=lambda kv: -kv[1][1]["score"]):
        print(f"  {nm:11s} {r['score']:>8.4f} {r['compactness']:>7.4f} "
              f"{r['min_pp']:>8.4f} {r['max_dev_pct']:>8.2f} {r['range_pct']:>7.2f} "
              f"{r['entropy']:>12.9f}")

    gw = max(best.items(), key=lambda kv: kv[1][1]["score"])
    p, r = gw[1]
    print(f"\n=== OVERALL BEST: {gw[0]} (score {r['score']:.4f}) ===")
    print(f"  {'dist':>4} {'n':>4} {'population':>11} {'dev%':>7} {'perim km':>9} {'PP':>6}")
    for d in range(g.D):
        print(f"  {d:>4} {len(p.members[d]):>4} {p.pop[d]:>11,.0f} "
              f"{100*(p.pop[d]-g.ideal)/g.ideal:>7.2f} {p.perim[d]:>9,.0f} {p.pp()[d]:>6.3f}")

    out = Path(A.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"geoid": g.gid, "name": g.name, "district": p.a,
                  "pop": g.pop}).to_csv(out / f"{A.unit}_best_by_init.csv", index=False)
    R.to_csv(out / f"{A.unit}_init_comparison.csv", index=False)
    print(f"\n  wrote {out}/{A.unit}_best_by_init.csv and {out}/{A.unit}_init_comparison.csv")


if __name__ == "__main__":
    main()
