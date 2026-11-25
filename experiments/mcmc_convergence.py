"""Does the MCMC distribution converge to the ZDD's?  KS distance vs chain length.

Uses the cached uniform reference from experiments.mcmc_benchmark (run that first).
Four flip chains per walk tolerance start from extreme reference plans (least / most
compact, lowest / highest D_race).  At each checkpoint, the pooled states so far
(first 20% dropped) are compared with the reference: KS D on all states, and the
median KS p over thinning offsets at stride 3 x autocorrelation time.

    python -m experiments.mcmc_convergence --bench ia4 --tol 0.05 --walk-tol 0.10 0.20 \\
        --steps 50000000 --thin 500 --checkpoints 5e6 1e7 2e7 5e7
"""
import argparse, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcmc_benchmark import load
from redistricting import stats


def _chain(args):
    bench, start, walk_tol, tol, steps, thin, seed = args
    sys.path.insert(0, str(ROOT))
    from redistricting.mcmc import FlipChain
    g, p = load(bench)
    ch = FlipChain(g, start, p, walk_tol, np.random.default_rng(seed))
    ideal = g.pop.sum() / p; out = []; t0 = time.time()
    for t in range(1, steps + 1):
        ch.step()
        if t % thin == 0:
            ok = max(abs(x / ideal - 1) for x in ch.dpop) <= tol
            out.append((t, ok, list(ch.a)))
    return walk_tol, seed, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", default="ia4")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--walk-tol", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--steps", type=int, default=50_000_000)
    ap.add_argument("--thin", type=int, default=500)
    ap.add_argument("--checkpoints", type=float, nargs="+", default=[5e6, 1e7, 2e7, 5e7])
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out-dir", default="outputs")
    A = ap.parse_args()
    from redistricting.mcmc import autocorr_time

    g, p = load(A.bench)
    R = np.load(Path(A.out_dir) / f"ref_{A.bench}_tol{A.tol}.npz")["L"]
    RS = stats.all_stats(R, g, p)
    RS["vi"] = stats.pairwise_vi(R, g.pop, p, 5000, np.random.default_rng(0))
    starts = [R[RS["mean_pp"].argmin()], R[RS["mean_pp"].argmax()],
              R[RS["D_race"].argmin()], R[RS["D_race"].argmax()]]
    tasks = [(A.bench, s, wt, A.tol, A.steps, A.thin, A.seed + 10 * i + k)
             for i, wt in enumerate(A.walk_tol) for k, s in enumerate(starts)]
    with get_context("spawn").Pool(len(tasks)) as pool:
        res = pool.map(_chain, tasks)

    rows = []; dump = {f"ref_{k}": v for k, v in RS.items()}
    for wt in A.walk_tol:
        runs = [r for r in res if r[0] == wt]
        for cp in A.checkpoints:
            chains = []
            for _, _, out, _ in runs:
                kept = [a for (t, ok, a) in out if t <= cp and ok]
                chains.append(np.array(kept[int(0.2 * len(kept)):], dtype=np.int8))
            X = np.concatenate([c for c in chains if len(c)])
            CS = stats.all_stats(X, g, p)
            CS["vi"] = stats.pairwise_vi(X, g.pop, p, 5000, np.random.default_rng(1))
            if cp == max(A.checkpoints):                  # keep per-plan values for plots
                for k, v in CS.items():
                    dump[f"walk{wt}_{k}"] = v
            for k in RS:
                x = CS[k]; D = ks_2samp(RS[k], x).statistic
                if k == "vi":
                    pm = ks_2samp(RS[k], x).pvalue
                else:
                    s = max(1, int(round(3 * autocorr_time(x))))
                    pm = float(np.median([ks_2samp(RS[k], x[o::s]).pvalue for o in range(s)]))
                rows.append({"walk_tol": wt, "steps_per_chain": int(cp), "stat": k,
                             "n_states": len(x), "KS_D": D, "KS_p_median": pm,
                             "dmean": x.mean() - RS[k].mean()})
        print(f"walk_tol {wt}: CPU-s per chain {np.mean([r[3] for r in runs]):.0f}", flush=True)
    D = pd.DataFrame(rows)
    tag = f"{A.bench}_tol{A.tol}_walk{'-'.join(str(w) for w in A.walk_tol)}"
    D.to_csv(Path(A.out_dir) / f"mcmc_convergence_{tag}.csv", index=False)
    np.savez_compressed(Path(A.out_dir) / f"mcmc_convergence_{tag}_stats.npz", **dump)
    piv = D.pivot_table(index=["walk_tol", "steps_per_chain"], columns="stat", values="KS_D")
    with pd.option_context("display.width", 250, "display.float_format", lambda v: f"{v:.3f}"):
        print("KS D by checkpoint:"); print(piv)
        print("median KS p by checkpoint:")
        print(D.pivot_table(index=["walk_tol", "steps_per_chain"], columns="stat", values="KS_p_median"))


if __name__ == "__main__":
    main()
