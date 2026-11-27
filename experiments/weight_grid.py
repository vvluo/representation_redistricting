"""Weight grid over two protected families, compactness always included.

Reward  r = (1/sd_c) d_compactness + (a/sd_A) d_A + (b/sd_B) d_B
  d_A, d_B   the quota terms (d_<family>) or the cohesion terms (d_coh_<family>)
  sd_*       std of each term's per-move change over random rollouts, so a weight of 1
             means "a typical move's change counts as much as compactness's"
For every (a, b) on the grid, greedy best-improvement runs from held-out starter maps;
the best map within tolerance is scored on what the pair is meant to protect: groups
cracked and district shortfall per family, cohesion per family, compactness, validity.

    python -m experiments.weight_grid --state AL --pair poverty ethnicity --kind quota cohesion
Writes outputs/weight_grid/<state>_<level>_<A>_<B>.csv and a heatmap PNG.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, itertools, sys, time, warnings
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def term_sd(env, steps=1500, seed=0):
    """Std of every dphi term over legal moves visited by a random walk."""
    rng = np.random.default_rng(seed); X = []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        n = int(obs["mask"].sum())
        if n == 0:
            obs, _ = env.reset(); continue
        X.append(env.cand_dphi)
        obs, r, te, tr, _ = env.step(int(rng.integers(n)))
        if te or tr:
            obs, _ = env.reset()
    X = np.vstack(X); names = [n[2:] for n in env.F.names[env.F.blocks["dphi"]]]
    sd = X.std(0)
    return {n: float(s) if s > 0 else 1.0 for n, s in zip(names, sd)}


_ENV = {}


def _env(cfg):
    from redistricting.rl.env import RedistrictingEnv
    key = (cfg["state"], cfg["level"], cfg["pool_seed"])
    if key not in _ENV:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ENV[key] = RedistrictingEnv(cfg["state"], cfg["level"], horizon=cfg["horizon"],
                                         pool_size=cfg["starts"], pool_seed=cfg["pool_seed"],
                                         explore_tol=cfg["explore_tol"])
    return _ENV[key]


def _run(task):
    cfg, kind, a, b, w = task
    env = _env(cfg); env.weights = w
    A, B = cfg["pair"]; rows = []
    for i in range(len(env.pool)):
        obs, info = env.reset(options={"index": i})
        for _ in range(cfg["horizon"]):
            if not len(env.cand_reward):
                break
            k = int(np.argmax(env.cand_reward))
            if env.cand_reward[k] <= 1e-12 and env.explore_tol == env.tol:
                break                                  # greedy local optimum
            obs, r, te, tr, info = env.step(k)
            if te or tr:
                break
        s = env.evaluate(env.start_assignment)
        e = env.evaluate(env.best_assignment if env.best_assignment is not None else env.st.a)
        row = dict(kind=kind, a=a, b=b, start=i, within_tol=e["within_tol"], contiguous=e["contiguous"],
                   compactness=e["phi"]["compactness"], compactness_start=s["phi"]["compactness"])
        for f in (A, B):
            row[f"{f}_at_risk"] = e["families"][f]["groups_at_risk"]
            row[f"{f}_cracked_start"] = s["families"][f]["groups_cracked"]
            row[f"{f}_cracked"] = e["families"][f]["groups_cracked"]
            row[f"{f}_shortfall"] = e["families"][f]["district_shortfall"]
            row[f"{f}_coh"] = e["phi"][f"coh_{f}"]; row[f"{f}_coh_start"] = s["phi"][f"coh_{f}"]
            row[f"{f}_quota"] = e["phi"][f]
        rows.append(row)
    return rows


def heatmap(D, A, B, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    kinds = list(D.kind.unique())
    metrics = [(f"{A}_cracked", f"{A}: groups cracked"), (f"{B}_cracked", f"{B}: groups cracked"),
               (f"{A}_coh", f"{A}: cohesion"), (f"{B}_coh", f"{B}: cohesion"), ("compactness", "compactness")]
    fig, axes = plt.subplots(len(kinds), len(metrics), figsize=(3.3 * len(metrics), 3.1 * len(kinds) + 0.8),
                             facecolor=SURF, squeeze=False)
    M = D.groupby(["kind", "a", "b"]).mean(numeric_only=True).reset_index()
    for r, kind in enumerate(kinds):
        for c, (m, lab) in enumerate(metrics):
            ax = axes[r, c]; S = M[M.kind == kind].pivot(index="b", columns="a", values=m)
            lo, hi = M[m].min(), M[m].max()
            ax.imshow(S.values, origin="lower", cmap=blues if "cracked" not in m else blues.reversed(),
                      vmin=lo, vmax=hi if hi > lo else lo + 1e-9, aspect="auto")
            for (y, x), v in np.ndenumerate(S.values):
                ax.text(x, y, f"{v:.2f}" if "cracked" in m else f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color=INK if (v - lo) / ((hi - lo) or 1) < 0.55 else SURF)
            ax.set_xticks(range(S.shape[1]), [f"{x:g}" for x in S.columns], fontsize=7)
            ax.set_yticks(range(S.shape[0]), [f"{y:g}" for y in S.index], fontsize=7)
            ax.set_xlabel(f"weight on {A}", fontsize=8, color=INK2); ax.set_ylabel(f"weight on {B}", fontsize=8, color=INK2)
            ax.set_title(f"{kind} · {lab}", fontsize=9, color=INK, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01, "Greedy from held-out starts; best map within tolerance; means over starts. "
             "Cracked: darker = fewer. Compactness weight fixed at 1; weights in units of each term's per-move std.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94)); fig.savefig(path, dpi=140, facecolor=SURF); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", default="AL"); ap.add_argument("--level", default="congress")
    ap.add_argument("--pair", nargs=2, default=["poverty", "ethnicity"])
    ap.add_argument("--kind", nargs="+", default=["quota", "cohesion"], choices=["quota", "cohesion"])
    ap.add_argument("--grid", default="0,0.5,1,2,4,8")
    ap.add_argument("--starts", type=int, default=16); ap.add_argument("--pool-seed", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=200); ap.add_argument("--explore-tol", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=10); ap.add_argument("--out-dir", default="outputs/weight_grid")
    A_ = ap.parse_args(argv)
    cfg = dict(state=A_.state, level=A_.level, pair=A_.pair, starts=A_.starts, pool_seed=A_.pool_seed,
               horizon=A_.horizon, explore_tol=A_.explore_tol)
    env = _env(cfg)
    for f in A_.pair:
        if f not in env.F.families:
            raise SystemExit(f"{f} has no data in {A_.state}; families: {env.F.families}")
    sd = term_sd(env)
    grid = [float(x) for x in A_.grid.split(",")]
    tasks = []
    for kind in A_.kind:
        t = (lambda f: f) if kind == "quota" else (lambda f: f"coh_{f}")
        for a, b in itertools.product(grid, grid):
            w = {"compactness": 1.0 / sd["compactness"], t(A_.pair[0]): a / sd[t(A_.pair[0])],
                 t(A_.pair[1]): b / sd[t(A_.pair[1])]}
            if env.explore_tol > env.tol:
                w["excess"] = 10.0 / sd["excess"]
            tasks.append((cfg, kind, a, b, w))
    t0 = time.time(); rows = []
    with get_context("spawn").Pool(A_.jobs) as pool:
        for r in pool.imap_unordered(_run, tasks):
            rows += r
    D = pd.DataFrame(rows)
    out = ROOT / A_.out_dir; out.mkdir(parents=True, exist_ok=True)
    stem = f"{A_.state}_{A_.level}_{A_.pair[0]}_{A_.pair[1]}"
    D.to_csv(out / f"{stem}.csv", index=False)
    risk = {f: int(D[f"{f}_at_risk"].iloc[0]) for f in A_.pair}
    heatmap(D, *A_.pair, out / f"{stem}.png",
            f"{A_.state} {A_.level}: weights on {A_.pair[0]} and {A_.pair[1]} "
            f"(groups that could get a district: {risk})")
    M = D.groupby(["kind", "a", "b"]).mean(numeric_only=True)
    cols = [f"{f}_cracked" for f in A_.pair] + [f"{f}_coh" for f in A_.pair] + ["compactness", "within_tol"]
    print(f"sd per move: " + ", ".join(f"{k} {sd[k]:.2e}" for k in ["compactness"] +
          [f for f in A_.pair] + [f"coh_{f}" for f in A_.pair]))
    print(f"start means: " + ", ".join(f"{f} cracked {D[f'{f}_cracked_start'].mean():.2f}, coh {D[f'{f}_coh_start'].mean():.3f}"
                                      for f in A_.pair) + f", compactness {D.compactness_start.mean():.3f}")
    print(M[cols].round(3).to_string())
    print(f"{len(tasks)} grid points in {time.time() - t0:.0f}s -> {out / stem}.csv/.png")


if __name__ == "__main__":
    main()
