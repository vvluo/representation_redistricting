"""Map of a state's high-poverty clusters, a starter map, and a trained agent's best
valid map with the district(s) that reach the poverty threshold outlined.

    python -m experiments.plot_poverty_district --run outputs/sweeps/s4/AL_congress_ddqn_explore0.03_s1_...
Replays the run's saved Q-network greedily from the held-out starts, picks the first
start whose best map within tolerance gives poverty a district, and draws it.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection, LineCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting.rl.env import RedistrictingEnv
from redistricting.rl.qnet import QNetwork
from redistricting.shapefile import read_dbf, read_shp
from redistricting.states import PRESETS

SURF, INK, INK2, BASE, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e6e1", "#fcfcfb"
HIGH, HIGHLIGHT = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
HALO = [pe.withStroke(linewidth=2.6, foreground=SURF)]


def replay(run, max_starts=16):
    a = json.load(open(run / "args.json")); wj = json.load(open(run / "weights.json"))
    ck = torch.load(run / "model.pt", weights_only=False)
    q = QNetwork(ck["dim"], ck["hidden"]); q.load_state_dict(ck["model"]); q.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = RedistrictingEnv(a["state"], a["level"], tol=a["tol"], explore_tol=a["explore_tol"],
                               weights=wj["raw_weights"], horizon=a["horizon"], pool_size=a["eval_episodes"],
                               pool_seed=a["eval_pool_seed"])
    for i in range(min(max_starts, len(env.pool))):
        obs, info = env.reset(options={"index": i})
        for _ in range(a["horizon"]):
            n = int(obs["mask"].sum())
            if n == 0:
                break
            with torch.no_grad():
                k = int(q(torch.as_tensor(obs["x"][:n])).argmax())
            obs, r, te, tr, info = env.step(k)
            if te or tr:
                break
        if env.best_assignment is None:
            continue
        ev = env.evaluate(env.best_assignment)
        if ev["families"]["poverty"]["groups_cracked"] == 0 and ev["families"]["poverty"]["groups_at_risk"] > 0:
            return env, i, env.start_assignment.copy(), env.best_assignment.copy(), ev
    raise SystemExit("no held-out start where the agent's best valid map gives poverty a district")


def geometry(env):
    fp = PRESETS[env.state_abbr][0]
    d = sorted((ROOT.parent / "TIGER").glob(f"tl_*_{fp}_tract"))[-1]
    base = str(d / d.name)
    gid = [r["GEOID"] for r in read_dbf(base + ".dbf")]
    rings = read_shp(base + ".shp")
    lat0 = float(np.nanmean(env.g.features["lat"]))
    cx = np.cos(np.radians(lat0))
    unit = {g: env.g.idx[g] for g in gid if g in env.g.idx}
    polys, owner, raw = [], [], []
    for g, rr in zip(gid, rings):
        if g not in unit:
            continue
        for b in rr:
            p = np.frombuffer(b, dtype="<f8").reshape(-1, 2).copy(); p[:, 0] *= cx
            polys.append(p); owner.append(unit[g]); raw.append(b)
    return polys, np.array(owner), raw, cx


def outlines(assign, owner, raw, cx, which):
    """Boundary segments of the districts in `which`: segments used once by their tracts."""
    segs = []
    for d in which:
        cnt = Counter()
        for b, u in zip(raw, owner):
            if assign[u] != d:
                continue
            for k in range(0, len(b) - 16, 16):
                s = b[k:k + 32]
                key = min(s[:16], s[16:]) + max(s[:16], s[16:])
                cnt[key] += 1
        for key, c in cnt.items():
            if c == 1:
                p = np.frombuffer(key, dtype="<f8").reshape(2, 2).copy(); p[:, 0] *= cx
                segs.append(p)
    return segs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    A = ap.parse_args(argv)
    run = Path(A.run)
    env, start_i, start, best, ev = replay(run)
    F, g = env.F, env.g
    k = [gr.name for gr in F.groups].index("poverty"); tau = F.groups[k].tau
    pb, pt = g.col("pov_below"), g.col("pov_total")
    share = np.where(pt > 0, pb / np.maximum(pt, 1), 0.0)
    high = (share >= tau) & (g.pop > 0)
    polys, owner, raw, cx = geometry(env)
    ideal = g.pop.sum() / env.p

    def dshare(a):
        P = np.bincount(a, pb, env.p) / np.maximum(np.bincount(a, pt, env.p), 1)
        return P

    # clusters of high tracts
    comp = -np.ones(g.N, int); nc = 0
    for s in np.flatnonzero(high):
        if comp[s] >= 0:
            continue
        comp[s] = nc; st = [s]
        while st:
            x = st.pop()
            for y in g.nbr[x]:
                if high[y] and comp[y] < 0:
                    comp[y] = nc; st.append(y)
        nc += 1
    cpop = np.bincount(comp[comp >= 0], g.pop[comp >= 0], nc)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.4), facecolor=SURF)
    fc_high = np.where(high[owner], HIGH, BASE)
    xy = np.c_[g.xy[:, 0] / 6371.0 / np.radians(1) / 1.0, g.xy[:, 1] / 6371.0 / np.radians(1)]   # degrees, x scaled
    cent_deg = np.c_[g.features["lon"].to_numpy() * cx, g.features["lat"].to_numpy()]

    # panel 1: clusters
    ax = axes[0]
    ax.add_collection(PolyCollection(polys, facecolors=fc_high, edgecolors=EDGE, linewidths=0.1))
    for c in np.argsort(-cpop)[:6]:
        m = comp == c
        w = g.pop[m]; x, y = (cent_deg[m] * w[:, None]).sum(0) / w.sum()
        ax.text(x, y, f"{cpop[c] / 1000:.0f}k\n{cpop[c] / ideal:.0%} of a district", fontsize=7.5, ha="center",
                va="center", color=INK, path_effects=HALO, zorder=5, linespacing=1.05)
    ax.set_title(f"High-poverty tracts (poverty ≥ {tau:.1%})\n{high.sum()} tracts in {nc} separate clusters; "
                 f"largest = {cpop.max() / 1000:.0f}k people", loc="left", fontsize=10, color=INK)

    # panels 2-3: start map and agent's best valid map
    for ax, a, lab in ((axes[1], start, "Starting map"), (axes[2], best, "Agent's best valid map")):
        ds = dshare(a)
        rep = [d for d in range(env.p) if ds[d] >= tau]
        ax.add_collection(PolyCollection(polys, facecolors=fc_high, edgecolors="none", alpha=0.55))
        ax.add_collection(LineCollection(outlines(a, owner, raw, cx, range(env.p)), colors=INK, linewidths=0.7))
        if rep:
            ax.add_collection(LineCollection(outlines(a, owner, raw, cx, rep), colors=HIGHLIGHT, linewidths=2.6))
        for d in range(env.p):
            m = a == d; w = g.pop[m]
            x, y = (cent_deg[m] * w[:, None]).sum(0) / w.sum()
            ax.text(x, y, f"{ds[d]:.1%}", fontsize=8.5, ha="center", va="center",
                    color=INK, path_effects=HALO, zorder=6, fontweight="bold" if d in rep else "normal")
        phi = env.F.phi(env.F.state(a))
        ax.set_title(f"{lab}: district poverty rates\n{len(rep)} district(s) ≥ {tau:.1%} (orange outline); "
                     f"compactness {phi['compactness']:.3f}", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    st = env.state_abbr
    fig.suptitle(f"{st} congressional districts ({env.p}): can poverty get a district?", x=0.01, ha="left",
                 fontsize=13, color=INK)
    wa = json.load(open(run / "args.json"))
    wtxt = ", ".join(f"{k} {v:g}" for k, v in json.loads(wa["weights"]).items() if k != "excess") \
        + (" (per-move std units)" if wa.get("normalized_weights") else "")
    fig.text(0.01, 0.015, f"Blue = tracts with poverty ≥ {tau:.1%} (US tract mean + 1 SD). Agent: DDQN "
             f"({wtxt}; {env.explore_tol:.0%} exploration band), held-out start {start_i}; "
             f"best map within {env.tol:.0%} of ideal population ({ideal:,.0f}). Labels: district poverty rate.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = Path(A.out) if A.out else ROOT / "outputs" / "figures" / f"{st}_poverty_district.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURF); plt.close(fig)
    print(out, "| start", start_i, "| district poverty rates best:", np.round(dshare(best), 3).tolist())


if __name__ == "__main__":
    main()
