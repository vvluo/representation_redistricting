"""DQN / Double DQN on the redistricting env, following cleanrl/dqn.py.

Same loop and defaults as CleanRL where they apply (Adam 2.5e-4, MSE TD loss,
gamma 0.99, batch 128, linear epsilon 1 -> 0.05 over half the run, train every 10
steps after learning_starts, target update every 500 steps with tau = 1).  Differences,
all forced by the action space: the network scores each legal flip's feature row,
Q(s, a) = w_Q^T h(x(s, a)) + b_Q, so the greedy action is an argmax over rows and the
replay buffer stores the next state's legal rows.  Features are standardised and
rewards rescaled once, from a short random rollout.

  --algo dqn                  y = r + g max Q_online(s', .)                (no target net)
  --algo dqn --target-network y = r + g max Q_target(s', .)                (cleanrl/dqn.py)
  --algo ddqn                 y = r + g Q_target(s', argmax Q_online(s', .))

    python -m experiments.train_dqn --state IA --level congress --algo ddqn \\
        --total-timesteps 50000 --horizon 200

Writes outputs/runs/<run>/{progress.csv, episodes.csv, eval.csv, model.pt}.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # small nets, many small arrays: threads only contend
import argparse, copy, csv, json, random, sys, time, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting.rl.env import RedistrictingEnv
from redistricting.rl.qnet import QNetwork, ReplayBuffer, td_target, linear_schedule, soft_update


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = ap.add_argument
    a("--exp-name", default="dqn"); a("--seed", type=int, default=1)
    a("--torch-deterministic", type=int, default=1)
    # environment
    a("--state", default="IA"); a("--level", default="congress", choices=["congress", "house"])
    a("--tol", type=float, default=None); a("--explore-tol", type=float, default=None)
    a("--horizon", type=int, default=200); a("--pool-size", type=int, default=32)
    a("--weights", default=None, help='JSON, e.g. \'{"compactness": 1, "poverty": 1}\'; default: env default')
    a("--normalized-weights", action="store_true",
      help="--weights are in units of each term's per-move std (as experiments/weight_grid.py)")
    # algorithm (CleanRL names and defaults)
    a("--algo", default="ddqn", choices=["dqn", "ddqn"])
    a("--target-network", action="store_true", help="dqn only: bootstrap from a delayed target net (as cleanrl/dqn.py)")
    a("--total-timesteps", type=int, default=50000)
    a("--learning-rate", type=float, default=2.5e-4)
    a("--buffer-size", type=int, default=10000)
    a("--gamma", type=float, default=0.99)
    a("--tau", type=float, default=1.0)
    a("--target-network-frequency", type=int, default=500)
    a("--batch-size", type=int, default=128)
    a("--start-e", type=float, default=1.0); a("--end-e", type=float, default=0.05)
    a("--exploration-fraction", type=float, default=0.5)
    a("--learning-starts", type=int, default=2000)
    a("--train-frequency", type=int, default=10)
    a("--hidden", default="120,84")
    a("--loss", default="mse", choices=["mse", "huber"], help="mse = CleanRL; huber = smooth L1")
    a("--reward-scale", default="auto", help="'auto' = 1 / std of random-rollout rewards, or a number")
    a("--norm-steps", type=int, default=1000)
    a("--eval-episodes", type=int, default=16)
    a("--eval-pool-seed", type=int, default=1000, help="held-out starter maps for evaluation")
    a("--out-dir", default="outputs/runs")
    return ap.parse_args(argv)


def legal(obs):
    return int(obs["mask"].sum())


def fit_scales(env, q, steps, seed, scale_arg):
    """Random rollout: fit feature standardisation and the reward scale."""
    rng = np.random.default_rng(seed); rows, rewards = [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        n = legal(obs)
        if n == 0:
            obs, _ = env.reset(); continue
        rows.append(obs["x"][:n][rng.choice(n, min(n, 16), replace=False)])
        obs, r, term, trunc, _ = env.step(int(rng.integers(n))); rewards.append(r)
        if term or trunc:
            obs, _ = env.reset()
    q.norm.fit(np.vstack(rows))
    sd = float(np.std(rewards))
    return (1.0 / sd if sd > 0 else 1.0) if scale_arg == "auto" else float(scale_arg)


def act(q, obs, eps, rng):
    n = legal(obs)
    if n == 0:
        return 0
    if rng.random() < eps:
        return int(rng.integers(n))
    with torch.no_grad():
        return int(q(torch.as_tensor(obs["x"][:n])).argmax())


def rollout(env, pick, index, horizon):
    """Run one episode from pool map `index` with action rule pick(obs) -> row."""
    obs, info = env.reset(options={"index": index}); J0 = info["J"]
    for _ in range(horizon):
        if legal(obs) == 0:
            break
        obs, r, term, trunc, info = env.step(pick(obs))
        if term or trunc:
            break
    fin = env.evaluate(env.st.a)
    best = env.evaluate(env.best_assignment) if env.best_assignment is not None else None
    cr = lambda ev: sum(v["groups_cracked"] for v in ev["families"].values()) if ev else None
    return {"J_start": J0, "J_final": fin["J"], "J_best": best["J"] if best else None,
            "final_within_tol": fin["within_tol"], "cracked_start": cr(env.evaluate(env.start_assignment)),
            "cracked_final": cr(fin), "cracked_best": cr(best),
            **{f"cracked_best_{f}": v["groups_cracked"] for f, v in (best or fin)["families"].items()},
            "best_min_pp": (best or fin)["min_pp"], "start_min_pp": env.evaluate(env.start_assignment)["min_pp"],
            **{f"best_rep_pp_{f}": v for f, v in (best or fin)["rep_pp"].items()},
            **{f"cracked_start_{f}": v["groups_cracked"] for f, v in env.evaluate(env.start_assignment)["families"].items()},
            **{f"start_{k}": v for k, v in env.evaluate(env.start_assignment)["phi"].items()},
            **{f"best_{k}": v for k, v in (best or fin)["phi"].items()}}


def main(argv=None):
    args = parse(argv)
    tag = f"{args.algo}{'_tgt' if args.algo == 'dqn' and args.target_network else ''}" + \
          (f"_explore{args.explore_tol}" if args.explore_tol else "")
    run = f"{args.state}_{args.level}_{tag}_s{args.seed}_{time.strftime('%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
    out = ROOT / args.out_dir / run; out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)
    rng = np.random.default_rng(args.seed)

    weights = json.loads(args.weights) if args.weights else None
    env = RedistrictingEnv(args.state, args.level, tol=args.tol, explore_tol=args.explore_tol,
                           weights=weights, horizon=args.horizon, pool_size=args.pool_size)
    if args.normalized_weights and weights:
        from experiments.weight_grid import term_sd
        sd = term_sd(env, seed=args.seed)
        weights = {k: v / sd[k] for k, v in weights.items()}
        env.weights = weights
        json.dump({"raw_weights": weights, "term_sd": sd}, open(out / "weights.json", "w"), indent=1)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    q = QNetwork(env.F.dim, hidden)
    scale = fit_scales(env, q, args.norm_steps, args.seed, args.reward_scale)
    use_target = args.algo == "ddqn" or args.target_network
    target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size, env.F.dim, seed=args.seed)
    print(f"{run}: {env.state_abbr} {env.level} p={env.p} tol={env.tol} explore={env.explore_tol} "
          f"F={env.F.dim} max_actions={env.max_actions} reward_scale={scale:.3g} weights={env.weights}", flush=True)

    prog = csv.writer(open(out / "progress.csv", "w")); prog.writerow(["step", "td_loss", "q_values", "epsilon", "sps"])
    eps_log = csv.writer(open(out / "episodes.csv", "w"))
    eps_log.writerow(["step", "return", "length", "J_final", "best_J", "final_within_tol", "cracked_final"])
    obs, info = env.reset(seed=args.seed); ep_ret, ep_len = 0.0, 0
    t0 = time.time()
    for step in range(args.total_timesteps):
        eps = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        a = act(q, obs, eps, rng)
        x_sa = obs["x"][a].copy()
        nobs, r, term, trunc, info = env.step(a)
        rb.add(x_sa, r * scale, term, nobs["x"][:legal(nobs)])
        ep_ret += r; ep_len += 1
        obs = nobs
        if term or trunc:
            eps_log.writerow([step, ep_ret, ep_len, info["J"], info["best_J"], info["within_tol"],
                              sum(info["cracked"].values())])
            obs, info = env.reset(); ep_ret, ep_len = 0.0, 0

        if step > args.learning_starts:
            if step % args.train_frequency == 0:
                x, rr, d, X2, M2 = rb.sample(args.batch_size)
                y = td_target(rr, d, X2, M2, q, target if use_target else q, args.gamma,
                              algo=args.algo, use_target=use_target)
                old = q(x)
                loss = F.mse_loss(y, old) if args.loss == "mse" else F.smooth_l1_loss(old, y)
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 100 == 0:
                    sps = int(step / (time.time() - t0))
                    prog.writerow([step, float(loss), float(old.mean()), eps, sps])
                    if step % 5000 == 0:
                        print(f"step {step}  td_loss {float(loss):.4g}  q {float(old.mean()):.3g}  eps {eps:.2f}  SPS {sps}", flush=True)
            if use_target and step % args.target_network_frequency == 0:
                soft_update(target, q, args.tau)

    torch.save({"model": q.state_dict(), "args": vars(args), "dim": env.F.dim, "names": env.F.names,
                "hidden": hidden, "reward_scale": scale}, out / "model.pt")

    # evaluation: learned greedy policy vs greedy best-improvement, same starts and horizon
    def policy(o):
        return act(q, o, 0.0, rng)
    # held-out starts: a pool built with a different seed than the training pool
    ev = RedistrictingEnv(args.state, args.level, tol=args.tol, explore_tol=args.explore_tol,
                          weights=weights, horizon=args.horizon, pool_size=args.eval_episodes,
                          pool_seed=args.eval_pool_seed, max_actions=env.max_actions)
    overlap = len({a.tobytes() for a in ev.pool} & {a.tobytes() for a in env.pool})
    def greedy(o):
        return int(np.argmax(ev.cand_reward))
    rows = []
    for i in range(len(ev.pool)):
        for name, pick in (("agent", policy), ("greedy", greedy)):
            rows.append({"start": i, "method": name, **rollout(ev, pick, i, args.horizon)})
    print(f"eval on {len(ev.pool)} held-out starts ({overlap} also in the training pool)")
    with open(out / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    E = pd.DataFrame(rows)
    E["dJ_final"] = E.J_final - E.J_start; E["dJ_best"] = E.J_best - E.J_start
    print(E.groupby("method")[["dJ_final", "dJ_best", "final_within_tol", "cracked_start",
                               "cracked_final", "cracked_best"]].mean().round(4).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    return out


if __name__ == "__main__":
    main()
