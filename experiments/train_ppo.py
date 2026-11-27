"""PPO on the redistricting env, following cleanrl/ppo.py.

Actor and critic (redistricting/rl/ppo.py):
    pi_theta(a | s) = softmax_a  W_pi^T h_theta(x(s, a)) + b_pi     over the legal flips
    V_phi(s)        = MLP_phi(x_state(s))
h_theta is the Q-network's embedding (same class, same 120 -> 84 default), so
--init-embedding can start it from a trained DQN/DDQN model.pt.  x_state is the map's
features that belong to no single flip: districts + family blocks by default; add
--critic-phi (objective levels phi(s)) and --critic-time (t / horizon) to give the
critic more.

Same loop and defaults as CleanRL where they apply (Adam 2.5e-4 annealed to 0, 4 envs
x 128 steps per rollout, GAE gamma 0.99 lambda 0.95, 4 epochs x 4 minibatches, clip
0.2, clipped value loss, entropy 0.01, value 0.5, grad norm 0.5).  Differences, all
forced by the env: the policy is a softmax over a variable number of legal rows (padded
per minibatch and masked); features are standardised and rewards rescaled once from a
short random rollout (as train_dqn); truncation at the horizon bootstraps,
r_T += gamma V(s_T), instead of treating the time limit as termination.

    python -m experiments.train_ppo --state IA --level congress --total-timesteps 100000

Writes outputs/runs/<run>/{args.json, weights.json, progress.csv, episodes.csv,
eval.csv, model.pt}; eval.csv has the same columns as train_dqn's (methods: agent =
argmax pi, agent_sample = a ~ pi, greedy = best one-step reward).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, json, random, sys, time, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting.rl.env import RedistrictingEnv
from redistricting.rl.ppo import Actor, Critic, critic_columns, gae, pad
from experiments.train_dqn import rollout


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = ap.add_argument
    a("--exp-name", default="ppo"); a("--seed", type=int, default=1)
    a("--torch-deterministic", type=int, default=1)
    # environment (as train_dqn)
    a("--state", default="IA"); a("--level", default="congress", choices=["congress", "house"])
    a("--tol", type=float, default=None); a("--explore-tol", type=float, default=None)
    a("--horizon", type=int, default=200); a("--pool-size", type=int, default=32)
    a("--weights", default=None, help='JSON, e.g. \'{"compactness": 1, "poverty": 1}\'; default: env default')
    a("--normalized-weights", action="store_true",
      help="--weights are in units of each term's per-move std (as experiments/weight_grid.py)")
    # algorithm (CleanRL names and defaults)
    a("--total-timesteps", type=int, default=100000)
    a("--learning-rate", type=float, default=2.5e-4)
    a("--num-envs", type=int, default=4)
    a("--num-steps", type=int, default=128, help="steps per env per rollout")
    a("--anneal-lr", type=int, default=1)
    a("--gamma", type=float, default=0.99); a("--gae-lambda", type=float, default=0.95)
    a("--num-minibatches", type=int, default=4); a("--update-epochs", type=int, default=4)
    a("--norm-adv", type=int, default=1)
    a("--clip-coef", type=float, default=0.2); a("--clip-vloss", type=int, default=1)
    a("--ent-coef", type=float, default=0.01); a("--vf-coef", type=float, default=0.5)
    a("--max-grad-norm", type=float, default=0.5)
    a("--target-kl", type=float, default=None)
    # networks
    a("--hidden", default="120,84", help="h_theta, as the Q-network")
    a("--critic-hidden", default="64,64")
    a("--critic-phi", action="store_true", help="critic also sees the objective levels phi(s)")
    a("--critic-time", action="store_true", help="critic also sees t / horizon")
    a("--init-embedding", default=None, help="a DQN/DDQN model.pt: start h_theta (and its feature norm) from it")
    a("--reward-scale", default="auto", help="'auto' = 1 / std of random-rollout rewards, or a number")
    a("--norm-steps", type=int, default=1000)
    a("--eval-episodes", type=int, default=16)
    a("--eval-pool-seed", type=int, default=1000, help="held-out starter maps for evaluation")
    a("--out-dir", default="outputs/runs")
    return ap.parse_args(argv)


def legal(obs):
    return int(obs["mask"].sum())


def fit_scales(env, actor, critic, steps, seed, scale_arg, fit_actor_norm=True):
    """Random rollout: fit both feature standardisations and the reward scale."""
    rng = np.random.default_rng(seed); rows, states, rewards = [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        n = legal(obs)
        if n == 0:
            obs, _ = env.reset(); continue
        rows.append(obs["x"][:n][rng.choice(n, min(n, 16), replace=False)]); states.append(obs["s"])
        obs, r, term, trunc, _ = env.step(int(rng.integers(n))); rewards.append(r)
        if term or trunc:
            obs, _ = env.reset()
    if fit_actor_norm:
        actor.norm.fit(np.vstack(rows))
    critic.norm.fit(critic.select(torch.as_tensor(np.array(states))).numpy())
    sd = float(np.std(rewards))
    return (1.0 / sd if sd > 0 else 1.0) if scale_arg == "auto" else float(scale_arg)


def main(argv=None):
    args = parse(argv)
    tag = "ppo" + (f"_explore{args.explore_tol}" if args.explore_tol else "")
    run = f"{args.state}_{args.level}_{tag}_s{args.seed}_{time.strftime('%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
    out = ROOT / args.out_dir / run; out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    weights = json.loads(args.weights) if args.weights else None
    mk = lambda **kw: RedistrictingEnv(args.state, args.level, tol=args.tol, explore_tol=args.explore_tol,
                                       weights=weights, horizon=args.horizon, **kw)
    envs = [mk(pool_size=args.pool_size) for _ in range(args.num_envs)]
    env = envs[0]
    sd = None
    if args.normalized_weights and weights:
        from experiments.weight_grid import term_sd
        sd = term_sd(env, seed=args.seed)
        weights = {k: v / sd[k] for k, v in weights.items()}
    for e in envs:
        if weights is not None:
            e.weights = dict(weights)
    json.dump({"raw_weights": env.weights, "term_sd": sd}, open(out / "weights.json", "w"), indent=1)

    hidden = tuple(int(h) for h in args.hidden.split(","))
    chidden = tuple(int(h) for h in args.critic_hidden.split(","))
    extra = [b for b, on in (("phi", args.critic_phi), ("time", args.critic_time)) if on]
    cols = critic_columns(env.F.state_blocks, extra)
    actor, critic = Actor(env.F.dim, hidden), Critic(cols, chidden)
    if args.init_embedding:
        actor.load_embedding(torch.load(args.init_embedding, weights_only=False))
    scale = fit_scales(env, actor, critic, args.norm_steps, args.seed, args.reward_scale,
                       fit_actor_norm=not args.init_embedding)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)
    E, T = args.num_envs, args.num_steps
    batch = E * T; mb = batch // args.num_minibatches
    iters = max(1, args.total_timesteps // batch)
    print(f"{run}: {env.state_abbr} {env.level} p={env.p} tol={env.tol} explore={env.explore_tol} "
          f"F={env.F.dim} x_state={len(cols)}/{env.F.state_dim} ({'+'.join(['districts', 'family'] + extra)}) "
          f"reward_scale={scale:.3g} weights={env.weights} | {iters} iterations x {batch} steps", flush=True)

    prog = csv.writer(open(out / "progress.csv", "w"))
    prog.writerow(["step", "lr", "pg_loss", "v_loss", "entropy", "old_approx_kl", "approx_kl", "clipfrac",
                   "explained_var", "mean_return", "sps"])
    eps_log = csv.writer(open(out / "episodes.csv", "w"))
    eps_log.writerow(["step", "return", "length", "J_final", "best_J", "final_within_tol", "cracked_final"])

    obs = [e.reset(seed=args.seed + k)[0] for k, e in enumerate(envs)]
    done = torch.zeros(E); ep_ret = np.zeros(E); ep_len = np.zeros(E, int)
    step = 0; t0 = time.time(); recent = []
    for it in range(1, iters + 1):
        if args.anneal_lr:
            opt.param_groups[0]["lr"] = (1.0 - (it - 1) / iters) * args.learning_rate
        rows = [[None] * E for _ in range(T)]
        S = torch.zeros(T, E, env.F.state_dim); A = torch.zeros(T, E, dtype=torch.long)
        LP = torch.zeros(T, E); R = torch.zeros(T, E); D = torch.zeros(T, E); V = torch.zeros(T, E)
        for t in range(T):
            step += E
            D[t] = done
            for k, e in enumerate(envs):
                n = legal(obs[k]); x = obs[k]["x"][:n].copy(); s = torch.as_tensor(obs[k]["s"])
                rows[t][k] = x; S[t, k] = s
                with torch.no_grad():
                    V[t, k] = critic(s)
                    if n:
                        dist = actor.dist(torch.as_tensor(x), torch.ones(n, dtype=torch.bool))
                        a = dist.sample(); A[t, k] = a; LP[t, k] = dist.log_prob(a)
                nobs, r, term, trunc, info = e.step(int(A[t, k]) if n else 0)
                rr = r * scale
                if trunc and not term:                         # time limit: bootstrap from V(s_T)
                    with torch.no_grad():
                        rr += args.gamma * float(critic(torch.as_tensor(nobs["s"])))
                R[t, k] = rr; ep_ret[k] += r; ep_len[k] += 1
                if n == 0:
                    term = True                               # no legal flip: nothing to learn here
                if term or trunc:
                    eps_log.writerow([step, ep_ret[k], ep_len[k], info["J"], info["best_J"], info["within_tol"],
                                      sum(info["cracked"].values())])
                    recent.append(ep_ret[k]); ep_ret[k] = 0; ep_len[k] = 0
                    nobs, _ = e.reset()
                obs[k] = nobs; done[k] = float(term or trunc)

        with torch.no_grad():
            nv = critic(torch.as_tensor(np.array([o["s"] for o in obs])))
        adv, ret = gae(R, V, D, nv, done, args.gamma, args.gae_lambda)

        # flatten; drop steps with no legal flip (their logprob is undefined)
        flat_rows = [rows[t][k] for t in range(T) for k in range(E)]
        keep = np.array([len(x) > 0 for x in flat_rows])
        idx_all = np.flatnonzero(keep)
        bS, bA, bLP = S.reshape(-1, env.F.state_dim), A.reshape(-1), LP.reshape(-1)
        bAdv, bRet, bV = adv.reshape(-1), ret.reshape(-1), V.reshape(-1)
        rng = np.random.default_rng(args.seed * 100003 + it)
        clipfracs = []
        for epoch in range(args.update_epochs):
            perm = rng.permutation(idx_all)
            for st in range(0, len(perm), mb):
                ix = perm[st:st + mb]
                X, M = pad([flat_rows[i] for i in ix], env.F.dim)
                ix_t = torch.as_tensor(ix)
                dist = actor.dist(X, M)
                newlp = dist.log_prob(bA[ix_t]); ent = dist.entropy()
                newv = critic(bS[ix_t])
                logratio = newlp - bLP[ix_t]; ratio = logratio.exp()
                with torch.no_grad():
                    old_kl = (-logratio).mean(); kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(float(((ratio - 1).abs() > args.clip_coef).float().mean()))
                a_ = bAdv[ix_t]
                if args.norm_adv and len(ix) > 1:
                    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                if args.clip_vloss:
                    vc = bV[ix_t] + (newv - bV[ix_t]).clamp(-args.clip_coef, args.clip_coef)
                    vl = 0.5 * torch.max((newv - bRet[ix_t]) ** 2, (vc - bRet[ix_t]) ** 2).mean()
                else:
                    vl = 0.5 * ((newv - bRet[ix_t]) ** 2).mean()
                loss = pg - args.ent_coef * ent.mean() + args.vf_coef * vl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm); opt.step()
            if args.target_kl is not None and float(kl) > args.target_kl:
                break

        yv, yr = bV[idx_all].numpy(), bRet[idx_all].numpy()
        var = np.var(yr); ev_ = float("nan") if var == 0 else 1 - np.var(yr - yv) / var
        sps = int(step / (time.time() - t0))
        mret = float(np.mean(recent[-20:])) if recent else float("nan")
        prog.writerow([step, opt.param_groups[0]["lr"], float(pg), float(vl), float(ent.mean()), float(old_kl),
                       float(kl), float(np.mean(clipfracs)), ev_, mret, sps])
        if it % max(1, iters // 10) == 0 or it == iters:
            print(f"step {step}  ret {mret:.4g}  pg {float(pg):.3g}  v {float(vl):.3g}  H {float(ent.mean()):.3g}  "
                  f"kl {float(kl):.3g}  clip {np.mean(clipfracs):.2f}  EV {ev_:.2f}  SPS {sps}", flush=True)

    torch.save({"model": actor.state_dict(), "critic": critic.state_dict(), "args": vars(args), "dim": env.F.dim,
                "names": env.F.names, "hidden": hidden, "critic_hidden": chidden, "critic_cols": cols,
                "state_names": env.F.state_names, "reward_scale": scale, "algo": "ppo"}, out / "model.pt")

    # evaluation on held-out starts, as train_dqn: argmax pi, a ~ pi, and greedy one-step reward
    ev = mk(pool_size=args.eval_episodes, pool_seed=args.eval_pool_seed, max_actions=env.max_actions)
    ev.weights = dict(env.weights)
    overlap = len({a.tobytes() for a in ev.pool} & {a.tobytes() for a in env.pool})
    erng = torch.Generator().manual_seed(args.seed)

    def agent(o):
        with torch.no_grad():
            return int(actor(torch.as_tensor(o["x"][:legal(o)])).argmax())

    def agent_sample(o):
        with torch.no_grad():
            p = torch.softmax(actor(torch.as_tensor(o["x"][:legal(o)])), 0)
            return int(torch.multinomial(p, 1, generator=erng))

    def greedy(o):
        return int(np.argmax(ev.cand_reward))
    out_rows = []
    for i in range(len(ev.pool)):
        for name, pick in (("agent", agent), ("agent_sample", agent_sample), ("greedy", greedy)):
            out_rows.append({"start": i, "method": name, **rollout(ev, pick, i, args.horizon)})
    print(f"eval on {len(ev.pool)} held-out starts ({overlap} also in the training pool)")
    with open(out / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    import pandas as pd
    Ev = pd.DataFrame(out_rows)
    Ev["dJ_final"] = Ev.J_final - Ev.J_start; Ev["dJ_best"] = Ev.J_best - Ev.J_start
    print(Ev.groupby("method")[["dJ_final", "dJ_best", "final_within_tol", "cracked_start",
                                "cracked_final", "cracked_best"]].mean().round(4).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    return out


if __name__ == "__main__":
    main()
