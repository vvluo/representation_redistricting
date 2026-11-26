"""Gymnasium environment: single-tract flips on a fixed state graph.

    env = RedistrictingEnv("IA", "congress", weights={"poverty": 1, "industry": 1})
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(action)

State       an assignment of every tract to one of p districts (p, tolerance and data
            from redistricting.states: any state, congress or house).
Actions     the LEGAL flips (i, t) of the current state, listed in obs["x"] (one row of
            features x(s, a) each, see features.py) and padded to max_actions;
            obs["mask"][k] = 1 for real rows.  The action is the row index.  Legal =
            source district stays non-empty and connected (hard contiguity) and no
            district moves outside the exploration band `explore_tol`, or further
            outside it.  explore_tol defaults to tol (every visited map valid); set it
            wider (e.g. 0.05 for congress) to let intermediate maps leave tol, with the
            `excess` reward term (default weight 1 when explore_tol > tol) pulling them
            back.  `tol` remains the requirement for a valid final / best map.
            An index on a padding row is a no-op with reward 0 and info["invalid"].
Reward      r(s, a) = sum_k w_k dphi_k(s, a) -- the change in J(s) = sum_k w_k phi_k(s),
            so an episode's return is J(final) - J(start).
Episode     reset draws a starter map from a pool (pool.py), or takes options["start"].
            Ends when no legal flip remains (terminated) or after `horizon` flips
            (truncated).  info tracks the best J among maps within tol and that map
            ("best_assignment"), so evaluation reports both the final and the best
            valid map.
Metrics     every step: J, max deviation, and per family the number of groups cracked
            (attainable N_g >= 1 but no district at tau); at episode end the full
            per-group table for the final and the best map.
"""
import hashlib
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .. import config
from ..states import load_state
from .features import FlipFeatures
from .pool import build_pool


def default_weights(families, explore=False):
    w = {"compactness": 1.0, **{f: 1.0 for f in families}}
    if explore:
        w["excess"] = 1.0
    return w


class RedistrictingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, state="IA", level="congress", p=None, tol=None, weights=None,
                 explore_tol=None, horizon=None, pool=None, pool_size=32, pool_seed=0, strict_pool=True,
                 max_actions=None, families=None, alpha=0.25, tau=None, data_dir=None,
                 pool_dir=None):
        super().__init__()
        self.g, self.p, self.tol = load_state(state, level, p=p, tol=tol, data_dir=data_dir)
        self.state_abbr, self.level = state.upper(), level
        self.F = FlipFeatures(self.g, self.p, tol=self.tol, alpha=alpha, tau=tau, families=families,
                              explore_tol=explore_tol)
        self.explore_tol = self.F.explore_tol
        self.weights = dict(weights) if weights is not None else \
            default_weights(self.F.families, explore=self.explore_tol > self.tol)
        unknown = set(self.weights) - {n[2:] for n in self.F.names[self.F.blocks["dphi"]]}
        if unknown:
            raise ValueError(f"unknown reward terms {sorted(unknown)}")
        self.horizon = horizon or self.g.N
        self.pool = np.asarray(pool) if pool is not None else self._pool(pool_size, pool_seed, strict_pool, pool_dir)

        if max_actions is None:                      # headroom over the pool's legal-move counts
            counts = [int(self.F.actions(self.F.state(a))[1].sum()) for a in self.pool[:8]]
            max_actions = min(len(self.F.du), max(64, int(2 * max(counts))))
        self.max_actions = int(max_actions)
        self.observation_space = spaces.Dict({
            "x": spaces.Box(-np.inf, np.inf, (self.max_actions, self.F.dim), np.float32),
            "mask": spaces.MultiBinary(self.max_actions)})
        self.action_space = spaces.Discrete(self.max_actions)
        self.st = None

    # ---------------------------------------------------------------- pool
    def _pool(self, n, seed, strict, pool_dir):
        d = Path(pool_dir) if pool_dir else config.REPO_ROOT / "outputs" / "pools"
        key = hashlib.md5(("".join(self.g.gid) + str(len(self.g.border))).encode()).hexdigest()[:8]
        f = d / f"{self.state_abbr}_{self.level}_p{self.p}_tol{self.tol}_n{n}_s{seed}_{'strict' if strict else 'loose'}_{key}.npz"
        if f.exists():
            return np.load(f)["pool"]
        P, names = build_pool(self.F, n=n, seed=seed, strict=strict)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, pool=P, strategy=np.array(names))
        return P

    # ---------------------------------------------------------------- core
    def J(self, phi):
        return float(sum(w * phi[k] for k, w in self.weights.items()))

    def _observe(self):
        acts, mask = self.F.actions(self.st)
        acts = acts[mask]
        self.truncated_actions = 0
        if len(acts) > self.max_actions:              # keep a random subset of the legal moves
            self.truncated_actions = len(acts) - self.max_actions
            acts = acts[self.np_random.choice(len(acts), self.max_actions, replace=False)]
        self.cand = acts
        X = self.F.features(self.st, acts) if len(acts) else np.zeros((0, self.F.dim))
        self.cand_reward = self.F.reward(X, self.weights) if len(acts) else np.zeros(0)
        self.cand_dphi = X[:, self.F.blocks["dphi"]]
        x = np.zeros((self.max_actions, self.F.dim), np.float32); x[:len(acts)] = X
        m = np.zeros(self.max_actions, np.int8); m[:len(acts)] = 1
        return {"x": x, "mask": m}

    def _info(self, invalid=False):
        _, fam = self.F.representation(self.st)
        r = self.st.sums[:, 0] / self.F.ideal - 1
        return {"J": self.J_now, "best_J": self.best_J, "step": self.t, "n_actions": len(self.cand),
                "max_dev": float(np.abs(r).max()), "within_tol": bool(np.abs(r).max() <= self.tol + 1e-12),
                "cracked": {f: v["groups_cracked"] for f, v in fam.items()},
                "invalid": invalid, "truncated_actions": self.truncated_actions}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "start" in options:
            a = np.asarray(options["start"])
        else:
            a = self.pool[self.np_random.integers(len(self.pool)) if "index" not in options else options["index"]]
        self.st = self.F.state(a); self.t = 0
        self.start_assignment = self.st.a.copy()
        self.J_now = self.J(self.F.phi(self.st))
        self.best_J, self.best_assignment = (self.J_now, self.st.a.copy()) if self._within() \
            else (-np.inf, None)
        obs = self._observe()
        return obs, self._info()

    def step(self, action):
        action = int(action)
        if action >= len(self.cand):
            self.t += 1
            info = self._info(invalid=True)
            return self._obs_cache(), 0.0, False, self.t >= self.horizon, info
        i, t = map(int, self.cand[action])
        reward = float(self.cand_reward[action])
        dphi = dict(zip([n[2:] for n in self.F.names[self.F.blocks["dphi"]]], self.cand_dphi[action]))
        self.st.apply(i, t); self.t += 1
        self.J_now += reward
        if self.J_now > self.best_J + 1e-12 and self._within():
            self.best_J, self.best_assignment = self.J_now, self.st.a.copy()
        obs = self._observe(); self._last_obs = obs
        terminated = len(self.cand) == 0
        truncated = self.t >= self.horizon
        info = self._info(); info.update(move=(i, t), dphi=dphi)
        if terminated or truncated:
            info["final"] = self.evaluate(self.st.a)
            info["best"] = self.evaluate(self.best_assignment) if self.best_assignment is not None else None
        return obs, reward, terminated, truncated, info

    def _within(self):
        return bool(np.abs(self.st.sums[:, 0] / self.F.ideal - 1).max() <= self.tol + 1e-12)

    def _obs_cache(self):
        if getattr(self, "_last_obs", None) is None:
            self._last_obs = self._observe()
        return self._last_obs

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, assignment):
        """Objective components, J, constraint status and cracking table of a map."""
        st = self.F.state(assignment)
        phi = self.F.phi(st)
        rows, fam = self.F.representation(st)
        r = st.sums[:, 0] / self.F.ideal - 1
        return {"phi": phi, "J": self.J(phi), "max_dev": float(np.abs(r).max()),
                "within_tol": bool(np.abs(r).max() <= self.tol + 1e-12),
                "contiguous": all(_connected(np.flatnonzero(st.a == d), self.g.nbr) for d in range(self.p)),
                "families": fam, "groups": rows}


def _connected(nodes, nbr):
    nodes = set(nodes.tolist())
    if not nodes:
        return False
    s = next(iter(nodes)); seen = {s}; st = [s]
    while st:
        x = st.pop()
        for y in nbr[x]:
            if y in nodes and y not in seen:
                seen.add(y); st.append(y)
    return len(seen) == len(nodes)
