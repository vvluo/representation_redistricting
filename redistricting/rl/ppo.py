"""PPO actor-critic over the flip features, on the same embedding as the Q-network.

    h_theta(s, a)    = MLP_theta(x(s, a))                 shared embedding (as QNetwork)
    l_theta(s, a)    = W_pi^T h_theta(s, a) + b_pi        one logit per legal flip
    pi_theta(a | s)  = softmax_a l_theta(s, a)            over the legal rows only
    V_phi(s)         = MLP_phi(x_state(s))                separate critic

x(s, a) is one row per legal flip (features.py).  x_state(s) holds only what belongs to
the map rather than to one candidate flip (FlipFeatures.state_features): district
descriptors summarised over all districts and per-family representation status, and
optionally the objective levels phi(s) and t / horizon.

Actor has exactly QNetwork's parameters (norm, h, q), with q = (W_pi, b_pi), so a PPO
checkpoint loads into QNetwork for greedy replay (argmax_a l = argmax_a pi), and a DQN
checkpoint's embedding can warm-start the actor (load_embedding).

Losses (cleanrl/ppo.py), with ratio rho = pi_theta(a|s) / pi_old(a|s) and normalised
GAE advantages A:
  L_pi  = E[max(-A rho, -A clip(rho, 1 - eps, 1 + eps))]
  L_V   = 0.5 E[max((V - R)^2, (V_old + clip(V - V_old, -eps, eps) - R)^2)]   (clipped)
  L     = L_pi - c_ent H[pi(.|s)] + c_V L_V
"""
import numpy as np
import torch
import torch.nn as nn
from .qnet import QNetwork, FeatureNorm

NEG = -1e8          # logit for padding rows: exp() underflows to 0, unlike -inf no 0 * inf in H


class Actor(QNetwork):
    """pi_theta(a | s) = softmax over legal rows of W_pi h_theta(x(s, a)) + b_pi."""

    def __init__(self, dim, hidden=(120, 84), head_std=0.01):
        super().__init__(dim, hidden)
        nn.init.orthogonal_(self.q.weight, head_std); nn.init.zeros_(self.q.bias)   # ~uniform start

    def logits(self, X, M):
        """X (..., K, F) rows, M (..., K) legal mask -> masked logits (..., K)."""
        return self(X).masked_fill(~M, NEG)

    def dist(self, X, M):
        return torch.distributions.Categorical(logits=self.logits(X, M))

    def load_embedding(self, ckpt):
        """Copy h_theta and the feature standardisation from a QNetwork checkpoint."""
        sd = ckpt["model"] if "model" in ckpt else ckpt
        keep = {k: v for k, v in sd.items() if k.startswith(("norm.", "h."))}
        missing = [k for k in self.state_dict() if k.startswith(("norm.", "h.")) and k not in keep]
        if missing:
            raise ValueError(f"checkpoint lacks {missing}")
        self.load_state_dict(keep, strict=False)


class Critic(nn.Module):
    """V_phi(s) = MLP_phi(x_state(s)) restricted to columns `cols` of the state vector."""

    def __init__(self, cols, hidden=(64, 64)):
        super().__init__()
        self.register_buffer("cols", torch.as_tensor(np.asarray(cols), dtype=torch.long))
        dim = len(cols)
        self.norm = FeatureNorm(dim)
        layers, d = [], dim
        for h in hidden:
            lin = nn.Linear(d, h); nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]; d = h
        out = nn.Linear(d, 1); nn.init.orthogonal_(out.weight, 1.0); nn.init.zeros_(out.bias)
        self.mlp = nn.Sequential(*layers, out)

    def select(self, s):
        return s[..., self.cols]

    def forward(self, s):
        """s (..., state_dim) full x_state -> V (...)"""
        return self.mlp(self.norm(self.select(s))).squeeze(-1)


def critic_columns(state_blocks, extra=()):
    """Columns of x_state the critic sees: districts + family, plus any of phi, time."""
    cols = []
    for b in ("districts", "family", *extra):
        sl = state_blocks[b]; cols += list(range(sl.start, sl.stop))
    return cols


def gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """Generalised advantage estimates for a (T, E) rollout.  dones[t] = 1 when the
    episode ended before step t (as in CleanRL: dones are the flags of the observation
    at t), so no bootstrapping across episode boundaries.  Truncation is handled by the
    caller folding gamma * V(s_T) into the last reward."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards); last = 0
    for t in reversed(range(T)):
        nonterm, nv = (1.0 - next_done, next_value) if t == T - 1 else (1.0 - dones[t + 1], values[t + 1])
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        adv[t] = last = delta + gamma * lam * nonterm * last
    return adv, adv + values


def pad(rows, dim):
    """List of (n_i, F) arrays -> X (B, K, F) float32, M (B, K) bool; K = max(1, max n_i)."""
    K = max(1, max(len(r) for r in rows))
    X = np.zeros((len(rows), K, dim), np.float32); M = np.zeros((len(rows), K), bool)
    for b, r in enumerate(rows):
        X[b, :len(r)] = r; M[b, :len(r)] = True
    return torch.as_tensor(X), torch.as_tensor(M)
