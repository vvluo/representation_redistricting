"""Q-networks over action features, a replay buffer for variable action sets, and the
DQN / Double-DQN targets.

    h_theta(s, a) = MLP_theta(x(s, a))                 shared embedding (CleanRL's 120 -> 84)
    Q_theta(s, a) = w_Q^T h_theta(s, a) + b_Q          scalar head

Each legal flip a is one row x(s, a) of the observation, so the network scores rows
and the greedy action is the argmax over the legal rows (padding rows are masked).

Targets, for a transition (s, a, r, s', d) with d = 1 only on termination (no legal
flip left; truncation at the horizon still bootstraps, as in CleanRL):
  dqn                    y = r + gamma (1 - d) max_a' Q_online(s', a')          (no target net)
  dqn + target network   y = r + gamma (1 - d) max_a' Q_target(s', a')          (= cleanrl/dqn.py)
  ddqn                   y = r + gamma (1 - d) Q_target(s', argmax_a' Q_online(s', a'))
The target network is a delayed copy of the online network, updated every
`target_network_frequency` steps by  target <- tau * online + (1 - tau) * target.
"""
import numpy as np
import torch
import torch.nn as nn


class FeatureNorm(nn.Module):
    """Fixed per-feature standardisation, fitted once on rollout features and stored in
    the network so checkpoints carry it."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.mean.copy_(X.mean(0))
        s = X.std(0)
        self.std.copy_(torch.where(s > 1e-8, s, torch.ones_like(s)))   # constant columns -> pass through

    def forward(self, x):
        return (x - self.mean) / self.std


class QNetwork(nn.Module):
    def __init__(self, dim, hidden=(120, 84)):
        super().__init__()
        self.norm = FeatureNorm(dim)
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.h = nn.Sequential(*layers)          # h_theta
        self.q = nn.Linear(d, 1)                 # w_Q, b_Q

    def embed(self, x):
        return self.h(self.norm(x))

    def forward(self, x):
        """x (..., F) raw features -> Q (...)"""
        return self.q(self.embed(x)).squeeze(-1)


class ReplayBuffer:
    """Circular buffer of (x(s,a), r, terminated, next-state legal rows).  Only the
    taken action's row of s is kept (all Q(s,a) needs); the next state's legal rows are
    stored ragged in float32 and padded per sampled batch.  Memory ~ capacity x mean
    legal moves x F x 4 bytes (Iowa congress: 10k x ~130 x 125 x 4 = ~0.65 GB)."""

    def __init__(self, capacity, dim, seed=0):
        self.cap, self.dim = capacity, dim
        self.x = np.zeros((capacity, dim), np.float32)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.nxt = [None] * capacity
        self.n = 0; self.i = 0
        self.rng = np.random.default_rng(seed)

    def add(self, x_sa, r, terminated, next_rows):
        k = self.i
        self.x[k] = x_sa; self.r[k] = r; self.d[k] = float(terminated)
        self.nxt[k] = np.asarray(next_rows, np.float32)
        self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def __len__(self):
        return self.n

    def sample(self, batch, device="cpu"):
        idx = self.rng.integers(self.n, size=batch)
        K = max(1, max(len(self.nxt[k]) for k in idx))
        X2 = np.zeros((batch, K, self.dim), np.float32); M2 = np.zeros((batch, K), bool)
        for b, k in enumerate(idx):
            m = len(self.nxt[k]); X2[b, :m] = self.nxt[k]; M2[b, :m] = True
        t = lambda a: torch.as_tensor(a, device=device)
        return t(self.x[idx]), t(self.r[idx]), t(self.d[idx]), t(X2), t(M2)


def td_target(r, d, X2, M2, online, target, gamma, algo="ddqn", use_target=True):
    """Bootstrapped targets for a batch; X2 (B, K, F) next-state rows, M2 (B, K) mask.
    A next state with no legal rows contributes 0 (it is terminal)."""
    with torch.no_grad():
        q_on = online(X2).masked_fill(~M2, -torch.inf)
        has = M2.any(1)
        if algo == "ddqn":
            a_star = q_on.argmax(1, keepdim=True)
            v = target(X2).gather(1, a_star).squeeze(1)
        else:
            net_q = target(X2).masked_fill(~M2, -torch.inf) if use_target else q_on
            v = net_q.max(1).values
        v = torch.where(has, v, torch.zeros_like(v))
        return r + gamma * (1 - d) * v


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def soft_update(target, online, tau):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
    for tb, b in zip(target.buffers(), online.buffers()):
        tb.data.copy_(b.data)
