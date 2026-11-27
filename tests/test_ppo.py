"""PPO pieces: masked policy, critic columns, GAE, state features, a smoke run."""
import warnings
import numpy as np
import pytest
import torch
from redistricting.rl.ppo import Actor, Critic, critic_columns, gae, pad
from redistricting.rl.qnet import QNetwork


def test_actor_masks_padding_and_loads_as_qnetwork():
    torch.manual_seed(0)
    pi = Actor(6)
    X, M = pad([np.random.randn(3, 6), np.random.randn(1, 6)], 6)
    d = pi.dist(X, M)
    assert torch.allclose(d.probs.sum(1), torch.ones(2))
    assert d.probs[1, 1:].max() == 0 and d.probs[1, 0] == 1
    assert torch.isfinite(d.entropy()).all() and d.entropy()[1] == 0
    q = QNetwork(6); q.load_state_dict(pi.state_dict())            # same parameters
    assert torch.allclose(q(X[0]), pi(X[0]))


def test_load_embedding_keeps_head():
    q = QNetwork(5); q.norm.mean.fill_(3.0)
    pi = Actor(5); head = pi.q.weight.clone()
    pi.load_embedding({"model": q.state_dict()})
    assert torch.equal(pi.h[0].weight, q.h[0].weight) and torch.equal(pi.norm.mean, q.norm.mean)
    assert torch.equal(pi.q.weight, head)
    with pytest.raises(ValueError):
        Actor(5, hidden=(7,)).load_embedding({"model": {}})


def test_critic_columns():
    blocks = {"districts": slice(0, 3), "family": slice(3, 5), "phi": slice(5, 8), "time": slice(8, 9)}
    assert critic_columns(blocks) == [0, 1, 2, 3, 4]
    assert critic_columns(blocks, ["time"]) == [0, 1, 2, 3, 4, 8]
    v = Critic(critic_columns(blocks))
    s = torch.zeros(7, 9); s[:, 5:] = 1e6                         # unseen columns cannot matter
    assert torch.allclose(v(s), v(torch.zeros(7, 9)))


def test_gae():
    # one env, 3 steps, episode boundary before step 2 (dones[2] = 1)
    r = torch.tensor([[1.], [2.], [3.]]); v = torch.tensor([[0.5], [1.], [2.]])
    d = torch.tensor([[0.], [0.], [1.]])
    adv, ret = gae(r, v, d, torch.tensor([4.]), torch.tensor([0.]), gamma=1.0, lam=1.0)
    # lam = gamma = 1: returns are rewards-to-go within the episode (+ bootstrap at the end)
    assert torch.allclose(ret[:, 0], torch.tensor([3., 2., 7.]))
    assert torch.allclose(adv, ret - v)
    adv0, _ = gae(r, v, d, torch.tensor([4.]), torch.tensor([0.]), gamma=0.5, lam=0.0)
    assert torch.allclose(adv0[:, 0], torch.tensor([1 + 0.5 * 1 - 0.5, 2 - 1, 3 + 0.5 * 4 - 2]))


@pytest.fixture(scope="module")
def env():
    from redistricting.rl.env import RedistrictingEnv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RedistrictingEnv("IA", "congress", pool_size=2, horizon=30)


def test_state_features(env):
    obs, _ = env.reset(seed=0)
    F = env.F
    assert obs["s"].shape == (F.state_dim,) and len(F.state_names) == F.state_dim
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, *_ = env.step(int(rng.integers(obs["mask"].sum())))
    ph = F.phi(env.st)
    assert max(abs(ph[k] - env.phi_now[k]) for k in ph) < 1e-9       # incremental phi is exact
    x = F.state_features(env.st, ph, 0.5)
    # relabelling districts leaves x_state unchanged
    perm = rng.permutation(env.p)
    x2 = F.state_features(F.state(perm[env.st.a]), ph, 0.5)
    assert np.allclose(x, x2)
    assert x[F.state_blocks["time"]][0] == 0.5


def test_training_smoke(tmp_path):
    from experiments.train_ppo import main
    from redistricting.rl.env import RedistrictingEnv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dim = RedistrictingEnv("IA", "congress", pool_size=4).F.dim
    q = QNetwork(dim); torch.save({"model": q.state_dict()}, tmp_path / "dqn.pt")
    out = main(["--total-timesteps", "128", "--num-envs", "2", "--num-steps", "32", "--norm-steps", "60",
                "--eval-episodes", "1", "--horizon", "20", "--pool-size", "4", "--critic-phi", "--critic-time",
                "--init-embedding", str(tmp_path / "dqn.pt"), "--out-dir", str(tmp_path)])
    for f in ("progress.csv", "episodes.csv", "eval.csv", "model.pt", "args.json", "weights.json"):
        assert (out / f).exists()
    ck = torch.load(out / "model.pt", weights_only=False)
    QNetwork(ck["dim"], ck["hidden"]).load_state_dict(ck["model"])     # greedy replay tools work
    Critic(ck["critic_cols"], ck["critic_hidden"]).load_state_dict(ck["critic"])
