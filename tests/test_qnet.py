"""DQN / DDQN pieces: network shapes, TD targets, replay padding, a smoke run."""
import numpy as np
import pytest
import torch
from redistricting.rl.qnet import QNetwork, ReplayBuffer, td_target, linear_schedule, soft_update


class Col(torch.nn.Module):
    """Q(s, a) = one feature of x(s, a): known values for checking targets."""
    def __init__(self, j):
        super().__init__(); self.j = j

    def forward(self, x):
        return x[..., self.j]


def batch():
    # two transitions; next states have 3 and 2 legal rows (row 2 of the second is padding)
    X2 = torch.tensor([[[1., 5.], [3., 0.], [2., 9.]],
                       [[4., 1.], [0., 7.], [99., 99.]]])
    M2 = torch.tensor([[True, True, True], [True, True, False]])
    r = torch.tensor([1., 2.]); d = torch.tensor([0., 0.])
    return r, d, X2, M2


def test_targets():
    r, d, X2, M2 = batch(); g = 0.5
    on, tg = Col(0), Col(1)
    # dqn, no target net: max over online = 3 and 4
    assert torch.allclose(td_target(r, d, X2, M2, on, on, g, "dqn", use_target=False), r + g * torch.tensor([3., 4.]))
    # dqn with target net: max over target = 9 and 7 (padding's 99 ignored)
    assert torch.allclose(td_target(r, d, X2, M2, on, tg, g, "dqn", use_target=True), r + g * torch.tensor([9., 7.]))
    # ddqn: online argmax rows 1 and 0, evaluated by target -> 0 and 1
    assert torch.allclose(td_target(r, d, X2, M2, on, tg, g, "ddqn"), r + g * torch.tensor([0., 1.]))
    # termination and empty next states do not bootstrap
    assert torch.allclose(td_target(r, torch.tensor([1., 0.]), X2, M2, on, tg, g, "ddqn"), torch.tensor([1., 2.5]))
    M0 = M2.clone(); M0[0] = False
    assert td_target(r, d, X2, M0, on, tg, g, "ddqn")[0] == 1.0


def test_network_and_buffer():
    q = QNetwork(5)
    assert q(torch.zeros(7, 5)).shape == (7,) and q(torch.zeros(3, 7, 5)).shape == (3, 7)
    q.norm.fit(np.c_[np.random.randn(100, 4), np.ones(100)])       # constant column keeps std 1
    assert q.norm.std[-1] == 1
    rb = ReplayBuffer(3, 5)
    for k, n in enumerate([2, 0, 4, 1]):                           # wraps; one empty next state
        rb.add(np.full(5, k), k, False, np.ones((n, 5)))
    assert len(rb) == 3
    x, r, d, X2, M2 = rb.sample(16)
    assert X2.shape[1] == 4 and set(M2.sum(1).tolist()) <= {0, 1, 4}


def test_schedule_and_soft_update():
    assert linear_schedule(1, 0.05, 100, 0) == 1 and linear_schedule(1, 0.05, 100, 500) == 0.05
    a, b = QNetwork(3), QNetwork(3)
    a.norm.mean.fill_(2.0)
    soft_update(b, a, 1.0)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert torch.equal(b.norm.mean, a.norm.mean)


@pytest.mark.parametrize("algo,extra", [("ddqn", []), ("dqn", []), ("dqn", ["--target-network"])])
def test_training_smoke(tmp_path, algo, extra):
    from experiments.train_dqn import main
    out = main(["--algo", algo, *extra, "--total-timesteps", "300", "--learning-starts", "100",
                "--batch-size", "16", "--norm-steps", "60", "--eval-episodes", "1", "--horizon", "40",
                "--pool-size", "4", "--target-network-frequency", "50", "--out-dir", str(tmp_path)])
    for f in ("progress.csv", "episodes.csv", "eval.csv", "model.pt", "args.json"):
        assert (out / f).exists()
    ck = torch.load(out / "model.pt", weights_only=False)
    q = QNetwork(ck["dim"], ck["hidden"]); q.load_state_dict(ck["model"])
