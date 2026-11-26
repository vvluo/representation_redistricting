"""Redistricting Gymnasium env: API, hard constraints, reward accounting, presets."""
import warnings
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from redistricting.states import PRESETS, TOLERANCE, districts, load_state
from redistricting.rl.env import RedistrictingEnv


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return RedistrictingEnv("IA", "congress", pool_size=6, pool_dir=tmp_path_factory.mktemp("pools"))


def test_presets():
    assert sum(v[1] for v in PRESETS.values()) == 435
    assert districts("IA", "congress") == 4 and districts("IA", "house") == 100
    assert districts("AZ", "house") == 30 and PRESETS["AZ"][3] == 60      # districts, not seats
    assert TOLERANCE == {"congress": 0.01, "house": 0.05}
    with pytest.raises(ValueError):
        load_state("WY", "congress")                                       # one district


def test_gymnasium_api(env):
    check_env(env, skip_render_check=True)


def test_rollout_keeps_constraints_and_accounts_reward(env):
    obs, info = env.reset(seed=1)
    J0 = info["J"]; R = 0.0; rng = np.random.default_rng(0)
    for k in range(150):
        a = rng.choice(np.flatnonzero(obs["mask"]))
        obs, r, term, trunc, info = env.step(a); R += r
        assert info["within_tol"] and not info["invalid"]
        if term or trunc:
            break
    ev = env.evaluate(env.st.a)
    assert ev["contiguous"] and ev["within_tol"]
    assert R == pytest.approx(ev["J"] - J0, abs=1e-9)
    assert info["J"] == pytest.approx(ev["J"], abs=1e-9)
    assert env.evaluate(env.best_assignment)["J"] == pytest.approx(info["best_J"], abs=1e-9)


def test_padding_action_is_noop(env):
    obs, info = env.reset(seed=2)
    a0 = env.st.a.copy()
    pad = int(obs["mask"].sum())
    if pad < env.max_actions:
        obs2, r, term, trunc, info2 = env.step(pad)
        assert info2["invalid"] and r == 0.0 and np.array_equal(env.st.a, a0)


def test_greedy_improves_J(env):
    obs, info = env.reset(seed=3); J0 = info["J"]
    for _ in range(40):
        k = int(np.argmax(np.where(obs["mask"][:len(env.cand_reward)] == 1, env.cand_reward, -np.inf)))
        if env.cand_reward[k] <= 0:
            break
        obs, r, term, trunc, info = env.step(k)
    assert info["J"] > J0


def test_episode_end_reports(env):
    e = RedistrictingEnv("IA", "congress", pool=env.pool, horizon=5)
    obs, info = e.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, info = e.step(0)
    assert trunc and "final" in info and "best" in info
    assert {"phi", "J", "families", "groups", "contiguous"} <= set(info["final"])


def test_islands_are_bridged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, p, tol = load_state("HI", "congress")
    assert len(g.bridges) >= 1
    seen = {0}; st = [0]
    while st:
        x = st.pop()
        for y in g.nbr[x]:
            if y not in seen:
                seen.add(y); st.append(y)
    assert len(seen) == g.N


def test_explore_band(env):
    e = RedistrictingEnv("IA", "congress", pool=env.pool, explore_tol=0.05, horizon=400)
    assert e.tol == 0.01 and e.explore_tol == 0.05 and e.weights.get("excess") == 1.0
    obs, info = e.reset(seed=4); J0 = info["J"]; R = 0.0
    rng = np.random.default_rng(1); left = False
    for _ in range(400):
        obs, r, term, trunc, info = e.step(rng.choice(np.flatnonzero(obs["mask"]))); R += r
        assert info["max_dev"] <= 0.05 + 1e-12            # never outside the exploration band
        left |= not info["within_tol"]
        if term or trunc:
            break
    assert left                                            # random flips do leave the 1% band
    assert R == pytest.approx(e.evaluate(e.st.a)["J"] - J0, abs=1e-9)
    assert info["best"] is None or info["best"]["within_tol"]
    assert e.evaluate(e.best_assignment)["within_tol"]
