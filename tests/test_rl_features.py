"""Flip features: every incremental quantity against a from-scratch recomputation."""
import random
import numpy as np
import pytest
from redistricting import Geo
from redistricting.inits import make_init
from redistricting.rl import FlipFeatures, articulation
from redistricting.rl.groups import Group, attainable

P = 50


@pytest.fixture(scope="module")
def setup():
    g = Geo.load("IA_tract", D=P)
    F = FlipFeatures(g, P)
    a = make_init("kmeans", g, random.Random(3))
    return g, F, a


def _connected(nodes, nbr):
    nodes = set(nodes)
    if not nodes:
        return False
    s = next(iter(nodes)); seen = {s}; st = [s]
    while st:
        x = st.pop()
        for y in nbr[x]:
            if y in nodes and y not in seen:
                seen.add(y); st.append(y)
    return len(seen) == len(nodes)


def test_dphi_matches_recomputation(setup):
    g, F, a = setup
    st = F.state(a)
    acts, _ = F.actions(st)
    X = F.features(st, acts)
    b = F.blocks["dphi"]; names = [n[2:] for n in F.names[b]]
    phi0 = F.phi(st)
    rng = np.random.default_rng(0)
    for k in rng.choice(len(acts), 60, replace=False):
        i, t = acts[k]
        a2 = st.a.copy(); a2[i] = t
        phi1 = F.phi(F.state(a2))
        for j, n in enumerate(names):
            assert X[k, b][j] == pytest.approx(phi1[n] - phi0[n], abs=1e-10), (n, i, t)


def test_family_signal_is_nonzero_somewhere(setup):
    g, F, a = setup
    st = F.state(a); acts, _ = F.actions(st)
    X = F.features(st, acts)
    j = F.names.index("d_poverty")
    assert np.abs(X[:, j]).max() > 0


def test_mask_matches_bfs(setup):
    g, F, a = setup
    st = F.state(a); acts, mask = F.actions(st)
    for (i, t), ok in zip(acts, mask):
        s = st.a[i]
        rest = [u for u in np.flatnonzero(st.a == s) if u != i]
        P_ = st.sums[:, 0]; pi = g.pop[i]; ideal = F.ideal
        rs0, rs1 = abs(P_[s] / ideal - 1), abs((P_[s] - pi) / ideal - 1)
        rt0, rt1 = abs(P_[t] / ideal - 1), abs((P_[t] + pi) / ideal - 1)
        pop_ok = (rs1 <= F.tol or rs1 < rs0) and (rt1 <= F.tol or rt1 < rt0)
        assert ok == (bool(rest) and _connected(rest, g.nbr) and pop_ok)


def test_incremental_state_equals_rebuilt(setup):
    g, F, a = setup
    st = F.state(a); rng = np.random.default_rng(1)
    for _ in range(40):
        acts, mask = F.actions(st)
        k = rng.choice(np.flatnonzero(mask)) if mask.any() else rng.integers(len(acts))
        st.apply(*acts[k])
    fresh = F.state(st.a)
    assert np.allclose(st.sums, fresh.sums) and np.allclose(st.perim, fresh.perim)
    assert np.array_equal(st.size, fresh.size)
    assert np.allclose(st.ccount, fresh.ccount) and np.allclose(st.cpop, fresh.cpop)
    assert all(x == y for x, y in zip(st.artic, fresh.artic))
    acts, _ = F.actions(st)
    assert np.allclose(F.features(st, acts), F.features(fresh, acts))


def test_articulation_on_small_graphs():
    # path 0-1-2-3: 1 and 2 are cut vertices;  cycle: none;  star: the centre
    path = [[1], [0, 2], [1, 3], [2]]
    assert articulation([0, 1, 2, 3], path, [0] * 4, 0) == {1, 2}
    cyc = [[1, 3], [0, 2], [1, 3], [2, 0]]
    assert articulation([0, 1, 2, 3], cyc, [0] * 4, 0) == set()
    star = [[1, 2, 3], [0], [0], [0]]
    assert articulation([0, 1, 2, 3], star, [0] * 4, 0) == {0}


def test_feature_names_and_reward(setup):
    g, F, a = setup
    assert len(F.names) == F.dim == len(set(F.names))
    st = F.state(a); acts, _ = F.actions(st)
    X = F.features(st, acts)
    r = F.reward(X, {"poverty": 2.0, "industry": 0.5})
    expect = 2.0 * X[:, F.names.index("d_poverty")] + 0.5 * X[:, F.names.index("d_industry")]
    assert np.allclose(r, expect)


def test_attainable_is_zero_when_tau_unreachable(setup):
    g, F, a = setup
    gr = [Group("poverty", "poverty", ("pov_below",), "pov_total", tau=0.99)]
    assert attainable(g, gr, P)[0].N == 0
    gr = [Group("poverty", "poverty", ("pov_below",), "pov_total", tau=0.0)]
    assert attainable(g, gr, P)[0].N >= 1


def test_families_switch_on_with_data(setup):
    g, F, a = setup
    assert F.families[:5] == ["poverty", "ethnicity", "industry", "rurality", "education"]
    assert "d_rurality" in F.names and "d_education" in F.names
    assert "d_food" not in F.names                       # no SNAP columns yet
    tiers = sum(g.col(f"pop_{t}") for t in ("micropolitan", "small_town", "rural"))
    assert np.all(tiers <= g.pop + 1e-9)
    F2 = FlipFeatures(g, P, families=list(F.families) + ["food"])   # pinned dims: zeros
    st = F2.state(a); acts, _ = F2.actions(st)
    X = F2.features(st, acts)
    assert np.all(X[:, F2.names.index("d_food")] == 0)


def test_derive_counts():
    import pandas as pd
    from redistricting.rl.groups import derive_counts
    M = pd.DataFrame({"pop": [100.0, 50, 10], "ruca_primary": [1, 5, 10], "emp_total": [40.0, 20, 0],
                      **{f"ind_{k:02d}_share": [100 / 13] * 3 for k in range(1, 14)}})
    D = derive_counts(M)
    assert D.pop_micropolitan.tolist() == [0, 50, 0] and D.pop_rural.tolist() == [0, 0, 10]
    assert np.allclose(D[[f"ind_{k:02d}" for k in range(1, 14)]].sum(axis=1), M.emp_total)


def test_cohesion_definition(setup):
    g, F, a = setup
    assert "d_coh_poverty" in F.names
    k = [gr.name for gr in F.groups].index("poverty")
    f = F.tshare[:, k]
    E = [(i, j, L) for (i, j), L in g.border.items()]
    num = sum(L * f[i] * f[j] for i, j, L in E if a[i] == a[j])
    den = sum(L * f[i] * f[j] for i, j, L in E)
    assert F.phi(F.state(a))["coh_poverty"] == pytest.approx(num / den, rel=1e-9)
    assert 0 <= F.phi(F.state(a))["coh_poverty"] <= 1
