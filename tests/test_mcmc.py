"""MCMC samplers and plan statistics: exactness against full enumeration."""
import numpy as np
import pytest
from scipy.stats import chisquare

from redistricting.geo import Geo
from redistricting import stats
from redistricting.enumerate.zdd import PartitionZDD, max_deviation
from redistricting.enumerate.bruteforce import grid, canonical
from redistricting.mcmc import FlipChain, ReComChain, rhat, autocorr_time


class _G:
    """Minimal unit graph for synthetic tests."""
    def __init__(self, n, E, pop):
        self.N = n; self.pop = np.asarray(pop, float)
        nb = [[] for _ in range(n)]
        for u, v in E: nb[u].append(v); nb[v].append(u)
        self.nbr = [tuple(x) for x in nb]
        self.border = {(min(u, v), max(u, v)): 1.0 for u, v in E}


def _ends(kind, g, W, p, wt, tol, R, T, rng):
    ends = []; ideal = g.pop.sum() / p
    for r in range(R):
        s = W[rng.integers(len(W))]
        ch = FlipChain(g, s, p, wt, np.random.default_rng(r)) if kind == "flip" \
            else ReComChain(g, s, p, wt, r)
        ch.run(T)
        if np.abs(np.bincount(ch.a, g.pop, p) / ideal - 1).max() <= tol:
            ends.append(tuple(ch.a))
    return np.array(ends)


@pytest.fixture(scope="module")
def grid_case():
    n, E = grid(3, 3); p = 3; tol, wt = 0.25, 0.5
    g = _G(n, E, np.random.default_rng(0).integers(50, 150, n))
    L = PartitionZDD(n, E, p).all_labels(); dev = max_deviation(L, g.pop, p)
    V = L[dev <= tol]; W = L[dev <= wt]
    return g, p, tol, wt, V, W


def test_flip_chain_is_uniform(grid_case):
    """End states of independent chains (fixed length, filtered to tol) are uniform
    over the valid plans enumerated by the ZDD."""
    g, p, tol, wt, V, W = grid_case
    idx = {q: i for i, q in enumerate(canonical(V))}
    E = _ends("flip", g, W, p, wt, tol, 4000, 200, np.random.default_rng(1))
    c = np.bincount([idx[q] for q in canonical(E)], minlength=len(V))
    assert c.min() > 0
    assert chisquare(c).pvalue > 1e-3


def test_recom_control_is_not_uniform(grid_case):
    g, p, tol, wt, V, W = grid_case
    idx = {q: i for i, q in enumerate(canonical(V))}
    E = _ends("recom", g, W, p, wt, tol, 2000, 30, np.random.default_rng(2))
    c = np.bincount([idx[q] for q in canonical(E)], minlength=len(V))
    assert chisquare(c).pvalue < 1e-6


def test_flip_chain_keeps_invariants():
    n, E = grid(4, 5); p = 3
    g = _G(n, E, np.ones(n))
    z = PartitionZDD(n, E, p); L = z.sample(2000, np.random.default_rng(0))
    s = L[max_deviation(L, g.pop, p) <= 0.3][0]
    ch = FlipChain(g, s, p, 0.3, np.random.default_rng(3))
    import networkx as nx
    G = nx.Graph(E)
    for _ in range(3000):
        ch.step()
        assert ch.B == sum(ch._b(u) for u in range(n))
    for d in range(p):
        assert nx.is_connected(G.subgraph([u for u in range(n) if ch.a[u] == d]))
    P = np.bincount(ch.a, g.pop, p)
    assert np.allclose(P, ch.dpop) and np.abs(P / (n / p) - 1).max() <= 0.3 + 1e-12


def test_stats_on_real_units():
    g = Geo.load("IA_county", 4)
    rng = np.random.default_rng(0)
    L = rng.integers(0, 4, size=(50, g.N)).astype(np.int8)
    S = stats.all_stats(L, g, 4)
    assert np.allclose(S["pop_dev"], max_deviation(L, g.pop, 4))
    for k in ("D_race", "D_edu", "D_pov"):
        assert ((S[k] >= 0) & (S[k] <= 1)).all()
    # one district = whole state: perimeter is the state's outer boundary
    one = np.zeros((1, g.N), dtype=np.int8)
    outer = g.perim.sum() - 2 * sum(g.border.values())
    assert np.isclose(stats.district_perimeters(one, g, 1)[0, 0], outer)
    # relabelling districts changes nothing
    perm = np.array([2, 0, 3, 1], dtype=np.int8)
    S2 = stats.all_stats(perm[L], g, 4)
    for k in S:
        assert np.allclose(S[k], S2[k])


def test_vi_properties():
    rng = np.random.default_rng(0); w = rng.random(30)
    x = rng.integers(0, 3, 30); y = rng.integers(0, 3, 30)
    assert stats.vi(x, x, w, 3) == pytest.approx(0, abs=1e-12)
    assert stats.vi(x, y, w, 3) == pytest.approx(stats.vi(y, x, w, 3))
    assert stats.vi(x, (x + 1) % 3, w, 3) == pytest.approx(0, abs=1e-12)
    assert stats.vi(x, y, w, 3) <= 2 * np.log(3) + 1e-12


def test_subset_keeps_true_perimeter():
    g = Geo.load("IA_tract", 2)
    ids = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "data" / "IA_tract_burlington70.txt").read_text().split()
    h = g.subset([g.idx[x] for x in ids], 2)
    assert h.N == 70 and len(h.nbr) == 70
    assert all(j < 70 for v in h.nbr for j in v)
    one = np.zeros((1, h.N), dtype=np.int8)
    assert stats.district_perimeters(one, h, 1)[0, 0] > 0


def test_diagnostics():
    rng = np.random.default_rng(0)
    iid = [rng.normal(size=4000) for _ in range(4)]
    assert abs(rhat(iid) - 1) < 0.01 and autocorr_time(iid[0]) < 1.5
    shifted = [rng.normal(size=4000) + k for k in range(4)]
    assert rhat(shifted) > 1.5
