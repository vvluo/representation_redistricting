"""Partition ZDD: exact agreement with independent enumerators, exact uniform sampling."""
import itertools
import numpy as np
import networkx as nx
import pytest

from redistricting.enumerate.zdd import (PartitionZDD, BuildAborted, district_pops,
                                         max_deviation, population_ok)
from redistricting.enumerate.ordering import (farthest_pair, order_cut, order_bfs,
                                              max_frontier, frontiers, best_order)
from redistricting.enumerate.bruteforce import (brute_partitions, canonical, grid, path,
                                                cycle, wheel, complete, random_tree,
                                                delaunay, closed_form)

RNG = np.random.default_rng(7)
SMALL = {
    "grid3x3": grid(3, 3),
    "grid2x6": grid(2, 6),
    "trigrid3x4": grid(3, 4, diagonals=True),
    "wheel9": wheel(9),
    "K5": complete(5),
    "delaunay11": delaunay(11, RNG),
    "tree11": random_tree(11, RNG),
}


@pytest.mark.parametrize("name", sorted(SMALL))
@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("order", ["cut", "bfs"])
def test_matches_brute_force(name, p, order):
    n, E = SMALL[name]
    z = PartitionZDD(n, E, p, order=order)
    B = brute_partitions(n, E, p)
    Z = canonical(z.all_labels()) if z.count else []
    assert z.count == len(B)
    assert len(set(Z)) == len(Z)          # every path a different partition
    assert set(Z) == B                    # and exactly the brute-force set


def _bipartitions_by_subsets(r, c):
    """Third, independent count: subsets S containing vertex 0 with S, V-S connected."""
    G = nx.grid_2d_graph(r, c); V = sorted(G); cnt = 0
    for k in range(1, len(V)):
        for S in itertools.combinations(V[1:], k - 1):
            S = set(S) | {V[0]}
            if nx.is_connected(G.subgraph(S)) and nx.is_connected(G.subgraph(set(V) - S)):
                cnt += 1
    return cnt


def test_grid_bipartitions_independent():
    assert PartitionZDD(*grid(4, 4), 2).count == _bipartitions_by_subsets(4, 4) == 627


@pytest.mark.parametrize("kind,graph,p", [
    ("cycle", cycle(60), 5), ("cycle", cycle(40), 8), ("path", path(50), 6),
    ("tree", random_tree(60, np.random.default_rng(3)), 4),
    ("complete", complete(8), 3), ("complete", complete(9), 4)])
def test_closed_forms(kind, graph, p):
    n, E = graph
    assert PartitionZDD(n, E, p).count == closed_form(kind, n, p)


def test_sampler_is_exactly_uniform():
    """Every partition's sampling probability (product of arc probabilities) is 1/count."""
    z = PartitionZDD(*grid(3, 4), 3)
    idx = np.arange(z.count); z.unrank(idx[:1])
    node = np.zeros(z.count, dtype=np.int64); prob = np.ones(z.count); r = idx.copy()
    for l in range(z.m):
        c0 = z._c0[l][node]; take = r >= c0; r = np.where(take, r - c0, r)
        ph = z.phi[l][node]; prob *= np.where(take, ph, 1 - ph)
        node = np.where(take, z.hi[l][node], z.lo[l][node]) - 2
    assert np.allclose(prob * z.count, 1.0, atol=1e-12)


def test_samples_are_valid_partitions():
    n, E = delaunay(14, np.random.default_rng(5)); p = 3
    z = PartitionZDD(n, E, p)
    S = z.sample(2000, np.random.default_rng(0))
    A = [[] for _ in range(n)]
    for u, v in E: A[u].append(v); A[v].append(u)
    G = nx.Graph(E)
    for row in S[:300]:
        assert set(row) == set(range(p))
        for d in range(p):
            assert nx.is_connected(G.subgraph([v for v in range(n) if row[v] == d]))


def test_unrank_is_a_bijection():
    z = PartitionZDD(*grid(3, 4), 3)
    L = canonical(z.all_labels())
    assert len(set(L)) == z.count == 1350


def test_farthest_pair_floyd_equals_bfs():
    for n, E in (grid(6, 9), delaunay(40, np.random.default_rng(1))):
        assert farthest_pair(n, E, "floyd")[2] == farthest_pair(n, E, "bfs")[2]


def test_orderings_keep_every_edge_once():
    for n, E in (grid(7, 7), delaunay(60, np.random.default_rng(2)), wheel(12)):
        want = sorted((min(u, v), max(u, v)) for u, v in E)
        assert sorted(order_cut(n, E)) == want
        assert sorted(order_bfs(n, E)) == want


def test_frontier_definition():
    n, E = path(10)
    assert max_frontier(n, E) == 1
    F = frontiers(n, E)
    assert F[-1] == set() and F[0] == {1}
    n, E = grid(5, 9)
    o, name, f, tried = best_order(n, E)
    assert f == min(tried.values()) and f <= 6


def test_population_helpers():
    pop = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9])
    z = PartitionZDD(*grid(3, 3), 3)
    L = z.all_labels()
    P = district_pops(L, pop, 3)
    assert np.allclose(P.sum(1), pop.sum())
    dev = max_deviation(L, pop, 3)
    assert np.allclose(dev, np.abs(P / 15 - 1).max(1))
    assert population_ok(L, pop, 3, 0.0).sum() == (dev == 0).sum()


def test_budget_aborts():
    with pytest.raises(BuildAborted):
        PartitionZDD(*grid(6, 12), 4, max_nodes=10_000)


def test_iowa_county_graph_shape():
    from redistricting.geo import Geo
    g = Geo.load("IA_county", 4)
    G = nx.Graph(list(g.border))
    assert g.N == 99 and nx.is_connected(G) and nx.check_planarity(G)[0]
