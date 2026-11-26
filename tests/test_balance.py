"""Starter-map balancing: keeps contiguity, never worsens imbalance, reaches tolerance."""
import random
import numpy as np
import networkx as nx
import pytest
from redistricting.geo import Geo
from redistricting.inits import make_init
from redistricting.inits.balance import balance
from redistricting import stats


@pytest.mark.parametrize("init", ["random", "kmeans", "spectral"])
def test_balance_iowa_counties(init):
    g = Geo.load("IA_county", 4); G = nx.Graph(list(g.border))
    hits = 0
    for s in range(5):
        rng = random.Random(s)
        a = np.array(make_init(init, g, rng))
        before = stats.pop_dev(a[None], g, 4)[0]
        b, steps, dev = balance(g, a, rng, tol=0.05)
        assert dev <= before + 1e-12
        assert np.isclose(dev, stats.pop_dev(b[None], g, 4)[0])
        for d in range(4):
            assert nx.is_connected(G.subgraph(np.flatnonzero(b == d)))
        hits += dev <= 0.05
    assert hits >= 4
