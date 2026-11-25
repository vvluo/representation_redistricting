"""The C++ builder must produce the Python builder's diagram exactly, node for node.

Skipped when no C++ compiler is available.
"""
import shutil
import numpy as np
import pytest

from redistricting.enumerate.zdd import PartitionZDD, BuildAborted
from redistricting.enumerate.bruteforce import (grid, wheel, complete, delaunay, random_tree,
                                                cycle, brute_partitions, canonical)

pytestmark = pytest.mark.skipif(
    not (shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")),
    reason="no C++ compiler")

RNG = np.random.default_rng(11)
GRAPHS = {"grid3x3": grid(3, 3), "trigrid3x4": grid(3, 4, True), "wheel9": wheel(9),
          "K6": complete(6), "delaunay14": delaunay(14, RNG), "tree30": random_tree(30, RNG),
          "cycle40": cycle(40), "grid5x9": grid(5, 9)}


def _same(a, b):
    return (a.nodes == b.nodes and a.count == b.count and len(a.lo) == len(b.lo)
            and all(np.array_equal(x, y) for x, y in zip(a.lo, b.lo))
            and all(np.array_equal(x, y) for x, y in zip(a.hi, b.hi)))


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_cpp_identical_to_python(name, p):
    n, E = GRAPHS[name]
    a = PartitionZDD(n, E, p, order="best")
    b = PartitionZDD(n, E, p, order=a.order, backend="cpp")
    assert _same(a, b)


@pytest.mark.parametrize("name", ["grid3x3", "trigrid3x4", "wheel9", "delaunay14"])
def test_cpp_matches_brute_force(name):
    n, E = GRAPHS[name]
    if n > 12:
        pytest.skip("brute force too slow")
    for p in (2, 3, 4):
        z = PartitionZDD(n, E, p, backend="cpp")
        assert set(canonical(z.all_labels())) == brute_partitions(n, E, p)


def test_iowa_tract_subsets_identical():
    from redistricting.geo import Geo
    from experiments.zdd_scaling import grow_subset
    g = Geo.load("IA_tract", 2)
    for k, size in enumerate([30, 40]):
        idx = grow_subset(g.nbr, size, np.random.default_rng(200 + k))
        pos = {v: i for i, v in enumerate(idx)}
        E = [(pos[a], pos[b]) for a, b in g.border if a in pos and b in pos]
        a = PartitionZDD(len(idx), E, 3)
        assert _same(a, PartitionZDD(len(idx), E, 3, order=a.order, backend="cpp"))


def test_cpp_budget_aborts():
    with pytest.raises(BuildAborted):
        PartitionZDD(*grid(6, 12), 4, max_nodes=10_000, backend="cpp")
