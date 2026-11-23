import random
import pytest
from redistricting import (Plan, verify, objective, hill_climb, hill_climb_first,
                           recom_climb, hybrid, TREE_MODES, make_init)
from redistricting.inits import seeded_growth
from redistricting.search import spanning_tree, cuts, adjacent_pairs


def start(g, seed, lam=6.0):
    return Plan(g, seeded_growth(g, random.Random(seed)), lam, "tvd")


def no_improving_flip(p):
    cur = p.score()
    return not any(p.valid(i, d) and p.score_after(i, d) - cur > 1e-12
                   for i, d in p.candidates())


def test_objective_is_zero_penalty_at_exact_uniformity():
    import numpy as np
    pop = np.full(7, 100.0); area = np.full(7, 3.0); per = np.full(7, 2.0)
    for mode in ("tvd", "rmsd", "entropy"):
        assert objective(pop, area, per, 100.0, 7, 5.0, mode) == pytest.approx(
            4 * 3.14159265358979 * 3.0 / 4.0, rel=1e-9)


def test_plan_score_is_the_shared_objective(county):
    p = start(county, 1)
    assert p.score() == objective(p.pop, p.area, p.perim, county.ideal, county.D, 6.0, "tvd")


@pytest.mark.parametrize("climb", [hill_climb, lambda p: hill_climb_first(p, random.Random(0))])
def test_flip_climb_improves_stays_valid_and_ends_at_a_local_optimum(county, climb):
    p = start(county, 3)
    s0 = p.score()
    p, it = climb(p)
    assert it > 0 and p.score() > s0
    assert not verify(p)
    assert no_improving_flip(p)


@pytest.mark.parametrize("tree", TREE_MODES)
def test_recom_improves_and_stays_valid(county, tree):
    rng = random.Random(5)
    p = Plan(county, seeded_growth(county, rng), 6.0, "tvd"); s0 = p.score()
    p, steps, props = recom_climb(p, rng, tree=tree, tol=0.02)
    assert steps > 0 and props >= steps
    assert p.score() > s0
    assert not verify(p)


@pytest.mark.parametrize("tree", TREE_MODES)
def test_every_cut_splits_the_union_into_two_connected_halves(county, tree):
    rng = random.Random(9)
    p = start(county, 9)
    for d1, d2 in adjacent_pairs(p)[:6]:
        union = p.members[d1] | p.members[d2]
        par = spanning_tree(county, union, rng, tree)
        assert set(par) == union
        assert sum(v is None for v in par.values()) == 1            # one root
        for A, B in cuts(county, par, 0.02, county.ideal):
            assert A and B and not (A & B) and (A | B) == union
            for half in (A, B):
                q = Plan(county, p.a.copy(), 6.0, "tvd")
                q.apply_split(d1, d2, A, B)
                assert not verify(q)


def test_hybrid_is_never_worse_than_the_recom_phase_it_starts_with(county):
    """hybrid's first phase consumes the rng exactly as recom_climb does, and the
    flips that follow only ever accept improvements."""
    a = recom_climb(start(county, 21), random.Random(8), tree="compact", tol=0.01)[0]
    h, rr, ff = hybrid(start(county, 21), random.Random(8), tree="compact", tol=0.01)
    assert h.score() >= a.score() - 1e-12
    assert not verify(h)


def test_hybrid_handles_a_thousand_units(tract):
    rng = random.Random(2)
    h, rr, ff = hybrid(Plan(tract, seeded_growth(tract, rng), 6.0, "tvd"), rng,
                       tree="compact", tol=0.01)
    assert not verify(h)
    assert h.pp().mean() > 0.30
    assert 200 * h.report()["tvd"] < 5.0          # < 5% of population misplaced
