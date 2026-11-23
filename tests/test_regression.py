"""Golden values recorded from the original scratch implementation (before it was
reorganised into this package), at fixed seeds.  If a refactor changes any of these,
it changed behaviour, not just structure.  Digests are of the final assignment arrays."""
import hashlib, random
import numpy as np
import pytest
from redistricting import Plan, make_init, hill_climb, recom_climb, hybrid
from redistricting.inits import seeded_growth


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype=int).tobytes()).hexdigest()[:16]


# init -> steepest flips, seed 6792, restart index -> (start, final, flips)
INIT_FLIP = {
    "random": [("c9df67768b0dd8bc", "53f37578c633b057", 33), ("83063552230dcab1", "d512c906b00afc2d", 51)],
    "pop_greedy": [("3c5735de9785720f", "2f8387881bbdb6ea", 20), ("96a618806ce9c4ce", "dba60c6b91170bf9", 42)],
    "sweep": [("bb550b9f95f0d7ac", "f130b6c4ad30344d", 5), ("0ddba794e95927e6", "bec2727d4b7a3ba8", 12)],
    "kmeans": [("e961cdff4fbe7d44", "d1f3cdf7f6b9832f", 14), ("6e351a702e15cb8b", "863d8f72b81b8fd6", 24)],
    "spectral": [("84dd006e4a5501be", "a2a61b34e5eb3785", 8), ("20be29e07ce60a8f", "2b5e915db81012da", 9)],
}

# ReCom per tree sampler, seed 4242 -> (final, steps, proposals)
RECOM = {
    "ust": ("616dae701a90db81", 23, 322),
    "rmst": ("feb013f3a48d4672", 39, 528),
    "compact": ("1c6eab5523fd5677", 16, 218),
}

HYBRID_COUNTY = ("5dd25ea6196d0ce1", 25, 0)
HYBRID_TRACT = ("466c88f654f4bb40", 47, 11)


@pytest.mark.parametrize("name", sorted(INIT_FLIP))
def test_init_then_flips_matches_original(county, name):
    rng = random.Random(6792)
    for want_init, want_final, want_flips in INIT_FLIP[name]:
        a0 = make_init(name, county, rng)
        p, it = hill_climb(Plan(county, a0, 6.0, "tvd"))
        assert (digest(a0), digest(p.a), it) == (want_init, want_final, want_flips)


@pytest.mark.parametrize("tree", sorted(RECOM))
def test_recom_matches_original(county, tree):
    rng = random.Random(4242)
    p0 = Plan(county, seeded_growth(county, rng), 6.0, "tvd")
    p, steps, props = recom_climb(p0, rng, tree=tree, tol=0.02)
    assert (digest(p.a), steps, props) == RECOM[tree]


def test_hybrid_matches_original_on_counties(county):
    rng = random.Random(5150)
    p, rr, ff = hybrid(Plan(county, seeded_growth(county, rng), 6.0, "tvd"), rng,
                       tree="compact", tol=0.01)
    assert (digest(p.a), rr, ff) == HYBRID_COUNTY


def test_hybrid_matches_original_on_tracts(tract):
    rng = random.Random(5150)
    p, rr, ff = hybrid(Plan(tract, seeded_growth(tract, rng), 6.0, "tvd"), rng,
                       tree="compact", tol=0.01)
    assert (digest(p.a), rr, ff) == HYBRID_TRACT
