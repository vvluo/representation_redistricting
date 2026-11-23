from redistricting import audit
from .conftest import needs_tiger, tiger_shp


def test_graph_level_invariants_hold_on_counties():
    assert audit.run("AL_county", nsub=30, nplans=2, nfuzz=300, verbose=False) == []


def test_graph_level_invariants_hold_on_tracts():
    assert audit.run("AL_tract", nsub=20, nplans=2, nfuzz=300, verbose=False) == []


@needs_tiger
def test_raw_geometry_matches_the_shipped_csvs_on_counties():
    assert audit.run("AL_county", nsub=30, nplans=2, nfuzz=100, verbose=False,
                     shp_base=tiger_shp("tl_2026_us_county"), state="01",
                     expected_outer_km=1789.1) == []


@needs_tiger
def test_raw_geometry_matches_the_shipped_csvs_on_tracts():
    assert audit.run("AL_tract", nsub=20, nplans=2, nfuzz=100, verbose=False,
                     shp_base=tiger_shp("tl_2026_01_tract"), state="01",
                     expected_outer_km=1789.1) == []
