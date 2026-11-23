import random
import pytest
from redistricting import Geo, Plan, verify, INITS, make_init, config

NAMES = sorted(INITS)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_init_returns_a_valid_plan_on_counties(county, name, seed):
    a = make_init(name, county, random.Random(seed))
    assert len(a) == county.N
    assert set(a) == set(range(county.D)), "every district must be non-empty"
    assert not verify(Plan(county, a, 6.0, "tvd")), "districts must be contiguous"


@pytest.mark.parametrize("name", NAMES)
def test_every_init_returns_a_valid_plan_on_tracts(tract, name):
    a = make_init(name, tract, random.Random(7))
    assert set(a) == set(range(tract.D))
    assert not verify(Plan(tract, a, 6.0, "tvd"))


@pytest.mark.parametrize("name", NAMES)
def test_init_is_deterministic_for_a_seed(county, name):
    a = make_init(name, county, random.Random(11))
    b = make_init(name, county, random.Random(11))
    assert list(a) == list(b)


def test_unknown_init_is_rejected(county):
    with pytest.raises(ValueError, match="unknown init"):
        make_init("nope", county, random.Random(0))


@pytest.mark.parametrize("name", ["sweep", "kmeans"])
def test_coordinate_inits_explain_missing_lat_lon(tmp_path, name):
    import pandas as pd
    d = config.data_dir()
    f = pd.read_csv(d / "AL_county_features.csv",
                    dtype={"geoid": str}).drop(columns=["lat", "lon"])
    f.to_csv(tmp_path / "X_features.csv", index=False)
    (tmp_path / "X_borders.csv").write_bytes((d / "AL_county_borders.csv").read_bytes())
    g = Geo.load("X", D=7, data_dir=tmp_path)
    with pytest.raises(ValueError, match="lat/lon"):
        make_init(name, g, random.Random(0))
    # the graph-only inits must still work on the same data
    assert not verify(Plan(g, make_init("spectral", g, random.Random(0)), 6.0, "tvd"))
