import pytest
from redistricting import Geo, config


@pytest.fixture(scope="session")
def county():
    return Geo.load("AL_county", D=7)


@pytest.fixture(scope="session")
def tract():
    return Geo.load("AL_tract", D=7)


def tiger_shp(name):
    """Path (no extension) of a raw TIGER shapefile, or None if REDISTRICTING_TIGER is unset."""
    d = config.tiger_dir()
    return None if d is None else str(d / name / name)


needs_tiger = pytest.mark.skipif(
    config.tiger_dir() is None, reason="set REDISTRICTING_TIGER to run raw-geometry checks")
