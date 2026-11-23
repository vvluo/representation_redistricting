"""Where data lives.  Nothing in this package hard-codes an absolute path.

REDISTRICTING_DATA   directory holding <name>_features.csv / <name>_borders.csv
                     (default: <repo>/data)
REDISTRICTING_TIGER  optional: directory of raw TIGER/Line shapefile folders; only
                     needed to regenerate borders or run the raw-geometry audit checks
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return Path(os.environ.get("REDISTRICTING_DATA", REPO_ROOT / "data"))


def tiger_dir():
    v = os.environ.get("REDISTRICTING_TIGER")
    return Path(v) if v else None
