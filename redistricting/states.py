"""Per-state presets and data: any state, congressional or state-house districts.

    g, p, tol = load_state("IA", "congress")       # Geo, 4 districts, 1% tolerance
    g, p, tol = load_state("IA", "house")          # Geo, 100 districts, 5% tolerance

District counts
  congress  seats from the 2020 apportionment (2022-2030 elections); sums to 435
  house     number of state-house (lower chamber) DISTRICTS, not seats: multi-member
            states draw fewer districts than they have seats (AZ 30 for 60 seats, NJ
            40/80, WA 49/98, NH ~203/400).  `house_verify` marks states whose
            sub-district count should be double-checked against the current plan.
            Nebraska's unicameral legislature counts as its house.

Tolerance  max |pop_d / ideal - 1|: 1% for congress (the tightest tracts reliably
           allow), 5% for the state house.

Units      tracts.  A state-house map needs enough tracts per district for the
           tolerance to be reachable; below MIN_UNITS_PER_DISTRICT the loader warns
           (19 states for the house; block groups are the planned fix), below
           HARD_MIN it refuses.

Data       built on first use by redistricting.prep from the TIGER tract shapefile and
           the tract master (paths from REDISTRICTING_TIGER / REDISTRICTING_MASTER, or
           ../TIGER and ../extracted/tract_master.csv next to the repo), into
           data/states/ (git-ignored; existing files in data/ are never overwritten).  Tract graphs
           with islands (HI, CA's Channel Islands, ...) are bridged: each extra
           component is joined to its nearest unit by a zero-length border, so plans
           can be contiguous.

    python -m redistricting.states build IA          # write data/IA_{tract,county}_*
    python -m redistricting.states list
"""
import argparse, math, os, warnings
from pathlib import Path
import numpy as np
from . import config
from .geo import Geo

#            fips  congress  house_districts  house_seats  house_verify
PRESETS = {
    "AL": ("01", 7, 105, 105, False), "AK": ("02", 1, 40, 40, False),
    "AZ": ("04", 9, 30, 60, False), "AR": ("05", 4, 100, 100, False),
    "CA": ("06", 52, 80, 80, False), "CO": ("08", 8, 65, 65, False),
    "CT": ("09", 5, 151, 151, False), "DE": ("10", 1, 41, 41, False),
    "FL": ("12", 28, 120, 120, False), "GA": ("13", 14, 180, 180, False),
    "HI": ("15", 2, 51, 51, False), "ID": ("16", 2, 35, 70, False),
    "IL": ("17", 17, 118, 118, False), "IN": ("18", 9, 100, 100, False),
    "IA": ("19", 4, 100, 100, False), "KS": ("20", 4, 125, 125, False),
    "KY": ("21", 6, 100, 100, False), "LA": ("22", 6, 105, 105, False),
    "ME": ("23", 2, 151, 151, False), "MD": ("24", 8, 71, 141, True),
    "MA": ("25", 9, 160, 160, False), "MI": ("26", 13, 110, 110, False),
    "MN": ("27", 8, 134, 134, False), "MS": ("28", 4, 122, 122, False),
    "MO": ("29", 8, 163, 163, False), "MT": ("30", 2, 100, 100, False),
    "NE": ("31", 3, 49, 49, False), "NV": ("32", 4, 42, 42, False),
    "NH": ("33", 2, 203, 400, True), "NJ": ("34", 12, 40, 80, False),
    "NM": ("35", 3, 70, 70, False), "NY": ("36", 26, 150, 150, False),
    "NC": ("37", 14, 120, 120, False), "ND": ("38", 1, 49, 94, True),
    "OH": ("39", 15, 99, 99, False), "OK": ("40", 5, 101, 101, False),
    "OR": ("41", 6, 60, 60, False), "PA": ("42", 17, 203, 203, False),
    "RI": ("44", 2, 75, 75, False), "SC": ("45", 7, 124, 124, False),
    "SD": ("46", 1, 37, 70, True), "TN": ("47", 9, 99, 99, False),
    "TX": ("48", 38, 150, 150, False), "UT": ("49", 4, 75, 75, False),
    "VT": ("50", 1, 109, 150, True), "VA": ("51", 11, 100, 100, False),
    "WA": ("53", 10, 49, 98, False), "WV": ("54", 2, 100, 100, False),
    "WI": ("55", 8, 99, 99, False), "WY": ("56", 1, 62, 62, False),
}
TOLERANCE = {"congress": 0.01, "house": 0.05}
MIN_UNITS_PER_DISTRICT = 10
HARD_MIN = 3


def districts(abbr, level):
    fips, cong, hd, hs, verify = PRESETS[abbr]
    if level == "congress":
        return cong
    if level == "house":
        if verify:
            warnings.warn(f"{abbr}: house sub-district count {hd} should be checked against the current plan")
        return hd
    raise ValueError("level must be 'congress' or 'house'")


def _paths():
    root = config.REPO_ROOT.parent
    tiger = config.tiger_dir() or root / "TIGER"
    master = Path(os.environ.get("REDISTRICTING_MASTER", root / "extracted" / "tract_master.csv"))
    return Path(tiger), master


def build(abbr, acs=None, data_dir=None):
    """Write <abbr>_{tract,county}_{features,borders}.csv with redistricting.prep."""
    from .prep import build_units
    fips = PRESETS[abbr][0]
    tiger, master = _paths()
    cands = sorted(tiger.glob(f"tl_*_{fips}_tract"))
    if not cands:
        raise FileNotFoundError(f"no TIGER tract shapefile for {abbr} under {tiger}")
    shp = cands[-1] / cands[-1].name
    T, TB, C, CB = build_units(str(shp), fips, str(master), acs)
    out = Path(data_dir) if data_dir else config.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"{abbr}_tract_features.csv", index=False)
    TB.to_csv(out / f"{abbr}_tract_borders.csv", index=False)
    C.to_csv(out / f"{abbr}_county_features.csv", index=False)
    CB.to_csv(out / f"{abbr}_county_borders.csv", index=False)
    return out


def bridge(g):
    """Join every connected component to the rest by a zero-length border between its
    unit and the nearest unit (by centroid) outside it.  Returns the added pairs."""
    added = []
    while True:
        comp = np.full(g.N, -1); k = 0
        for s in range(g.N):
            if comp[s] >= 0:
                continue
            comp[s] = k; st = [s]
            while st:
                x = st.pop()
                for y in g.nbr[x]:
                    if comp[y] < 0:
                        comp[y] = k; st.append(y)
            k += 1
        if k <= 1:
            return added
        sizes = np.bincount(comp)
        c = int(np.argmin(sizes))                     # smallest component first
        inside = np.flatnonzero(comp == c); outside = np.flatnonzero(comp != c)
        xy = g.xy
        d = ((xy[inside, None, :] - xy[None, outside, :]) ** 2).sum(-1)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        i, j = int(inside[a]), int(outside[b])
        nb = [list(v) for v in g.nbr]; nb[i].append(j); nb[j].append(i)
        g.nbr = [tuple(v) for v in nb]
        g.border[(min(i, j), max(i, j))] = 0.0
        added.append((g.gid[i], g.gid[j], float(math.sqrt(d[a, b]))))


def _unit_dir(abbr, data_dir=None):
    """Directory holding <abbr>_tract files WITH group counts: data_dir if given, else
    data/ (e.g. IA) then data/states/ (built on demand).  Files without the counts
    (the reduced AL files the regression tests pin) are left alone."""
    dirs = [Path(data_dir)] if data_dir else [config.data_dir(), config.data_dir() / "states"]
    for d in dirs:
        f = d / f"{abbr}_tract_features.csv"
        if f.exists() and "pov_total" in open(f).readline():
            return d
    return None


def load_state(abbr, level="congress", p=None, tol=None, data_dir=None, build_missing=True):
    """(Geo, p, tol) for a state at a level; builds the unit files if missing."""
    abbr = abbr.upper()
    if abbr not in PRESETS:
        raise ValueError(f"unknown state {abbr!r}")
    p = p or districts(abbr, level)
    if p < 2:
        raise ValueError(f"{abbr} has {p} {level} district(s): nothing to draw")
    tol = TOLERANCE[level] if tol is None else tol
    d = _unit_dir(abbr, data_dir)
    if d is None:
        if not build_missing:
            raise FileNotFoundError(f"{abbr} unit files with group counts not found")
        d = build(abbr, data_dir=Path(data_dir) if data_dir else config.data_dir() / "states")
    g = Geo.load(f"{abbr}_tract", D=p, data_dir=d)
    g.bridges = bridge(g)
    if g.bridges:
        warnings.warn(f"{abbr}: joined {len(g.bridges)} island component(s) by zero-length borders")
    per = int((g.pop > 0).sum()) / p
    if per < HARD_MIN:
        raise ValueError(f"{abbr} {level}: {per:.1f} populated tracts per district; tracts are too coarse")
    if per < MIN_UNITS_PER_DISTRICT:
        warnings.warn(f"{abbr} {level}: only {per:.1f} populated tracts per district; the "
                      f"{tol:.0%} tolerance may be unreachable (block groups planned)")
    return g, p, tol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("states", nargs="+"); b.add_argument("--acs")
    sub.add_parser("list")
    A = ap.parse_args()
    if A.cmd == "build":
        for s in A.states:
            print(s, "->", build(s.upper(), A.acs))
    else:
        print(f"{'st':3s} {'fips':4s} {'congress':>8s} {'house':>6s} {'seats':>6s}  verify")
        for s, (f, c, hd, hs, v) in PRESETS.items():
            print(f"{s:3s} {f:4s} {c:8d} {hd:6d} {hs:6d}  {'yes' if v else ''}")


if __name__ == "__main__":
    main()
