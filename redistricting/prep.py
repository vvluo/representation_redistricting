"""Build <ST>_tract and <ST>_county unit files from a TIGER tract shapefile.

Counties are unions of tracts, so county adjacency and shared-border lengths are
sums of tract borders that cross a county line, and a county's perimeter is the sum
of its tracts' perimeters minus twice their internal shared borders.  No county
shapefile is needed.

    python -m redistricting.prep --state 19 --abbr IA \\
        --shp ../TIGER/tl_2026_19_tract/tl_2026_19_tract \\
        --master ../extracted/tract_master.csv --out data

Writes <out>/<abbr>_{tract,county}_{features,borders}.csv.  Features carry the
Geo columns (geoid, name, pop, area_land_km2, area_water_km2, perimeter_km, lat,
lon) plus the count columns the plan statistics need (CVAP by race, BA+ of 25+
population, people below poverty, employed residents and workers in each of the 13 ACS
industry sectors, population by non-metro RUCA tier) and the tract's RUCA code.  Counts aggregate exactly; the county
median_hh_income is a household-weighted proxy (pop-weighted mean of tract medians)
and is labelled as such.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd
from .borders import build_borders
from .shapefile import read_dbf
from .rl.groups import RUCA_TIERS, derive_counts

COUNTS = ["pop", "cvap_total", "cvap_white", "cvap_black", "cvap_hispanic",
          "cvap_asian", "cvap_aian", "cvap_nhpi", "edu_total", "edu_ba_plus",
          "pov_total", "pov_below", "emp_total"] + [f"ind_{k:02d}" for k in range(1, 14)] \
    + [f"pop_{t}" for t in RUCA_TIERS]
# ind_kk = workers (employed residents 16+) in ACS industry sector kk, rebuilt from the
# master's percentage shares so that districts sum them exactly; pop_<tier> = population
# of tracts in each non-metro RUCA tier (see redistricting.rl.groups).


def build_units(shp_base, state, master_csv, acs_csv=None):
    """Return (tract_features, tract_borders, county_features, county_borders).
    acs_csv: optional output of redistricting.acs (extra group counts), merged by GEOID."""
    gid, name, perim, border, _, _ = build_borders(shp_base, state)
    recs = {r["GEOID"]: r for r in read_dbf(shp_base + ".dbf") if r["STATEFP"] == state}
    M = pd.read_csv(master_csv, dtype={"geoid": str, "county_fips": str, "state_fips": str})
    M = derive_counts(M[M.geoid.isin(set(gid))]).set_index("geoid")
    missing = set(gid) - set(M.index)
    if missing:
        raise ValueError(f"{len(missing)} tracts lack a tract_master row, e.g. {sorted(missing)[:3]}")

    rows = []
    for k, g in enumerate(gid):
        r, m = recs[g], M.loc[g]
        row = {"geoid": g, "name": r["NAMELSAD"], "county_fips": g[:5],
               "area_land_km2": int(r["ALAND"]) / 1e6, "area_water_km2": int(r["AWATER"]) / 1e6,
               "perimeter_km": perim[k] / 1000,
               "lat": float(r["INTPTLAT"]), "lon": float(r["INTPTLON"])}
        for c in COUNTS:
            row[c] = float(m[c]) if pd.notna(m[c]) else 0.0
        row["ruca_primary"] = m["ruca_primary"]
        row["median_hh_income"] = m["median_hh_income"]
        rows.append(row)
    T = pd.DataFrame(rows)
    counts = list(COUNTS)
    if acs_csv:
        X = pd.read_csv(acs_csv, dtype={"geoid": str}).set_index("geoid")
        miss = (~T.geoid.isin(X.index)).sum()
        if miss:
            print(f"warning: {miss} tracts have no ACS row; their ACS counts are 0")
        X = X.reindex(T.geoid).fillna(0.0)
        for c in X.columns:
            T[c] = X[c].to_numpy(float)
        counts += list(X.columns)
    TB = pd.DataFrame([{"a": gid[i], "b": gid[j], "shared_km": L / 1000}
                       for (i, j), L in sorted(border.items())])

    # ---- counties: sums of tracts; borders = cross-county tract borders --------
    cty = T.county_fips.tolist()
    internal = defaultdict(float); cb = defaultdict(float)
    for (i, j), L in border.items():
        a, b = cty[i], cty[j]
        if a == b: internal[a] += L / 1000
        else:      cb[(min(a, b), max(a, b))] += L / 1000
    T["_w"] = T.area_land_km2 + T.area_water_km2
    T["_inc"] = T.median_hh_income * T["pop"]
    T["_wlat"] = T.lat * T._w; T["_wlon"] = T.lon * T._w
    C = T.groupby("county_fips")[counts + ["area_land_km2", "area_water_km2",
                                          "perimeter_km", "_inc", "_w", "_wlat", "_wlon"]].sum()
    C["perimeter_km"] = C.perimeter_km - 2 * pd.Series(internal).reindex(C.index).fillna(0)
    C["lat"] = C._wlat / C._w          # area-weighted mean of tract interior points
    C["lon"] = C._wlon / C._w
    C["median_hh_income_proxy"] = C._inc / C["pop"]
    C = C.drop(columns=["_inc", "_w", "_wlat", "_wlon"]).reset_index() \
         .rename(columns={"county_fips": "geoid"})
    C.insert(1, "name", C.geoid)
    CB = pd.DataFrame([{"a": a, "b": b, "shared_km": L} for (a, b), L in sorted(cb.items())])
    T = T.drop(columns=["_w", "_inc", "_wlat", "_wlon"])
    return T, TB, C, CB


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shp", required=True, help="tract shapefile path WITHOUT extension")
    ap.add_argument("--state", required=True, help="2-digit STATEFP")
    ap.add_argument("--abbr", required=True, help="prefix for output files, e.g. IA")
    ap.add_argument("--master", required=True, help="tract_master.csv")
    ap.add_argument("--acs", help="extra ACS group counts from redistricting.acs (optional)")
    ap.add_argument("--out", default="data")
    A = ap.parse_args()
    T, TB, C, CB = build_units(A.shp, A.state, A.master, A.acs)
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"{A.abbr}_tract_features.csv", index=False)
    TB.to_csv(out / f"{A.abbr}_tract_borders.csv", index=False)
    C.to_csv(out / f"{A.abbr}_county_features.csv", index=False)
    CB.to_csv(out / f"{A.abbr}_county_borders.csv", index=False)
    print(f"{A.abbr}: {len(T)} tracts / {len(TB)} tract borders;  "
          f"{len(C)} counties / {len(CB)} county borders")


if __name__ == "__main__":
    main()
