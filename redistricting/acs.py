"""Extra ACS 5-year tract tables for the protected-group families.

Fetches the tables below from the Census Data API and derives one count column per
group numerator / denominator.  Variable codes were checked against the 2024 ACS
5-year group metadata (api.census.gov/data/2024/acs/acs5/groups/<table>.json).

  column             table    definition
  hh_inc_total       B19001   households                                     _001
  hh_inc_lt35k       B19001   income under $35,000                           _002.._007
  hh_inc_ge150k      B19001   income $150,000 or more                        _016 + _017
  cow_total          B24080   civilian employed 16+                          _001
  cow_government     B24080   local + state + federal government, M + F      _007-009, _017-019
  cow_self_employed  B24080   self-employed, incorporated or not, M + F      _005, _010, _015, _020
  cow_nonprofit      B24080   private not-for-profit, M + F                  _006, _016
  edu_lt_hs          B15003   25+ without a high-school diploma or GED       _002.._016
  hh_occupied        B25003   occupied housing units                         _001
  hh_renter          B25003   renter occupied                                _003
  rent_total         B25070   renter households with rent burden computed    _001 - _011
  rent_burden30      B25070   gross rent >= 30% of household income          _007.._010
  hu_total           B25002   housing units                                  _001
  hu_vacant          B25002   vacant                                         _003
  hi_total           B27010   civilian non-institutionalised population      _001
  hi_uninsured       B27010   no health insurance coverage, all ages         _017, _033, _050, _066
  hh_total           B22003   households                                     _001
  snap_hh            B22003   received Food Stamps/SNAP in past 12 months    _002
  lang_total         B16007   population 5+                                  _001
  lang_spanish       B16007   speak Spanish at home, all ages                _004, _010, _016
  lang_indo_european B16007   other Indo-European languages                  _005, _011, _017
  lang_asian_pacific B16007   Asian and Pacific Island languages             _006, _012, _018
  lang_other         B16007   other languages                                _007, _013, _019
  gq_pop_total       B26103   total population                               _001
  gq_correctional    B26103   adult correctional facilities                  _004
  gq_nursing         B26103   nursing / skilled-nursing facilities           _005
  gq_college         B26103   college / university student housing           _007

Two ways in:
  * the Census Data API, which needs a free key
    (https://api.census.gov/data/key_signup.html) read from CENSUS_API_KEY;
  * data.census.gov downloads: each table with Geography = "All Census Tracts within
    <state>", product "ACS 5-Year Estimates Detailed Tables", saved with the "ZIP: Flat,
    machine-readable .csv files including GEOIDs" option.  Point --files at the ZIPs
    (or the extracted *-Data.csv files).

    python -m redistricting.acs --state 19 --out ../extracted/acs_api/IA_2024.csv
    python -m redistricting.acs --files ~/Downloads/ACSDT5Y2024.*.zip --out ../extracted/acs_api/IA_2024.csv
    python -m redistricting.acs --state all --out ../extracted/acs_api/US_2024.csv   # national
    python -m redistricting.acs --state all --p5 --out ../extracted/acs_api/US_2020_P5.csv

Files may be split any way (one ZIP per table per state, or several of each); rows
stack across states and columns join across tables.  For national thresholds:
    python -m redistricting.rl.groups --master ../extracted/tract_master.csv \
        --acs ../extracted/acs_api/US_2024.csv
    python -m redistricting.prep ... --acs ../extracted/acs_api/IA_2024.csv
"""
import argparse, glob, io, json, os, sys, urllib.parse, urllib.request, zipfile
from pathlib import Path
import pandas as pd


def _v(table, *nums):
    return [f"{table}_{n:03d}E" for n in nums]


# column -> (variables added, variables subtracted)
SPEC = {
    "hh_inc_total": (_v("B19001", 1), []),
    "hh_inc_lt35k": (_v("B19001", *range(2, 8)), []),
    "hh_inc_ge150k": (_v("B19001", 16, 17), []),
    "cow_total": (_v("B24080", 1), []),
    "cow_government": (_v("B24080", 7, 8, 9, 17, 18, 19), []),
    "cow_self_employed": (_v("B24080", 5, 10, 15, 20), []),
    "cow_nonprofit": (_v("B24080", 6, 16), []),
    "edu_lt_hs": (_v("B15003", *range(2, 17)), []),
    "edu_total_b15003": (_v("B15003", 1), []),          # cross-check against edu_total
    "hh_occupied": (_v("B25003", 1), []),
    "hh_renter": (_v("B25003", 3), []),
    "rent_total": (_v("B25070", 1), _v("B25070", 11)),
    "rent_burden30": (_v("B25070", 7, 8, 9, 10), []),
    "hu_total": (_v("B25002", 1), []),
    "hu_vacant": (_v("B25002", 3), []),
    "hi_total": (_v("B27010", 1), []),
    "hi_uninsured": (_v("B27010", 17, 33, 50, 66), []),
    "hh_total": (_v("B22003", 1), []),
    "snap_hh": (_v("B22003", 2), []),
    "lang_total": (_v("B16007", 1), []),
    "lang_spanish": (_v("B16007", 4, 10, 16), []),
    "lang_indo_european": (_v("B16007", 5, 11, 17), []),
    "lang_asian_pacific": (_v("B16007", 6, 12, 18), []),
    "lang_other": (_v("B16007", 7, 13, 19), []),
    "gq_pop_total": (_v("B26103", 1), []),
    "gq_correctional": (_v("B26103", 4), []),
    "gq_nursing": (_v("B26103", 5), []),
    "gq_college": (_v("B26103", 7), []),
}
TABLES = sorted({v.split("_")[0] for add, sub in SPEC.values() for v in add + sub})


# Tables the 5-year release does not publish for tracts (B26103 stops at states in 2024);
# the API would reject a whole request containing them.  Tract group-quarters counts
# need the 2020 decennial table P5 instead.
NO_TRACT = {"B26103"}


def variables(tract=True):
    return sorted({v for add, sub in SPEC.values() for v in add + sub
                   if not (tract and v.split("_")[0] in NO_TRACT)})


STATES = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17",
          "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
          "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
          "47", "48", "49", "50", "51", "53", "54", "55", "56", "72"]   # 50 states, DC, PR


# Group quarters for tracts: 2020 Census redistricting file (PL 94-171), table P5, a
# full count.  Same column names as the B26103 groups, so redistricting.rl.groups picks
# them up unchanged; gq_pop_total is the tract's total population (as B26103_001 is).
P5_SPEC = {
    "gq_pop_total": "P1_001N",       # total population
    "gq_total": "P5_001N",           # all group quarters
    "gq_correctional": "P5_003N",    # correctional facilities for adults
    "gq_nursing": "P5_005N",         # nursing / skilled-nursing facilities
    "gq_college": "P5_008N",         # college / university student housing
    "gq_military": "P5_009N",        # military quarters
}


def fetch_p5(state, key=None):
    """2020 PL 94-171 group-quarters counts for every tract in a state."""
    raw = fetch(state, key=key, dataset="2020/dec/pl", V=sorted(set(P5_SPEC.values())))
    out = raw[["geoid"]].copy()
    for col, v in P5_SPEC.items():
        out[col] = raw[v].where(raw[v] >= 0)
    return out


def remap_ct(D, current_geoids):
    """Re-key Connecticut tracts from 2020 county codes to the 2022 planning-region codes
    used by ACS 2022+ and TIGER 2022+.  The 6-digit tract number was kept, so match on it
    where it is unique (the duplicates are zero-population water tracts)."""
    cur = pd.Series([g for g in current_geoids if g.startswith("09")])
    cur = cur[~cur.str[5:].duplicated(keep=False)]
    new = dict(zip(cur.str[5:], cur))
    ct = D.geoid.str.startswith("09")
    tr = D.loc[ct, "geoid"].str[5:]
    dup = tr.duplicated(keep=False)
    D = D.copy()
    D.loc[ct, "geoid"] = [new.get(t, g) if not d else g for t, g, d in zip(tr, D.loc[ct, "geoid"], dup)]
    return D


def _tiger_geoids(fips):
    """Current tract GEOIDs of one state from ../TIGER (empty if not there)."""
    from redistricting.shapefile import read_dbf
    d = sorted((Path(__file__).resolve().parents[2] / "TIGER").glob(f"tl_*_{fips}_tract"))
    return [r["GEOID"] for r in read_dbf(str(d[-1] / d[-1].name) + ".dbf")] if d else []


def fetch(state, year=2024, key=None, chunk=45, dataset=None, V=None):
    """Raw estimates for every tract in a state: DataFrame with geoid + variables.
    dataset defaults to the ACS 5-year release of `year`; V to the SPEC variables."""
    key = key or os.environ.get("CENSUS_API_KEY")
    if not key:
        raise SystemExit("set CENSUS_API_KEY (free key: https://api.census.gov/data/key_signup.html)")
    V = V or variables(); out = None
    dataset = dataset or f"{year}/acs/acs5"
    for k in range(0, len(V), chunk):
        q = {"get": ",".join(V[k:k + chunk]), "for": "tract:*", "in": f"state:{int(state):02d}", "key": key}
        url = f"https://api.census.gov/data/{dataset}?" + urllib.parse.urlencode(q, safe=":*,")
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read().decode()
        if not body.startswith("["):
            raise RuntimeError(f"Census API returned no data for state {state} ({body[:120]!r})")
        rows = json.loads(body)
        D = pd.DataFrame(rows[1:], columns=rows[0])
        D["geoid"] = D.state + D.county + D.tract
        D = D.drop(columns=["state", "county", "tract"]).set_index("geoid")
        out = D if out is None else out.join(D)
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.reset_index()


def _read_data_csv(f):
    """One data.census.gov flat CSV: row 1 = variable codes, row 2 = labels."""
    D = pd.read_csv(f, skiprows=[1], dtype=str)
    if "GEO_ID" not in D:
        raise ValueError("not a data.census.gov flat file (no GEO_ID column)")
    tract = D.GEO_ID.str.startswith("1400000US")
    if not tract.any():
        raise ValueError("no tract rows: set Geography to 'All Census Tracts within <state>' "
                         f"(first GEO_ID is {D.GEO_ID.iloc[0]!r})")
    D = D[tract].copy()
    D["geoid"] = D.GEO_ID.str.slice(9)
    est = [c for c in D.columns if c.endswith("E") and c[:1] in "BC" and "_" in c]
    out = D[["geoid"] + est].set_index("geoid")
    return out.apply(pd.to_numeric, errors="coerce")       # '-', 'N', '(X)' -> NaN


def load_files(paths):
    """Raw estimates from data.census.gov ZIPs or *-Data.csv files, joined on GEOID."""
    frames = []
    for p in paths:
        if str(p).endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith("-Data.csv"):
                        frames.append(_read_data_csv(io.BytesIO(z.read(n))))
        else:
            frames.append(_read_data_csv(p))
    if not frames:
        raise ValueError("no *-Data.csv found in the given files")
    # combine_first unions rows and columns: per-state files of one table stack,
    # different tables join on GEOID, and an overlap keeps the first file's value
    R = frames[0]
    for F in frames[1:]:
        R = R.combine_first(F)
    missing = sorted(set(variables()) - set(R.columns))
    if missing:
        tabs = sorted({m.split("_")[0] for m in missing})
        raise ValueError(f"missing tables {tabs} (e.g. {missing[:3]})")
    return R.reset_index()


def derive(raw):
    """Group count columns from raw estimates.  Negative API sentinels (-666666666 for
    not available) are treated as missing, then as 0 in sums."""
    R = raw.set_index("geoid")
    R = R.where(R >= 0)
    out = pd.DataFrame(index=R.index)
    for col, (add, sub) in SPEC.items():
        if not set(add + sub) <= set(R.columns):
            continue                                    # table not fetched (e.g. NO_TRACT)
        v = R[add].sum(axis=1, min_count=1)
        if sub:
            v = v - R[sub].fillna(0).sum(axis=1)
        out[col] = v.clip(lower=0)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--state", nargs="+", help="2-digit STATEFP(s) or 'all', fetched from the API")
    src.add_argument("--files", nargs="+", help="data.census.gov ZIP / *-Data.csv downloads")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", help="also keep the raw variables")
    ap.add_argument("--p5", action="store_true",
                    help="fetch 2020 Census P5 group quarters (tracts) instead of the ACS tables")
    A = ap.parse_args()
    if A.p5:
        if not A.state:
            raise SystemExit("--p5 needs --state")
        states = STATES if A.state == ["all"] else A.state
        D = pd.concat([fetch_p5(s) for s in states], ignore_index=True)
        if (D.geoid.str[:2] == "09").any():
            cur = _tiger_geoids("09")
            if cur:
                D = remap_ct(D, cur)
            else:
                print("warning: no ../TIGER Connecticut tracts; CT keeps 2020 county codes")
        Path(A.out).parent.mkdir(parents=True, exist_ok=True)
        D.to_csv(A.out, index=False)
        print(f"{len(D)} tracts (2020 counts; CT re-keyed to planning regions), {len(P5_SPEC)} P5 columns -> {A.out}")
        return
    if A.files:
        paths = [q for p in A.files for q in (glob.glob(os.path.expanduser(p)) or [p])]
        raw = load_files(paths)
    else:
        states = STATES if A.state == ["all"] else A.state
        raw = pd.concat([fetch(s, A.year) for s in states], ignore_index=True)
    if A.raw_out:
        raw.to_csv(A.raw_out, index=False)
    D = derive(raw)
    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(A.out, index=False)
    got = sorted({v.split("_")[0] for v in raw.columns if "_" in v})
    print(f"{len(D)} tracts, {len(D.columns) - 1} columns from {', '.join(got)} -> {A.out}")


if __name__ == "__main__":
    main()
