"""ACS group counts derived from raw API estimates."""
import numpy as np
import pytest
import pandas as pd
from redistricting.acs import SPEC, derive, variables


def test_derive_sums_and_sentinels():
    V = variables(tract=False)                      # every table, incl. state-only B26103
    raw = pd.DataFrame([{**{v: 1.0 for v in V}, "geoid": "19001000100"},
                        {**{v: 2.0 for v in V}, "geoid": "19001000200"}])
    raw.loc[1, "B25070_011E"] = 5.0                 # 'not computed' renters
    raw.loc[1, "B19001_002E"] = -666666666          # API sentinel -> missing
    D = derive(raw).set_index("geoid")
    assert D.loc["19001000100", "hh_inc_lt35k"] == 6          # _002.._007
    assert D.loc["19001000200", "hh_inc_lt35k"] == 10         # sentinel dropped
    assert D.loc["19001000100", "cow_government"] == 6        # 3 levels x 2 sexes
    assert D.loc["19001000100", "edu_lt_hs"] == 15            # _002.._016
    assert D.loc["19001000200", "rent_total"] == 0            # 2 - 5, clipped
    assert D.loc["19001000100", "hi_uninsured"] == 4          # four age groups
    assert D.loc["19001000100", "lang_spanish"] == 3          # three age bands
    assert D.loc["19001000100", "gq_college"] == 1
    assert set(D.columns) == set(SPEC)
    # tract fetches leave out B26103; its columns are skipped, not an error
    T = derive(raw[["geoid"] + variables()]).set_index("geoid")
    assert "gq_college" not in T and "snap_hh" in T


def test_every_numerator_has_its_denominator():
    from redistricting.rl.groups import default_groups
    cols = list(SPEC) + ["pop", "edu_total"]
    fams = {g.family for g in default_groups(cols)}
    assert {"income", "class_of_worker", "housing", "health", "food", "education",
            "language", "group_quarters"} <= fams


def _flat_csv(table, geoids, value, geo_prefix="1400000US"):
    from redistricting.acs import variables
    V = [v for v in variables() if v.startswith(table)]
    head = ["GEO_ID", "NAME"] + sum([[v, v[:-1] + "M"] for v in V], [])
    lab = ["Geography", "Geographic Area Name"] + ["Estimate!!x", "Margin of Error!!x"] * len(V)
    rows = [[geo_prefix + g, f"Tract {g}"] + sum([[str(value), "12"] for _ in V], []) for g in geoids]
    return "\n".join(",".join(r) for r in [head, lab] + rows) + "\n"


def test_load_datacensus_zips(tmp_path):
    import zipfile
    from redistricting.acs import TABLES, load_files
    import pytest
    paths = []
    for t in TABLES:
        p = tmp_path / f"ACSDT5Y2024.{t}.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(f"ACSDT5Y2024.{t}-Data.csv", _flat_csv(t, ["19001960100", "19001960200"], 3))
            z.writestr(f"ACSDT5Y2024.{t}-Column-Metadata.csv", "x\n")
        paths.append(p)
    D = derive(load_files(paths)).set_index("geoid")
    assert list(D.index) == ["19001960100", "19001960200"]
    assert D.loc["19001960100", "hh_inc_lt35k"] == 18 and D.loc["19001960100", "rent_total"] == 0
    with pytest.raises(ValueError, match="missing tables"):
        load_files(paths[:-1])
    nat = tmp_path / "us.csv"
    nat.write_text(_flat_csv("B19001", ["US"], 5, geo_prefix="0100000"))
    with pytest.raises(ValueError, match="All Census Tracts"):
        load_files([nat])


def test_load_files_split_by_state(tmp_path):
    """One file per table per state: rows must stack, not be dropped."""
    import zipfile
    from redistricting.acs import TABLES, load_files
    paths = []
    for st, geo, val in (("IA", "19001960100", 3), ("IL", "17001000100", 7)):
        for t in TABLES:
            p = tmp_path / f"{st}.{t}.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr(f"ACSDT5Y2024.{t}-Data.csv", _flat_csv(t, [geo], val))
            paths.append(p)
    D = derive(load_files(paths)).set_index("geoid")
    assert D.loc["19001960100", "hh_total"] == 3 and D.loc["17001000100", "hh_total"] == 7
    assert D.loc["17001000100", "hh_inc_lt35k"] == 42


def test_national_thresholds_with_acs(tmp_path):
    from redistricting.rl.groups import national_thresholds
    M = pd.DataFrame({"geoid": ["a", "b", "c"], "pop": [10.0, 10, 10], "pov_total": [10.0, 10, 10],
                      "pov_below": [1.0, 2, 3]})
    X = pd.DataFrame({"geoid": ["a", "b", "c"], "hh_total": [10.0, 10, 10], "snap_hh": [1.0, 2, 6]})
    M.to_csv(tmp_path / "m.csv", index=False); X.to_csv(tmp_path / "x.csv", index=False)
    T = national_thresholds(tmp_path / "m.csv", tmp_path / "x.csv").set_index("group")
    sh = np.array([0.1, 0.2, 0.6])
    assert T.loc["snap", "tau"] == pytest.approx(sh.mean() + sh.std(ddof=1))
    assert "poverty" in T.index


def test_remap_ct_keeps_tract_number():
    from redistricting.acs import remap_ct
    D = pd.DataFrame({"geoid": ["09001010100", "09003990000", "09009990000", "19001000100"], "x": [1, 2, 3, 4]})
    cur = ["09190010100", "09110990000", "09170990000"]          # 2022 planning-region codes
    R = remap_ct(D, cur)
    assert R.geoid.tolist() == ["09190010100", "09003990000", "09009990000", "19001000100"]  # dup water tracts kept
