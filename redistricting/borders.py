"""Exact adjacency and shared-border lengths from TIGER, by segment hashing.

TIGER polygons are edge-matched: a boundary shared by two units is stored as
byte-identical vertex pairs in both.  Adjacency and shared-border length are
therefore a hash join over segments, and no geometry library is needed.

    python -m redistricting.borders --shp TIGER/tl_2026_us_county/tl_2026_us_county \\
        --state 01 --out data/AL_county_borders.csv
"""
import argparse
from collections import defaultdict
import pandas as pd
from .shapefile import read_dbf, read_shp, seg_len


def build_borders(shp_base, state=None):
    """Hash every segment; return (gids, names, perim_m, border{(i,j):m}, owners)."""
    recs = read_dbf(shp_base + ".dbf"); polys = read_shp(shp_base + ".shp")
    assert len(recs) == len(polys)
    sel = [i for i, r in enumerate(recs)
           if state is None or r["STATEFP"] == state]
    gid  = [recs[i]["GEOID"] for i in sel]
    name = [recs[i].get("NAME", recs[i].get("NAMELSAD", "")) for i in sel]
    N = len(sel)
    owners = defaultdict(set); perim = [0.0] * N; dup_in_unit = 0
    for k, si in enumerate(sel):
        seen_here = set()
        for rb in polys[si]:
            n = len(rb) // 16
            pts = [rb[j*16:(j+1)*16] for j in range(n)]
            for j in range(n - 1):
                a, b = pts[j], pts[j+1]
                key = (a, b) if a < b else (b, a)
                if key in seen_here: dup_in_unit += 1
                seen_here.add(key)
                owners[key].add(k)
                perim[k] += seg_len(a, b)
    border = defaultdict(float)
    for key, own in owners.items():
        if len(own) < 2: continue
        o = sorted(own); L = seg_len(*key)
        for x in range(len(o)):
            for y in range(x + 1, len(o)):
                border[(o[x], o[y])] += L
    return gid, name, perim, dict(border), owners, dup_in_unit


def write_borders(shp_base, out_csv, state="01"):
    gid, name, perim, border, owners, dup = build_borders(shp_base, state)
    multi = sum(1 for v in owners.values() if len(v) > 2)
    print(f"  units {len(gid)}  segments {len(owners)}  pairs {len(border)}  "
          f"dup-within-unit {dup}  >2 owners {multi}")
    pd.DataFrame([{"a": gid[i], "b": gid[j], "shared_km": L/1000}
                  for (i, j), L in sorted(border.items())]).to_csv(out_csv, index=False)
    tot = sum(perim)/1000; internal = sum(border.values())/1000
    print(f"  sum unit perimeter {tot:,.1f} km   internal {internal:,.1f} km"
          f"   => outer boundary {tot - 2*internal:,.1f} km")
    return gid, perim, border


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shp", required=True, help="shapefile path WITHOUT extension")
    ap.add_argument("--state", default=None, help="2-digit STATEFP to keep (default: all)")
    ap.add_argument("--out", required=True, help="output borders CSV (a, b, shared_km)")
    A = ap.parse_args()
    write_borders(A.shp, A.out, state=A.state)


if __name__ == "__main__":
    main()
