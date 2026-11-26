"""Where each ACS industry sector concentrates, at tract level.

For each of the 13 ACS industry sectors (share of employed residents 16+, by where
workers LIVE), a tract is "high" when its share is at least the US tract mean + 1 SD --
the same rule USDA ERS uses for its county flags, applied to tracts.

Outputs (outputs/industry/):
  * industry_state_annotations.csv -- one row per (sector, state): state share, US share,
    ratio, high tracts, share of the state's workers living in them, top county, and a
    one-line note.  Long format, so it can later be regrouped by state.
  * sector_<nn>_<slug>.md          -- the same annotations, one file per sector, all states.
  * tracts_<ST>.png                -- one state's tracts, one small panel per sector.
  * <ST>/<nn>_<slug>.png            -- one figure per sector: the state's tracts with high
    tracts highlighted and counties labelled, a zoom on the county with the most high
    tracts, and the state's numbers.

    python -m experiments.industry_tracts --state IA
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting.shapefile import read_dbf, read_shp

PROJ = ROOT.parent
MASTER = PROJ / "extracted" / "tract_master.csv"
KEY = PROJ / "extracted" / "industry_sector_key.csv"
TIGER = PROJ / "TIGER"

SHORT = {1: "Agriculture & mining", 2: "Arts, lodging & food", 3: "Construction",
         4: "Education & health care", 5: "Finance & real estate", 6: "Information",
         7: "Manufacturing", 8: "Other services", 9: "Professional & admin services",
         10: "Public administration", 11: "Retail trade", 12: "Transport & utilities",
         13: "Wholesale trade"}
ABBR = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT',
        '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL',
        '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD',
        '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE',
        '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
        '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
        '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV',
        '55': 'WI', '56': 'WY', '72': 'PR'}
FIPS = {v: k for k, v in ABBR.items()}
MIN_COUNTY_WORKERS = 1000           # top-county pick ignores tiny counties (noisy shares)

SURF, INK, INK2, MUTED, BASE, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e6e1", "#fcfcfb"
HIGH = "#2a78d6"                    # validated categorical slot 1 / sequential blue 450


def sectors():
    return list(range(1, 14))


def col(k):
    return f"ind_{k:02d}_share"


def load():
    t = pd.read_csv(MASTER, dtype={"geoid": str, "state_fips": str, "county_fips": str})
    t["cf"] = t.geoid.str[:5]
    t["st"] = t.state_fips.map(ABBR)
    cn = pd.DataFrame(read_dbf(str(TIGER / "tl_2026_us_county" / "tl_2026_us_county.dbf")))
    names = cn.set_index("GEOID")["NAME"]
    t["county"] = t.cf.map(names)
    return t, names


def thresholds(t):
    """US tract mean + 1 SD per sector (tracts with a valid share, unweighted)."""
    return {k: float(t[col(k)].mean() + t[col(k)].std()) for k in sectors()}


def annotate(t, thr):
    key = pd.read_csv(KEY).set_index("column")["sector_label"]
    rows = []
    v = t[t.emp_total > 0]
    for k in sectors():
        c = col(k)
        w = v.emp_total * v[c].fillna(0) / 100           # workers in the sector
        us = w.sum() / v.emp_total.sum() * 100
        hi = v[c] >= thr[k]
        for st, g in v.groupby("st"):
            wg = w[g.index]
            share = wg.sum() / g.emp_total.sum() * 100
            h = hi[g.index]
            cty = (pd.DataFrame({"w": wg, "e": g.emp_total, "county": g.county})
                   .groupby("county").sum())
            cty = cty[cty.e >= MIN_COUNTY_WORKERS]
            cty["s"] = cty.w / cty.e * 100
            top = cty.s.idxmax() if len(cty) else ""
            topv = float(cty.s.max()) if len(cty) else np.nan
            hw = g.emp_total[h].sum() / g.emp_total.sum() * 100
            hc = (pd.DataFrame({"e": g.emp_total[h], "county": g.county[h]})
                  .groupby("county").e.sum().sort_values(ascending=False))
            where = ", ".join(hc.index[:3]) if len(hc) else "none"
            rows.append(dict(sector=k, sector_label=key[c], sector_short=SHORT[k], state=st,
                             state_share=round(share, 2), us_share=round(us, 2),
                             ratio=round(share / us, 2), threshold=round(thr[k], 2),
                             tracts=len(g), high_tracts=int(h.sum()),
                             high_tract_pct=round(h.mean() * 100, 1),
                             workers_in_high_pct=round(hw, 1),
                             top_county=top, top_county_share=round(topv, 1),
                             high_tract_counties=where))
    A = pd.DataFrame(rows)
    A["rank"] = A.groupby("sector").state_share.rank(ascending=False, method="min").astype(int)
    n = A.groupby("sector").state.transform("size")

    def note(r):
        rel = ("well above" if r.ratio >= 1.5 else "above" if r.ratio >= 1.1 else
               "near" if r.ratio > 0.9 else "below" if r.ratio > 0.67 else "well below")
        s = (f"{r.state}: {r.state_share:.1f}% of employed residents work in {r.sector_short.lower()} "
             f"({rel} the US {r.us_share:.1f}%; rank {r['rank']} of {n[r.name]}). "
             f"{r.high_tracts} of {r.tracts} tracts ({r.high_tract_pct:.0f}%) clear the "
             f"{r.threshold:.1f}% high mark, home to {r.workers_in_high_pct:.0f}% of the state's workers")
        s += f", mostly in {r.high_tract_counties}." if r.high_tracts else "."
        if r.top_county:
            s += f" Highest county: {r.top_county} ({r.top_county_share:.1f}%)."
        return s
    A["note"] = A.apply(note, axis=1)
    return A


def write_annotations(A, out):
    out.mkdir(parents=True, exist_ok=True)
    A.to_csv(out / "industry_state_annotations.csv", index=False)
    for k, g in A.groupby("sector"):
        g = g.sort_values("rank")
        r0 = g.iloc[0]
        slug = re.sub(r"[^a-z]+", "_", r0.sector_short.lower()).strip("_")
        L = [f"# {r0.sector_short} (ACS `{col(k)}`)", "",
             f"*{r0.sector_label}.* Share of employed residents 16+, by where workers live. "
             f"US share {r0.us_share:.1f}%. A tract is **high** when its share is at least "
             f"{r0.threshold:.1f}% (US tract mean + 1 SD).", "",
             "| Rank | State | Share | × US | High tracts | Workers in high tracts | Highest county (≥1,000 workers) | High tracts mostly in |",
             "|---:|---|---:|---:|---:|---:|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r['rank']} | {r.state} | {r.state_share:.1f}% | {r.ratio:.2f} | "
                     f"{r.high_tracts} / {r.tracts} | {r.workers_in_high_pct:.0f}% | "
                     f"{r.top_county} {r.top_county_share:.1f}% | {r.high_tract_counties} |")
        L += ["", "## Notes by state", ""] + [f"- {r.note}" for _, r in g.sort_values("state").iterrows()]
        (out / f"sector_{k:02d}_{slug}.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------- state maps
def rings_xy(rings, lat0):
    out = []
    for b in rings:
        p = np.frombuffer(b, dtype="<f8").reshape(-1, 2).copy()
        p[:, 0] *= np.cos(np.radians(lat0))
        out.append(p)
    return out


class StateGeo:
    """Tract and county polygons of one state, in a local equirectangular frame."""

    def __init__(self, t, st):
        fp = FIPS[st]
        d = sorted(TIGER.glob(f"tl_*_{fp}_tract"))[-1]
        base = str(d / d.name)
        geo = [r["GEOID"] for r in read_dbf(base + ".dbf")]
        self.s = t[t.st == st].set_index("geoid")
        self.lat0 = float(np.nanmean(self.s.lat)) if "lat" in self.s else 40.0
        self.polys, owner = [], []
        for gid, rings in zip(geo, read_shp(base + ".shp")):
            for p in rings_xy(rings, self.lat0):
                if len(p) > 3:
                    self.polys.append(p); owner.append(gid)
        self.owner = np.array(owner)
        self.county_of = np.array([g[:5] for g in owner])
        self.cpolys, self.cpoly_cf, self.cent, self.cname = [], [], {}, {}
        for r, rings in zip(read_dbf(str(TIGER / "tl_2026_us_county" / "tl_2026_us_county.dbf")),
                            read_shp(str(TIGER / "tl_2026_us_county" / "tl_2026_us_county.shp"))):
            if r["STATEFP"] != fp:
                continue
            for p in rings_xy(rings, self.lat0):
                self.cpolys.append(p); self.cpoly_cf.append(r["GEOID"])
            self.cent[r["GEOID"]] = (float(r["INTPTLON"]) * np.cos(np.radians(self.lat0)),
                                     float(r["INTPTLAT"]))
            self.cname[r["GEOID"]] = r["NAME"]
        allp = np.vstack(self.polys)
        self.width = float(allp[:, 0].max() - allp[:, 0].min())

    def county_table(self, k, thr):
        """Per county: tracts, high tracts, workers, share of workers living in high tracts."""
        s = self.s[self.s.emp_total > 0]
        hi = s[col(k)] >= thr[k]
        g = pd.DataFrame({"cf": s.cf, "hi": hi, "e": s.emp_total, "eh": s.emp_total * hi})
        c = g.groupby("cf").agg(tracts=("hi", "size"), high=("hi", "sum"), e=("e", "sum"), eh=("eh", "sum"))
        c["cover"] = c.eh / c.e
        c["name"] = c.index.map(self.cname)
        return c

    def labels(self, k, thr, n_cover=4, n_count=2, min_sep=0.11):
        """Counties to name: where high tracts cover most of the county, plus where the
        most high-tract workers live; skip a label that would sit on top of another."""
        c = self.county_table(k, thr)
        c = c[c.high > 0]
        pick = list(c[c.e >= MIN_COUNTY_WORKERS].sort_values("cover", ascending=False).index[:n_cover])
        pick += [i for i in c.sort_values("eh", ascending=False).index[:n_count] if i not in pick]
        placed = []
        for cf in pick:
            x, y = self.cent[cf]
            if all(np.hypot(x - a, y - b) > min_sep * self.width for a, b, _ in placed):
                placed.append((x, y, cf))
        return placed, c

    def draw(self, ax, k, thr, lw=0.15, clip=None):
        x = self.s[col(k)].reindex(self.owner).values
        fc = np.where(x >= thr[k], HIGH, BASE)
        polys, cp = self.polys, self.cpolys
        if clip is not None:
            m = self.county_of == clip
            polys = [p for p, keep in zip(self.polys, m) if keep]; fc = fc[m]
            cp = [p for p, cf in zip(self.cpolys, self.cpoly_cf) if cf == clip]
        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors=EDGE, linewidths=lw))
        ax.add_collection(PolyCollection(cp, facecolors="none", edgecolors=MUTED, linewidths=0.4))
        ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")


HALO = [pe.withStroke(linewidth=2.8, foreground=SURF)]


def overview(G, thr, A, st, out):
    a = A[A.state == st].set_index("sector")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(5, 3, figsize=(17, 19), facecolor=SURF)
    for ax, k in zip(axes.flat, sectors()):
        G.draw(ax, k, thr)
        r = a.loc[k]
        ax.set_title(f"{k:02d}  {SHORT[k]}", loc="left", fontsize=12, color=INK, pad=17)
        ax.text(0, 1.0, f"{st} {r.state_share:.1f}% vs US {r.us_share:.1f}%  ·  high ≥ {r.threshold:.1f}%  ·  "
                f"{r.high_tracts}/{r.tracts} tracts, {r.workers_in_high_pct:.0f}% of workers",
                transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        placed, _ = G.labels(k, thr, n_cover=3, n_count=1, min_sep=0.14)
        for x, y, cf in placed:
            ax.text(x, y, G.cname[cf], fontsize=8.5, color=INK, ha="center", va="center",
                    path_effects=HALO, zorder=5)
    for ax in list(axes.flat)[13:]:
        ax.axis("off")
    axes.flat[13].text(0, 0.9, READ.format(n=G.n_us), va="top", fontsize=10, color=INK2,
                       transform=axes.flat[13].transAxes)
    fig.suptitle(f"{st}: tracts with a high share of residents in each industry (ACS)",
                 x=0.01, ha="left", fontsize=15, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    f = out / f"tracts_{st}.png"; fig.savefig(f, dpi=130, facecolor=SURF); plt.close(fig)
    return f


READ = ("How to read this\n\n"
        "Each panel is one ACS industry sector: the share of\n"
        "a tract's employed residents (16+) working in it,\n"
        "by where they live, not where the jobs are.\n\n"
        "Blue tracts are high: share at least the US tract\n"
        "mean + 1 SD (the ERS county rule, applied to all\n"
        "{n:,} US tracts with data). Grey lines are counties.\n\n"
        "Labelled: counties where high tracts hold the\n"
        "largest share of the county's workers, plus the\n"
        "county with the most workers in high tracts.\n"
        "Shares in small tracts carry wide ACS margins of error.")


def sector_figure(G, thr, A, st, k, out):
    r = A[(A.state == st) & (A.sector == k)].iloc[0]
    placed, c = G.labels(k, thr)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(15, 7.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.15], height_ratios=[1, 1.05],
                          left=0.01, right=0.99, top=0.80, bottom=0.03, wspace=0.04, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    G.draw(ax, k, thr, lw=0.2)
    for x, y, cf in placed:
        row = c.loc[cf]
        ax.text(x, y, f"{G.cname[cf]}\n{int(row.high)}/{int(row.tracts)} high", fontsize=8.5,
                color=INK, ha="center", va="center", path_effects=HALO, zorder=5, linespacing=1.1)
    # zoom: the county with the most high tracts, if it has small (urban) tracts worth seeing
    z = c.sort_values(["high", "eh"], ascending=False)
    axz = fig.add_subplot(gs[0, 1])
    if len(z):
        cf = z.index[0]
        G.draw(axz, k, thr, lw=0.3, clip=cf)
        zr = z.iloc[0]
        axz.set_title(f"Zoom: {G.cname[cf]} County (most high tracts)\n{int(zr.high)} of {int(zr.tracts)} tracts high, "
                      f"home to {zr.cover:.0%} of its workers", loc="left", fontsize=9.5, color=INK2)
    else:
        axz.axis("off")
    axt = fig.add_subplot(gs[1, 1]); axt.axis("off")
    top = (c[c.e >= MIN_COUNTY_WORKERS].sort_values("cover", ascending=False).head(5))
    lines = ["Counties where high tracts hold the most", "workers (counties with 1,000+ workers)"]
    lines += [f"  {n:<16s} {v:5.0%}   ({int(h)}/{int(t_)} tracts)"
              for n, v, h, t_ in zip(top.name, top.cover, top.high, top.tracts)] or ["  none"]
    us_rank = f"rank {r['rank']} of {A[A.sector == k].state.nunique()}, incl. DC and PR"
    body = (f"{st} share {r.state_share:.1f}% vs US {r.us_share:.1f}%\n"
            f"  {r.ratio:.2f}× US, {us_rank}\n"
            f"High mark {r.threshold:.1f}% (US tract mean + 1 SD)\n"
            f"{r.high_tracts} of {r.tracts} tracts high, home to {r.workers_in_high_pct:.0f}% of workers\n"
            f"Highest county overall: {r.top_county} {r.top_county_share:.1f}%\n\n" + "\n".join(lines))
    axt.text(0, 1, body, va="top", fontsize=9.5, color=INK2, family="DejaVu Sans Mono",
             transform=axt.transAxes)
    fig.text(0.01, 0.955, f"{st} · {k:02d} {SHORT[k]}", fontsize=15, color=INK, va="top")
    fig.text(0.01, 0.905, f"{r.sector_label}.\nBlue = tract share of employed residents ≥ {r.threshold:.1f}% "
             f"(ACS, by residence). Labels: county, high tracts / all tracts.", fontsize=9.5, color=INK2, va="top",
             linespacing=1.5)
    slug = re.sub(r"[^a-z]+", "_", SHORT[k].lower()).strip("_")
    d = out / st; d.mkdir(parents=True, exist_ok=True)
    f = d / f"{k:02d}_{slug}.png"; fig.savefig(f, dpi=130, facecolor=SURF); plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", nargs="+", default=["IA"])
    ap.add_argument("--out-dir", default="outputs/industry")
    A_ = ap.parse_args()
    out = ROOT / A_.out_dir
    t, _ = load()
    thr = thresholds(t)
    A = annotate(t, thr)
    write_annotations(A, out)
    print("thresholds:", {k: round(v, 1) for k, v in thr.items()})
    for st in A_.state:
        G = StateGeo(t, st)
        G.n_us = int(t[[col(k) for k in sectors()]].notna().all(axis=1).sum())
        print(overview(G, thr, A, st, out))
        for k in sectors():
            print(sector_figure(G, thr, A, st, k, out))
        print(A[A.state == st][["sector_short", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
