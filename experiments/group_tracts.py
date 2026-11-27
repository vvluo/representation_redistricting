"""Where each protected group concentrates, at tract level, for every state.

The Iowa industry maps (experiments/industry_tracts.py) generalised to every group with
national tract data in extracted/tract_master.csv:

  industry    13 ACS sectors, share of employed residents 16+ (by residence)
  ethnicity   Hispanic, Black, Asian, American Indian/Alaska Native, Native Hawaiian/
              Pacific Islander -- share of citizen voting-age population (CVAP)
  poverty     share of people below the poverty line
  rurality    RUCA tier of the tract: micropolitan (4-6), small town (7-9), rural (10)
and, from the ACS 2024 tables in extracted/acs_api/US_2024.csv (redistricting.acs):
  education        bachelor's or more; no high-school diploma (B15003)
  income           households under $35k; $150k or more (B19001)
  class_of_worker  government, self-employed, nonprofit (B24080)
  housing          renters (B25003), rent >= 30% of income (B25070), vacant units (B25002)
  health           uninsured (B27010)
  food             households receiving SNAP (B22003)
  language         Spanish, other Indo-European, Asian/Pacific, other languages at home (B16007)
and, from the 2020 Census redistricting file (extracted/acs_api/US_2020_P5.csv, a full count):
  group_quarters   adult correctional facilities, nursing facilities, college housing (P5),
                   as a share of the tract's 2020 population (ACS B26103 stops at states)

A tract is "high" when its share is at least the US tract mean + 1 SD -- the threshold
tau the RL objective uses (data/group_thresholds.csv); groups not in that file get the
same rule computed here.  For a RUCA tier the share is 0 or 1, so "high" = the tract is
in the tier.

Outputs (outputs/groups/):
  annotations.csv            one row per (group, state), long format
  by_group/<key>.md          one group, all states ranked
  <ST>/README.md             one state, all groups: notes and links to the figures
  <ST>/overview.png          one state, a small panel per group
  <ST>/<family>__<key>.png   one state and group: high tracts, labelled counties, a zoom
                             on the county with the most high tracts, the numbers
PNGs are palette-quantised (most are 40-120 KB).

    python -m experiments.group_tracts --state IA AL       # or --state all --jobs 10
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, io, sys, time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False      # "$78,008" is money, not math
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting.shapefile import read_dbf, read_shp
from experiments.industry_tracts import (MASTER, TIGER, ABBR, FIPS, SHORT, SURF, INK, INK2, MUTED, BASE,
                                         EDGE, HIGH, HALO, rings_xy)

THRESH = ROOT / "data" / "group_thresholds.csv"
ACS = ROOT.parent / "extracted" / "acs_api" / "US_2024.csv"
P5 = ROOT.parent / "extracted" / "acs_api" / "US_2020_P5.csv"
MIN_COUNTY_W = 1000        # top-county picks ignore counties with fewer units of weight


@dataclass
class Var:
    key: str               # matches data/group_thresholds.csv where it exists
    family: str
    label: str             # short name
    desc: str              # what the share is
    unit: str              # who the weight counts ("workers", "residents", ...)
    weight: str            # master column used as the denominator / weight
    low: bool = False      # high = share <= tau
    money: bool = False    # values are dollars, not shares
    num: tuple = ()        # ACS count columns: share = sum(num) / weight


RACE = {"hispanic": "Hispanic", "black": "Black", "asian": "Asian", "aian": "American Indian & Alaska Native",
        "nhpi": "Native Hawaiian & Pacific Islander"}
VARS = (
    [Var(f"ind_{k:02d}", "industry", SHORT[k], f"share of employed residents 16+ working in {SHORT[k].lower()}",
         "workers", "emp_total") for k in range(1, 14)]
    + [Var(k, "ethnicity", v, f"{v} share of citizen voting-age population", "eligible voters", "cvap_total")
       for k, v in RACE.items()]
    + [Var("poverty", "poverty", "Poverty", "share of people below the poverty line", "residents", "pov_total"),
       Var("ba_plus", "education", "Bachelor's or more", "share of adults 25+ with a bachelor's degree or more",
           "adults 25+", "edu_total")]
    + [Var(k, "rurality", lab, f"RUCA {lo}-{hi} ({lab.lower()})", "residents", "pop")
       for k, lab, lo, hi in (("micropolitan", "Micropolitan", 4, 6), ("small_town", "Small town", 7, 9),
                              ("rural", "Rural", 10, 10))]
)
# ACS groups: (key = RL group name, family, label, description, unit, numerators, denominator)
ACS_VARS = [
    Var("less_than_hs", "education", "No high-school diploma", "share of adults 25+ without a high-school diploma",
        "adults 25+", "edu_total", num=("edu_lt_hs",)),
    Var("low_income", "income", "Household income < $35k", "share of households with income under $35,000",
        "households", "hh_inc_total", num=("hh_inc_lt35k",)),
    Var("high_income", "income", "Household income $150k+", "share of households with income $150,000 or more",
        "households", "hh_inc_total", num=("hh_inc_ge150k",)),
    Var("government_workers", "class_of_worker", "Government workers",
        "share of civilian employed 16+ working for local, state or federal government", "workers", "cow_total",
        num=("cow_government",)),
    Var("self_employed", "class_of_worker", "Self-employed", "share of civilian employed 16+ who are self-employed",
        "workers", "cow_total", num=("cow_self_employed",)),
    Var("nonprofit_workers", "class_of_worker", "Nonprofit workers",
        "share of civilian employed 16+ working for private nonprofits", "workers", "cow_total",
        num=("cow_nonprofit",)),
    Var("renters", "housing", "Renters", "share of occupied housing units that are renter-occupied",
        "households", "hh_occupied", num=("hh_renter",)),
    Var("rent_burdened", "housing", "Rent-burdened", "share of renter households paying 30%+ of income in rent",
        "renter households", "rent_total", num=("rent_burden30",)),
    Var("vacant", "housing", "Vacant housing", "share of housing units that are vacant", "housing units",
        "hu_total", num=("hu_vacant",)),
    Var("uninsured", "health", "Uninsured", "share of civilian non-institutionalised people with no health insurance",
        "residents", "hi_total", num=("hi_uninsured",)),
    Var("snap", "food", "SNAP households", "share of households receiving SNAP (food stamps)", "households",
        "hh_total", num=("snap_hh",)),
    Var("spanish_speakers", "language", "Spanish at home", "share of people 5+ speaking Spanish at home",
        "residents 5+", "lang_total", num=("lang_spanish",)),
    Var("indo_european_speakers", "language", "Other Indo-European at home",
        "share of people 5+ speaking another Indo-European language at home", "residents 5+", "lang_total",
        num=("lang_indo_european",)),
    Var("asian_pacific_speakers", "language", "Asian/Pacific language at home",
        "share of people 5+ speaking an Asian or Pacific Island language at home", "residents 5+", "lang_total",
        num=("lang_asian_pacific",)),
    Var("other_language_speakers", "language", "Other language at home",
        "share of people 5+ speaking another language at home", "residents 5+", "lang_total",
        num=("lang_other",)),
]
GQ_VARS = [
    Var("correctional", "group_quarters", "Adult correctional", "share of 2020 population in adult correctional "
        "facilities", "residents (2020)", "gq_pop_total", num=("gq_correctional",)),
    Var("nursing_facilities", "group_quarters", "Nursing facilities", "share of 2020 population in nursing / "
        "skilled-nursing facilities", "residents (2020)", "gq_pop_total", num=("gq_nursing",)),
    Var("college_housing", "group_quarters", "College housing", "share of 2020 population in college / university "
        "student housing", "residents (2020)", "gq_pop_total", num=("gq_college",)),
]
if P5.exists():
    VARS = VARS + GQ_VARS
if ACS.exists():
    _poverty = [i for i, v in enumerate(VARS) if v.key == "ba_plus"][0] + 1
    VARS = VARS[:_poverty] + ACS_VARS + VARS[_poverty:]          # rurality stays last
    _fam = list(dict.fromkeys(v.family for v in VARS))
    VARS = sorted(VARS, key=lambda v: _fam.index(v.family))     # keep each family together
RUCA = {"micropolitan": (4, 6), "small_town": (7, 9), "rural": (10, 10)}
TH_ALIAS = {}    # aian and nhpi are separate here; RL's "native" pools them, so tau is computed here


def load():
    t = pd.read_csv(MASTER, dtype={"geoid": str, "state_fips": str, "county_fips": str})
    t["cf"] = t.geoid.str[:5]
    t["st"] = t.state_fips.map(ABBR)
    t["county"] = t.cf.map(pd.DataFrame(read_dbf(str(TIGER / "tl_2026_us_county" / "tl_2026_us_county.dbf")))
                           .set_index("GEOID")["NAME"])
    for f in (ACS, P5):
        if f.exists():
            X = pd.read_csv(f, dtype={"geoid": str})
            t = t.merge(X.drop(columns=[c for c in X.columns if c in t.columns and c != "geoid"]), on="geoid",
                        how="left")
    for v in VARS:
        if v.num:
            s = t[list(v.num)].sum(axis=1, min_count=1) / t[v.weight]
        elif v.family == "industry":
            s = t[f"{v.key}_share"] / 100
        elif v.family == "ethnicity":
            s = t[f"pct_cvap_{v.key}"] / 100
        elif v.key == "poverty":
            s = t.poverty_rate / 100
        elif v.key == "ba_plus":
            s = t.pct_ba_plus / 100
        else:
            lo, hi = RUCA[v.key]
            s = t.ruca_primary.between(lo, hi).astype(float).where(t.ruca_primary.notna())
        t[f"s_{v.key}"] = s.where(t[v.weight] > 0)
    return t


def thresholds(t):
    """tau per group: data/group_thresholds.csv (the RL objective's), else US tract mean
    +/- 1 SD of the share computed here."""
    th = pd.read_csv(THRESH).set_index("group").tau.to_dict() if THRESH.exists() else {}
    out = {}
    for v in VARS:
        k = TH_ALIAS.get(v.key, v.key)
        if k in th and not v.low:
            out[v.key] = float(th[k])
        else:
            s = t[f"s_{v.key}"]
            out[v.key] = float(s.mean() - s.std()) if v.low else float(s.mean() + s.std())
    return out


def is_high(v, s, tau):
    return (s <= tau) if v.low else (s >= tau)


def fmt(v, x):
    return f"${x:,.0f}" if v.money else f"{x:.1%}"


def annotate(t, tau):
    rows = []
    for v in VARS:
        s, w = t[f"s_{v.key}"], t[v.weight]
        ok = s.notna()
        us = float((s[ok] * w[ok]).sum() / w[ok].sum())
        hi = is_high(v, s, tau[v.key]) & ok
        for st, idx in t[ok].groupby("st").groups.items():
            g = t.loc[idx]; sg, wg, hg = s[idx], w[idx], hi[idx]
            share = float((sg * wg).sum() / wg.sum())
            cty = pd.DataFrame({"x": sg * wg, "w": wg, "county": g.county}).groupby("county").sum()
            cty = cty[cty.w >= MIN_COUNTY_W]; cty["s"] = cty.x / cty.w
            top = (cty.s.idxmin() if v.low else cty.s.idxmax()) if len(cty) else ""
            topv = float(cty.loc[top, "s"]) if len(cty) else np.nan
            hw = float(wg[hg].sum() / wg.sum())
            hc = pd.DataFrame({"w": wg[hg], "county": g.county[hg]}).groupby("county").w.sum().sort_values(ascending=False)
            rows.append(dict(family=v.family, group=v.key, label=v.label, state=st, state_value=share, us_value=us,
                             ratio=share / us if us else np.nan, tau=tau[v.key], tracts=len(g),
                             high_tracts=int(hg.sum()), high_tract_pct=float(hg.mean()),
                             weight_in_high_pct=hw, top_county=top, top_county_value=topv,
                             high_tract_counties=", ".join(hc.index[:3]) if len(hc) else ""))
    A = pd.DataFrame(rows)
    A["rank"] = A.groupby("group").weight_in_high_pct.rank(ascending=False, method="min").astype(int)
    n = A.groupby("group").state.transform("size")
    V = {v.key: v for v in VARS}

    def note(r):
        v = V[r.group]
        if v.money:
            lead = f"{r.state}: population-weighted mean tract median household income {fmt(v, r.state_value)} (US {fmt(v, r.us_value)})."
        elif v.family == "rurality":
            lead = f"{r.state}: {r.state_value:.1%} of residents live in {v.label.lower()} tracts (US {r.us_value:.1%})."
        else:
            rel = ("well above" if r.ratio >= 1.5 else "above" if r.ratio >= 1.1 else "near" if r.ratio > 0.9
                   else "below" if r.ratio > 0.67 else "well below")
            lead = f"{r.state}: {v.label} {r.state_value:.1%} ({rel} the US {r.us_value:.1%})."
        mark = f"{'≤' if v.low else '≥'} {fmt(v, r.tau)}" if v.family != "rurality" else "in the tier"
        s = (f"{lead} {r.high_tracts} of {r.tracts} tracts ({r.high_tract_pct:.0%}) are high ({mark}), "
             f"home to {r.weight_in_high_pct:.0%} of the state's {v.unit} (rank {r['rank']} of {n[r.name]})")
        s += f", mostly in {r.high_tract_counties}." if r.high_tracts else "."
        if r.top_county and v.family != "rurality":
            s += f" {'Lowest' if v.low else 'Highest'} county: {r.top_county} ({fmt(v, r.top_county_value)})."
        return s
    A["note"] = A.apply(note, axis=1)
    return A


# ----------------------------------------------------------------- state maps
class StateGeo:
    """Tract and county polygons of one state, in a local equirectangular frame."""

    def __init__(self, t, st):
        fp = FIPS[st]
        d = sorted(TIGER.glob(f"tl_*_{fp}_tract"))[-1]
        base = str(d / d.name)
        geo = [r["GEOID"] for r in read_dbf(base + ".dbf")]
        self.s = t[t.st == st].set_index("geoid")
        # uninhabited Northwestern Hawaiian Islands (part of Honolulu County) would shrink the map
        self.west = -160.6 if st == "HI" else None
        rings_all = read_shp(base + ".shp")
        lat = [np.frombuffer(r[0], dtype="<f8").reshape(-1, 2)[:, 1].mean() for r in rings_all if r]
        self.lat0 = float(np.median(lat))
        self.polys, owner = [], []
        for gid, rings in zip(geo, rings_all):
            for p in rings_xy(rings, self.lat0):
                if len(p) > 3 and not self._outlier(p):
                    self.polys.append(p); owner.append(gid)
        self.owner = np.array(owner)
        self.county_of = np.array([g[:5] for g in owner])
        self.cpolys, self.cpoly_cf, self.cent, self.cname = [], [], {}, {}
        for r, rings in zip(read_dbf(str(TIGER / "tl_2026_us_county" / "tl_2026_us_county.dbf")),
                            read_shp(str(TIGER / "tl_2026_us_county" / "tl_2026_us_county.shp"))):
            if r["STATEFP"] != fp:
                continue
            for p in rings_xy(rings, self.lat0):
                if not self._outlier(p):
                    self.cpolys.append(p); self.cpoly_cf.append(r["GEOID"])
            self.cent[r["GEOID"]] = (float(r["INTPTLON"]) * np.cos(np.radians(self.lat0)), float(r["INTPTLAT"]))
            self.cname[r["GEOID"]] = r["NAME"]
        allp = np.vstack(self.polys)
        self.width = float(allp[:, 0].max() - allp[:, 0].min())
        if st == "AK":   # Aleutians cross 180 deg; keep the mainland frame
            self.width = min(self.width, 25.0)

    def _outlier(self, p):
        return self.west is not None and p[:, 0].mean() < self.west * np.cos(np.radians(self.lat0))

    def high(self, v, tau):
        s = self.s[f"s_{v.key}"]
        return is_high(v, s, tau[v.key]) & s.notna()

    def county_table(self, v, tau):
        s = self.s[self.s[f"s_{v.key}"].notna()]
        hi = self.high(v, tau)[s.index]
        w = s[v.weight]
        c = pd.DataFrame({"cf": s.cf, "hi": hi, "w": w, "wh": w * hi}).groupby("cf").agg(
            tracts=("hi", "size"), high=("hi", "sum"), w=("w", "sum"), wh=("wh", "sum"))
        c["cover"] = c.wh / c.w
        c["name"] = c.index.map(self.cname)
        return c

    def labels(self, v, tau, n_cover=4, n_count=2, min_sep=0.11):
        c = self.county_table(v, tau)
        c = c[c.high > 0]
        pick = list(c[c.w >= MIN_COUNTY_W].sort_values("cover", ascending=False).index[:n_cover])
        pick += [i for i in c.sort_values("wh", ascending=False).index[:n_count] if i not in pick]
        placed = []
        for cf in pick:
            x, y = self.cent[cf]
            if all(np.hypot(x - a, y - b) > min_sep * self.width for a, b, _ in placed):
                placed.append((x, y, cf))
        return placed, c

    def draw(self, ax, v, tau, lw=0.15, clip=None):
        h = self.high(v, tau).reindex(self.owner).fillna(False).values
        fc = np.where(h, HIGH, BASE)
        polys, cp = self.polys, self.cpolys
        if clip is not None:
            m = self.county_of == clip
            polys = [p for p, keep in zip(self.polys, m) if keep]; fc = fc[m]
            cp = [p for p, cf in zip(self.cpolys, self.cpoly_cf) if cf == clip]
        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors=EDGE, linewidths=lw))
        ax.add_collection(PolyCollection(cp, facecolors="none", edgecolors=MUTED, linewidths=0.35))
        ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
        if clip is not None and fc.size and (fc == HIGH).any():
            # frame the high tracts (not outlying islands), at least 40% of the county's extent
            P = np.vstack(polys); H = np.vstack([p for p, f in zip(polys, fc) if f == HIGH])
            lo, hi = H.min(0), H.max(0); span = P.max(0) - P.min(0)
            half = np.maximum((hi - lo) / 2 * 1.15, 0.2 * span); mid = (lo + hi) / 2
            ax.set_xlim(mid[0] - half[0], mid[0] + half[0]); ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
        elif self.s.st.iloc[0] == "AK":
            ax.set_xlim(-180 * np.cos(np.radians(self.lat0)), -129 * np.cos(np.radians(self.lat0)))


def save_png(fig, path, dpi, colors=48):
    """Render, then palette-quantise: the maps use a handful of flat colours, so 48
    colours keep text antialiasing while cutting size ~4x."""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, facecolor=SURF, format="png"); plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    im.save(path, format="PNG", optimize=True)
    return path


def mark(v, tau):
    if v.family == "rurality":
        return "tract in the tier"
    return f"{v.desc} {'≤' if v.low else '≥'} {fmt(v, tau)}"


def group_figure(G, tau, A, st, v, out, dpi):
    r = A[(A.state == st) & (A.group == v.key)].iloc[0]
    placed, c = G.labels(v, tau)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(14, 6.8), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.15], height_ratios=[1, 1.05],
                          left=0.01, right=0.99, top=0.80, bottom=0.03, wspace=0.04, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    G.draw(ax, v, tau, lw=0.2)
    for x, y, cf in placed:
        row = c.loc[cf]
        ax.text(x, y, f"{G.cname[cf]}\n{int(row.high)}/{int(row.tracts)} high", fontsize=8.5, color=INK,
                ha="center", va="center", path_effects=HALO, zorder=5, linespacing=1.1)
    z = c.sort_values(["high", "wh"], ascending=False)
    axz = fig.add_subplot(gs[0, 1])
    if len(z):
        cf = z.index[0]; zr = z.iloc[0]
        G.draw(axz, v, tau, lw=0.3, clip=cf)
        axz.set_title(f"Zoom: {G.cname[cf]} (most high tracts)\n{int(zr.high)} of {int(zr.tracts)} tracts high,\n"
                      f"home to {zr.cover:.0%} of its {v.unit}", loc="left", fontsize=9.5, color=INK2)
    else:
        axz.axis("off")
    axt = fig.add_subplot(gs[1, 1]); axt.axis("off")
    top = c[c.w >= MIN_COUNTY_W].sort_values("cover", ascending=False).head(5)
    lines = [f"Counties where high tracts hold the most", f"{v.unit} (counties with {MIN_COUNTY_W:,}+)"]
    lines += [f"  {n[:16]:<16s} {x:5.0%}   ({int(h)}/{int(t_)} tracts)"
              for n, x, h, t_ in zip(top.name, top.cover, top.high, top.tracts)] or ["  none"]
    n_st = A[A.group == v.key].state.nunique()
    if v.money:
        head = f"{st} mean tract median {fmt(v, r.state_value)} vs US {fmt(v, r.us_value)}\n"
    elif v.family == "rurality":
        head = f"{st}: {r.state_value:.1%} of residents in tier vs US {r.us_value:.1%}\n"
    else:
        head = f"{st} {r.state_value:.1%} vs US {r.us_value:.1%} ({r.ratio:.2f}× US)\n"
    body = (head + f"High mark: {'tract in tier' if v.family == 'rurality' else ('≤ ' if v.low else '≥ ') + fmt(v, r.tau)}\n"
            f"{r.high_tracts} of {r.tracts} tracts high, home to {r.weight_in_high_pct:.0%} of {v.unit}\n"
            f"  rank {r['rank']} of {n_st} states by that share\n\n" + "\n".join(lines))
    axt.text(0, 1, body, va="top", fontsize=9.5, color=INK2, family="DejaVu Sans Mono", transform=axt.transAxes)
    fig.text(0.01, 0.955, f"{st} · {v.family} · {v.label}", fontsize=15, color=INK, va="top")
    rule = ("US tract mean − 1 SD" if v.low else "US tract mean + 1 SD") if v.family != "rurality" else "USDA RUCA 2020"
    fig.text(0.01, 0.905, f"Blue = {mark(v, r.tau)} ({rule}).\nLabels: county, high tracts / all tracts. "
             f"Grey lines are counties.", fontsize=9.5, color=INK2, va="top", linespacing=1.5)
    return save_png(fig, out / f"{v.family}__{v.key}.png", dpi)


def overview(G, tau, A, st, out, dpi):
    a = A[A.state == st].set_index("group")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    rows = -(-(len(VARS) + 1) // 5)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 4 * rows), facecolor=SURF)
    for ax, v in zip(axes.flat, VARS):
        G.draw(ax, v, tau, lw=0.1)
        r = a.loc[v.key]
        ax.set_title(f"{v.family} · {v.label}", loc="left", fontsize=10.5, color=INK, pad=14)
        val = (f"{fmt(v, r.state_value)} vs US {fmt(v, r.us_value)}" if v.family != "rurality"
               else f"{r.state_value:.0%} of residents")
        ax.text(0, 1.0, f"{val} · {r.high_tracts}/{r.tracts} high · {r.weight_in_high_pct:.0%} of {v.unit}",
                transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    ax = axes.flat[len(VARS)]; ax.axis("off")
    ax.text(0, 0.95, "How to read this\n\nBlue tracts are high for the group:\n"
            "share ≥ US tract mean + 1 SD (the RL\nobjective's τ); rurality: tract in\n"
            "the RUCA tier. Grey lines are counties.\n\n"
            "Industry: by where workers live.\nEthnicity: share of citizen voting-\n"
            "age population. Small tracts carry\nwide ACS margins of error.",
            va="top", fontsize=9.5, color=INK2, transform=ax.transAxes)
    for ax in list(axes.flat)[len(VARS) + 1:]:
        ax.axis("off")
    fig.suptitle(f"{st}: tracts with a high share of each group (ACS 2024, CVAP, RUCA)", x=0.01, ha="left",
                 fontsize=15, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (4 * rows)), h_pad=2.2)
    return save_png(fig, out / "overview.png", dpi)


def state_readme(A, st, out):
    a = A[A.state == st]
    L = [f"# {st}: where protected groups concentrate", "",
         "Blue tracts in each figure are **high** for the group (share ≥ US tract mean + 1 SD, the RL objective's τ; "
         "rurality: tract in the RUCA tier). Rank = among states, by the share of "
         "the group's weight living in high tracts.", "", "![overview](overview.png)", ""]
    V = {v.key: v for v in VARS}
    for fam, g in a.groupby("family", sort=False):
        L += [f"## {fam}", "", "| Group | State | US | High tracts | Weight in high tracts | Rank | Figure |",
              "|---|---:|---:|---:|---:|---:|---|"]
        for _, r in g.iterrows():
            v = V[r.group]
            L.append(f"| {v.label} | {fmt(v, r.state_value)} | {fmt(v, r.us_value)} | {r.high_tracts}/{r.tracts} | "
                     f"{r.weight_in_high_pct:.0%} of {v.unit} | {r['rank']} | [png]({v.family}__{v.key}.png) |")
        L += [""] + [f"- {r.note}" for _, r in g.iterrows()] + [""]
    (out / "README.md").write_text("\n".join(L))


def by_group(A, out):
    d = out / "by_group"; d.mkdir(parents=True, exist_ok=True)
    V = {v.key: v for v in VARS}
    for k, g in A.groupby("group"):
        v = V[k]; g = g.sort_values("rank")
        L = [f"# {v.family} · {v.label}", "", f"{v.desc}. High = {mark(v, g.tau.iloc[0])}. US {fmt(v, g.us_value.iloc[0])}.", "",
             f"| Rank | State | Value | High tracts | {v.unit} in high tracts | High tracts mostly in |",
             "|---:|---|---:|---:|---:|---|"]
        L += [f"| {r['rank']} | [{r.state}](../{r.state}/README.md) | {fmt(v, r.state_value)} | {r.high_tracts}/{r.tracts} | "
              f"{r.weight_in_high_pct:.0%} | {r.high_tract_counties} |" for _, r in g.iterrows()]
        (d / f"{k}.md").write_text("\n".join(L) + "\n")


def index(A, out):
    V = {v.key: v for v in VARS}
    L = ["# Tract concentration maps, by state and group", "",
         "Blue tracts are high for a group: share >= US tract mean + 1 SD (the RL objective's tau); "
         "rurality: tract in the RUCA tier. Data: ACS 2024 5-year (industry, poverty, education, income, class "
         "of worker, housing, health insurance, SNAP, language), CVAP (ethnicity), USDA RUCA (rurality), "
         "2020 Census P5 (group quarters). `annotations.csv` has every number.", "",
         "## By state", "", " · ".join(f"[{s}]({s}/README.md)" for s in sorted(A.state.unique())), "",
         "## By group (states ranked by share of the group's weight in high tracts)", ""]
    for fam in dict.fromkeys(v.family for v in VARS):
        L.append(f"- **{fam}**: " + " · ".join(f"[{v.label}](by_group/{v.key}.md)" for v in VARS if v.family == fam))
    (out / "README.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------- driver
_W = {}


def _init(t, tau, A, out, dpi):
    _W.update(t=t, tau=tau, A=A, out=out, dpi=dpi)


def render_state(st):
    t, tau, A, out, dpi = _W["t"], _W["tau"], _W["A"], _W["out"], _W["dpi"]
    t0 = time.time()
    d = out / st; d.mkdir(parents=True, exist_ok=True)
    G = StateGeo(t, st)
    files = [overview(G, tau, A, st, d, dpi)]
    for v in VARS:
        files.append(group_figure(G, tau, A, st, v, d, dpi))
    state_readme(A, st, d)
    kb = sum(f.stat().st_size for f in files) / 1024
    return st, len(files), kb, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", nargs="+", default=["IA"], help="abbreviations, or 'all'")
    ap.add_argument("--out-dir", default="outputs/groups")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    A_ = ap.parse_args(argv)
    out = ROOT / A_.out_dir; out.mkdir(parents=True, exist_ok=True)
    t = load(); tau = thresholds(t); A = annotate(t, tau)
    A.to_csv(out / "annotations.csv", index=False); by_group(A, out); index(A, out)
    states = sorted(A.state.unique()) if A_.state == ["all"] else A_.state
    t0 = time.time(); tot_f = tot_kb = 0
    if A_.jobs > 1:
        with get_context("fork").Pool(A_.jobs, initializer=_init, initargs=(t, tau, A, out, A_.dpi)) as pool:
            it = pool.imap_unordered(render_state, states)
            for st, n, kb, s in it:
                tot_f += n; tot_kb += kb
                print(f"{st}: {n} figures, {kb / 1024:.1f} MB, {s:.0f}s", flush=True)
    else:
        _init(t, tau, A, out, A_.dpi)
        for st in states:
            st, n, kb, s = render_state(st)
            tot_f += n; tot_kb += kb
            print(f"{st}: {n} figures, {kb / 1024:.1f} MB, {s:.0f}s", flush=True)
    print(f"{tot_f} figures, {tot_kb / 1024:.0f} MB in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
