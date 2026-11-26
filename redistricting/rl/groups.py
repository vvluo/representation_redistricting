"""Protected groups: what each one counts, when a district represents it, and how
many representing districts the map can supply at once.

A group g is a (numerator, denominator) pair of count columns, e.g. people below
poverty / people with poverty status.  Its share in a district is the ratio of the
district sums.  Groups belong to families; the reward has one signal per family:

  family           groups                                        data
  poverty          below the poverty line / poverty universe     have
  ethnicity        Black, Hispanic, Asian, Native CVAP / CVAP    have
  industry         workers in each of 13 ACS sectors / employed  have
  rurality         population in micropolitan (RUCA 4-6), small  have (RUCA codes)
                   town (7-9) and rural (10) tracts / population
  education        BA+ / adults 25+;  less than high school      BA+ have; <HS: B15003
  income           households < $35k;  >= $150k / households     B19001
  class_of_worker  government, self-employed, non-profit         C24080 / S2408
                   workers / civilian employed
  housing          renter households / occupied;  rent >= 30%    B25003, B25070, B25002
                   of income / renters;  vacant / housing units
  health           uninsured / civilian non-institutionalised    B27010 / S2701
  food             SNAP households / households                  B22003
  language         speak Spanish, other Indo-European, Asian /   B16007
                   Pacific Island, other languages at home / 5+
  group_quarters   adult correctional, nursing-facility and      B26103
                   college-housing residents / population

A family is used only when its columns exist in the unit features, so adding an ACS
table to prep switches it on without code changes.

tau_g  the share at which a district counts as representing g.  Default: the US
       tract mean + 1 SD of the group's share (the USDA ERS concentration rule,
       applied to tracts), computed once from the national tract master and stored
       in data/group_thresholds.csv.  Groups with no national value (the extra ACS
       tables until they are fetched for every state) use the same rule over the
       map's own tracts (tau_source = "state_tracts").  Override with `tau=`.
N_g    how many districts can represent g at once: a greedy lower bound (attainable).
       N_g = 0 means districting cannot give g a district, so g carries no reward.

    python -m redistricting.rl.groups --master ../extracted/tract_master.csv
"""
import argparse, heapq
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from .. import config

SECTORS = {1: "agriculture_mining", 2: "arts_lodging_food", 3: "construction",
           4: "education_health", 5: "finance_real_estate", 6: "information",
           7: "manufacturing", 8: "other_services", 9: "professional_admin",
           10: "public_administration", 11: "retail", 12: "transport_utilities",
           13: "wholesale"}


@dataclass
class Group:
    name: str
    family: str
    num: tuple                  # numerator count columns (summed)
    den: str                    # denominator count column
    tau: float = np.nan         # district share that counts as representation
    N: int = 0                  # attainable representing districts (set by attainable())
    tau_source: str = ""        # "given", "us_tracts" or "state_tracts"


RUCA_TIERS = {"micropolitan": (4, 6), "small_town": (7, 9), "rural": (10, 10)}


def derive_counts(M):
    """Add the derived count columns to a tract table with master-style columns:
    ind_kk (workers per sector, from percentage shares) and pop_<tier> (population of
    tracts in each non-metro RUCA tier)."""
    M = M.copy()
    if "emp_total" in M:
        for k in SECTORS:
            if f"ind_{k:02d}_share" in M:
                M[f"ind_{k:02d}"] = (M[f"ind_{k:02d}_share"].fillna(0) / 100 * M.emp_total.fillna(0))
    if "ruca_primary" in M:
        for tier, (lo, hi) in RUCA_TIERS.items():
            M[f"pop_{tier}"] = np.where(M.ruca_primary.between(lo, hi), M["pop"].fillna(0), 0.0)
    return M


def default_groups(columns):
    """All groups whose columns exist in a feature table."""
    G = [Group("poverty", "poverty", ("pov_below",), "pov_total"),
         Group("black", "ethnicity", ("cvap_black",), "cvap_total"),
         Group("hispanic", "ethnicity", ("cvap_hispanic",), "cvap_total"),
         Group("asian", "ethnicity", ("cvap_asian",), "cvap_total"),
         Group("native", "ethnicity", ("cvap_aian", "cvap_nhpi"), "cvap_total")]
    G += [Group(f"ind_{k:02d}_{s}", "industry", (f"ind_{k:02d}",), "emp_total")
          for k, s in SECTORS.items()]
    G += [Group(t, "rurality", (f"pop_{t}",), "pop") for t in RUCA_TIERS]
    G += [Group("ba_plus", "education", ("edu_ba_plus",), "edu_total"),
          Group("less_than_hs", "education", ("edu_lt_hs",), "edu_total"),
          Group("low_income", "income", ("hh_inc_lt35k",), "hh_inc_total"),
          Group("high_income", "income", ("hh_inc_ge150k",), "hh_inc_total"),
          Group("government_workers", "class_of_worker", ("cow_government",), "cow_total"),
          Group("self_employed", "class_of_worker", ("cow_self_employed",), "cow_total"),
          Group("nonprofit_workers", "class_of_worker", ("cow_nonprofit",), "cow_total"),
          Group("renters", "housing", ("hh_renter",), "hh_occupied"),
          Group("rent_burdened", "housing", ("rent_burden30",), "rent_total"),
          Group("vacant", "housing", ("hu_vacant",), "hu_total"),
          Group("uninsured", "health", ("hi_uninsured",), "hi_total"),
          Group("snap", "food", ("snap_hh",), "hh_total"),
          Group("spanish_speakers", "language", ("lang_spanish",), "lang_total"),
          Group("indo_european_speakers", "language", ("lang_indo_european",), "lang_total"),
          Group("asian_pacific_speakers", "language", ("lang_asian_pacific",), "lang_total"),
          Group("other_language_speakers", "language", ("lang_other",), "lang_total"),
          Group("correctional", "group_quarters", ("gq_correctional",), "gq_pop_total"),
          Group("nursing_facilities", "group_quarters", ("gq_nursing",), "gq_pop_total"),
          Group("college_housing", "group_quarters", ("gq_college",), "gq_pop_total")]
    cols = set(columns)
    return [g for g in G if g.den in cols and all(c in cols for c in g.num)]


# ------------------------------------------------------------------ thresholds
def national_thresholds(master_csv, acs_csv=None):
    """US tract mean + 1 SD of each group's share, over tracts with a denominator.
    acs_csv: national output of redistricting.acs, adding the extra ACS groups."""
    M = derive_counts(pd.read_csv(master_csv, dtype={"geoid": str}))
    if acs_csv:
        X = pd.read_csv(acs_csv, dtype={"geoid": str})
        M = M.merge(X, on="geoid", how="left")
    rows = []
    for g in default_groups(M.columns):
        den = M[g.den]; ok = den > 0
        sh = M.loc[ok, list(g.num)].sum(axis=1) / den[ok]
        rows.append(dict(group=g.name, family=g.family, us_tract_mean=sh.mean(),
                         us_tract_sd=sh.std(), tau=sh.mean() + sh.std(), tracts=int(ok.sum())))
    return pd.DataFrame(rows)


def load_thresholds(path=None):
    p = Path(path) if path else config.data_dir() / "group_thresholds.csv"
    return pd.read_csv(p).set_index("group").tau.to_dict() if p.exists() else {}


def group_matrices(g, groups):
    """(num, den) arrays of shape (N_units, G)."""
    num = np.column_stack([sum(g.col(c) for c in gr.num) for gr in groups])
    den = np.column_stack([g.col(gr.den) for gr in groups])
    return num, den


# ------------------------------------------------------------------ attainability
def _grow(g, seed, share_num, share_den, used, lo):
    """Grow a contiguous region from `seed`, always adding the frontier unit with the
    highest group share, until its population reaches lo.  Returns (units, share)."""
    region = [seed]; inr = {seed}
    P = g.pop[seed]; A = share_num[seed]; B = share_den[seed]
    heap = []
    def push(u):
        for v in g.nbr[u]:
            if v not in inr and not used[v]:
                s = share_num[v] / share_den[v] if share_den[v] > 0 else 0.0
                heapq.heappush(heap, (-s, -g.pop[v], v))
    push(seed)
    while P < lo and heap:
        _, _, v = heapq.heappop(heap)
        if v in inr:
            continue
        region.append(v); inr.add(v)
        P += g.pop[v]; A += share_num[v]; B += share_den[v]
        push(v)
    if P < lo:
        return None, 0.0
    return region, (A / B if B > 0 else 0.0)


def attainable(g, groups, p, tol=0.05, seeds=12, cap=None):
    """Set gr.N for each group: greedily carve districts (population >= ideal*(1-tol))
    whose group share reaches tau, best of `seeds` seeds each round, until none does.
    A lower bound on what a plan could achieve; it ignores whether the rest of the map
    can still be split into valid districts."""
    ideal = g.pop.sum() / p; lo = ideal * (1 - tol)
    num, den = group_matrices(g, groups)
    for k, gr in enumerate(groups):
        if not np.isfinite(gr.tau):
            gr.N = 0; continue
        used = np.zeros(g.N, bool); n = 0
        share = np.where(den[:, k] > 0, num[:, k] / np.maximum(den[:, k], 1e-12), 0.0)
        while n < (cap or p):
            order = [u for u in np.argsort(-share) if not used[u] and g.pop[u] > 0][:seeds]
            best = None
            for s in order:
                reg, sh = _grow(g, s, num[:, k], den[:, k], used, lo)
                if reg is not None and sh >= gr.tau and (best is None or sh > best[1]):
                    best = (reg, sh)
            if best is None:
                break
            used[best[0]] = True; n += 1
        gr.N = n
    return groups


def build_groups(g, p, tol=0.05, tau=None, families=None, thresholds=None):
    """Groups for unit map g and p districts, with tau and N set."""
    th = load_thresholds(thresholds)
    groups = default_groups(g.features.columns)
    if families:
        groups = [gr for gr in groups if gr.family in families]
    num, den = group_matrices(g, groups) if groups else (None, None)
    for k, gr in enumerate(groups):
        if gr.name in (tau or {}):
            gr.tau, gr.tau_source = tau[gr.name], "given"
        elif gr.name in th:
            gr.tau, gr.tau_source = th[gr.name], "us_tracts"
        else:                    # no national value yet: same rule over this map's tracts
            ok = den[:, k] > 0
            sh = num[ok, k] / den[ok, k]
            gr.tau = float(sh.mean() + sh.std(ddof=1)) if ok.sum() > 1 else np.nan
            gr.tau_source = "state_tracts"
    return attainable(g, groups, p, tol)


def main():
    ap = argparse.ArgumentParser(description="write data/group_thresholds.csv")
    ap.add_argument("--master", required=True)
    ap.add_argument("--acs", help="national redistricting.acs output (optional)")
    ap.add_argument("--out", default=str(config.data_dir() / "group_thresholds.csv"))
    A = ap.parse_args()
    T = national_thresholds(A.master, A.acs)
    T.to_csv(A.out, index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
