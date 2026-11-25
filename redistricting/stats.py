"""Plan statistics for comparing samplers, vectorised over many plans at once.

Every function takes labels L of shape (M, N) (district of each unit, 0..p-1, one
plan per row) and returns per-plan values.  All statistics are invariant to how
districts are numbered, so plans from different samplers compare directly.

  pop_dev        max_d |pop_d / ideal - 1|
  mean_pp        mean over districts of Polsby-Popper 4*pi*A/P^2, with
                 P_d = sum of unit perimeters - 2 * shared borders inside d
  dissimilarity  D = 1/2 sum_d |a_d/A - b_d/B| for two complementary groups
                 (D_race: minority vs non-Hispanic-white CVAP;
                  D_edu: BA+ vs not, among 25+;  D_pov: below vs above poverty line)
  minority_share minority CVAP share of each district, sorted high -> low (M, p)
  vi             population-weighted variation of information between two plans
"""
import math
import numpy as np


def district_sums(L, w, p):
    M, N = L.shape
    flat = L.astype(np.int64) + (np.arange(M) * p)[:, None]
    return np.bincount(flat.ravel(), weights=np.tile(np.asarray(w, float), M),
                       minlength=M * p).reshape(M, p)


def pop_dev(L, g, p):
    P = district_sums(L, g.pop, p)
    return np.abs(P / (g.pop.sum() / p) - 1).max(axis=1)


def district_perimeters(L, g, p):
    M = L.shape[0]
    per = district_sums(L, g.perim, p)
    if g.border:
        E = np.array(list(g.border.keys())); Lk = np.array(list(g.border.values()))
        same = L[:, E[:, 0]] == L[:, E[:, 1]]
        r, e = np.nonzero(same)
        lab = L[r, E[e, 0]].astype(np.int64) + r * p
        per -= 2 * np.bincount(lab, weights=Lk[e], minlength=M * p).reshape(M, p)
    return per


def mean_pp(L, g, p):
    A = district_sums(L, g.area, p)
    P = district_perimeters(L, g, p)
    return (4 * math.pi * A / P ** 2).mean(axis=1)


def dissimilarity(L, a, b, p):
    Sa = district_sums(L, a, p); Sb = district_sums(L, b, p)
    return 0.5 * np.abs(Sa / Sa.sum(1, keepdims=True) - Sb / Sb.sum(1, keepdims=True)).sum(1)


def groups(g):
    """The complementary group pairs used by the D indices."""
    cv = g.col("cvap_total"); white = g.col("cvap_white")
    edu = g.col("edu_total"); ba = g.col("edu_ba_plus")
    pv = g.col("pov_total"); below = g.col("pov_below")
    return {"race": (cv - white, white), "edu": (ba, edu - ba), "pov": (below, pv - below)}


def minority_share(L, g, p):
    cv = district_sums(L, g.col("cvap_total"), p)
    mi = district_sums(L, g.col("cvap_total") - g.col("cvap_white"), p)
    return -np.sort(-(mi / np.maximum(cv, 1e-12)), axis=1)


def all_stats(L, g, p):
    """Dict of 1-D arrays (plus minority-share ranks as min1..minp)."""
    out = {"pop_dev": pop_dev(L, g, p), "mean_pp": mean_pp(L, g, p)}
    if "cvap_total" in g.features.columns:
        for k, (a, b) in groups(g).items():
            out[f"D_{k}"] = dissimilarity(L, a, b, p)
        ms = minority_share(L, g, p)
        for r in range(p):
            out[f"min{r+1}"] = ms[:, r]
    return out


def vi(x, y, w, p):
    """Variation of information between plans x and y (pop-weighted), in nats."""
    w = np.asarray(w, float); w = w / w.sum()
    J = np.bincount(x.astype(np.int64) * p + y, weights=w, minlength=p * p).reshape(p, p)
    px, py = J.sum(1), J.sum(0)
    nz = J > 0
    return float(-(J[nz] * (np.log(J[nz] / px[:, None].repeat(p, 1)[nz])
                            + np.log(J[nz] / py[None, :].repeat(p, 0)[nz]))).sum())


def pairwise_vi(L, w, p, pairs, rng):
    """VI for `pairs` random pairs of distinct rows of L."""
    M = L.shape[0]
    i = rng.integers(0, M, pairs); j = (i + rng.integers(1, M, pairs)) % M
    return np.array([vi(L[a], L[b], w, p) for a, b in zip(i, j)])
