"""Action features x(s, a) and objective changes for single-tract flips a = (i, t).

    F = FlipFeatures(g, p, tol=0.05)                # groups, thresholds, clusters
    st = F.state(assignment)                        # incremental DistrictState
    acts, mask = F.actions(st)                      # all boundary flips, validity
    X = F.features(st, acts)                        # (A, F.dim), names in F.names
    R = F.reward(X, {"poverty": 1, "ethnicity": 1, "industry": 1})
    st.apply(i, t)

Objective terms phi (higher is better); the first feature block is their change
dphi = phi(s') - phi(s) under each action:

  population   -sum_d r_d^2,  r_d = pop_d / ideal - 1
  excess       -sum_d max(0, |r_d| - tol): how far districts sit outside the FINAL
               tolerance; zero on valid maps.  With explore_tol > tol the mask lets
               intermediate maps leave tol (up to explore_tol) and this term, weighted,
               pulls them back.
  compactness  mean_d Polsby-Popper 4 pi A_d / P_d^2
  <family>     -L_family, the representation shortfall of one protected family (the
               families in groups.py that have data: poverty, ethnicity, industry,
               rurality, education, ... ):

      L_family = (1/Z) sum_{g in family} sum_{m=1}^{N_g} alpha^(m-1) max(0, 1 - s_g^(m) / tau_g)

      s_g^(m)  the m-th largest district share of group g
      tau_g    the share at which a district represents g     (groups.py)
      N_g      districts that can represent g at once          (groups.py)
      Z        sum of the weights, so L in [0, 1]: 0 = every attainable district
               reaches tau, 1 = none has any of the group

  Losing a group's only district costs weight 1; merging two into one costs alpha.

  coh_<family> border cohesion, an alternative to the quota term: per group
      C_g(z) = sum_E b_ij f_i f_j 1[z_i = z_j] / sum_E b_ij f_i f_j
  (b_ij shared border length, f_i the group's share of tract i), averaged over the
  family's groups.  1 = no border between the group's tracts is cut.
  A group that neutral maps always represent (N_g reached anyway) contributes ~0.

Reward: r(s, a) = sum_k w_k dphi_k over the chosen terms -- a linear function of the
first feature block, so a grid search over w only re-weights precomputed columns.

Feature blocks (names in self.names, slices in self.blocks):
  dphi         objective changes above, plus d_maxdev (change in max |r_d|) and
               d_excess
  tract        unit i: population / ideal, log income (z), poverty, CVAP race shares,
               BA+ share, 13 industry shares
  src, dst     district before the move: signed deviation, |deviation| after the
               move, Polsby-Popper, composition (poverty, race, BA+, income), and per
               family the share of i's main group g* relative to tau and whether the
               district is among g*'s top-N_g
  geo          neighbours (all / in src / in dst), distinct neighbouring districts,
               shared-border fraction with src and dst, outer-boundary fraction,
               distance to the src and dst population centroids
  community    per family: whether i is a high tract for g*, fraction of i's high
               neighbours in src and dst, fraction of i's cluster's population in
               src and dst, cluster population / ideal
  consequence  per family: change in districts spanned by i's cluster
               (+1 = fragments it, -1 = reunites it); population slack to the
               tolerance for src and dst after the move; whether i is an
               articulation unit of src

g*(i) for a family = the group in which tract i is most concentrated relative to
tau.  A cluster = a connected set of tracts whose share of a group is >= tau.
"""
import math
import numpy as np
from .groups import build_groups, group_matrices
from .state import DistrictState

FAMILIES = ("poverty", "ethnicity", "industry", "rurality", "education", "income",
            "class_of_worker", "housing", "health", "food", "language", "group_quarters")
COMPOSITION = [("poverty", ("pov_below",), "pov_total"),
               ("white", ("cvap_white",), "cvap_total"),
               ("black", ("cvap_black",), "cvap_total"),
               ("hispanic", ("cvap_hispanic",), "cvap_total"),
               ("asian", ("cvap_asian",), "cvap_total"),
               ("native", ("cvap_aian", "cvap_nhpi"), "cvap_total"),
               ("ba_plus", ("edu_ba_plus",), "edu_total")]


def _components(units, nbr):
    """Connected components of the subgraph induced by a set of units."""
    units = set(units); comp = {}; k = 0
    for u in units:
        if u in comp:
            continue
        comp[u] = k; st = [u]
        while st:
            x = st.pop()
            for y in nbr[x]:
                if y in units and y not in comp:
                    comp[y] = k; st.append(y)
        k += 1
    return comp, k


class FlipFeatures:
    def __init__(self, g, p, tol=0.05, alpha=0.25, tau=None, families=None, thresholds=None,
                 explore_tol=None):
        self.g, self.p, self.tol, self.alpha = g, p, tol, alpha
        # tol: the final requirement;  explore_tol: the band the action mask enforces on
        # intermediate maps (>= tol; default tol, i.e. every visited map within tol)
        self.explore_tol = tol if explore_tol is None else max(explore_tol, tol)
        self.ideal = g.pop.sum() / p
        self.groups = build_groups(g, p, tol=tol, tau=tau, thresholds=thresholds)
        # families: those with data (in FAMILIES order) unless given; a given family
        # without data yields zero columns, so dimensions can be pinned across states
        have = {gr.family for gr in self.groups}
        self.families = [f for f in FAMILIES if f in have] if families is None else list(families)
        cols = set(g.features.columns)

        # ---- count matrix: pop, area, pop*x, pop*y, pop*log-income-z, then every column
        xy = g.xy
        inc = g.features["median_hh_income"].to_numpy(float) if "median_hh_income" in cols \
            else np.full(g.N, np.nan)
        linc = np.log(np.where(np.isfinite(inc) & (inc > 0), inc, np.nanmedian(inc)))
        self.lz = (linc - linc.mean()) / (linc.std() or 1.0)
        base = [g.pop, g.area, g.pop * xy[:, 0], g.pop * xy[:, 1], g.pop * self.lz]
        self.comp = [c for c in COMPOSITION if c[2] in cols and all(x in cols for x in c[1])]
        num_g, den_g = group_matrices(g, self.groups) if self.groups else (np.zeros((g.N, 0)),) * 2
        num_c = np.column_stack([sum(g.col(x) for x in c[1]) for c in self.comp]) if self.comp \
            else np.zeros((g.N, 0))
        den_c = np.column_stack([g.col(c[2]) for c in self.comp]) if self.comp else np.zeros((g.N, 0))
        G, Q = len(self.groups), len(self.comp)
        self.C = np.column_stack(base + [num_g, den_g, num_c, den_c])
        o = 5
        self.ix_gnum = slice(o, o + G); self.ix_gden = slice(o + G, o + 2 * G)
        self.ix_cnum = slice(o + 2 * G, o + 2 * G + Q); self.ix_cden = slice(o + 2 * G + Q, o + 2 * G + 2 * Q)

        # ---- tract-level static attributes
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tshare = np.where(den_g > 0, num_g / den_g, 0.0)           # (N, G)
            self.tcomp = np.where(den_c > 0, num_c / den_c, 0.0)            # (N, Q)
        self.ind_idx = [k for k, gr in enumerate(self.groups) if gr.family == "industry"]
        self.tau = np.array([gr.tau for gr in self.groups], float)
        self.Ng = np.array([gr.N for gr in self.groups], int)
        self.fam_idx = {f: [k for k, gr in enumerate(self.groups) if gr.family == f] for f in self.families}

        # main group per family and tract; high flags; clusters per group
        self.gstar = {}
        for f, ks in self.fam_idx.items():
            if ks:
                r = self.tshare[:, ks] / np.where(np.isfinite(self.tau[ks]), self.tau[ks], np.inf)
                self.gstar[f] = np.array(ks)[np.argmax(r, axis=1)]
            else:
                self.gstar[f] = np.full(g.N, -1)
        self.high = (self.tshare >= np.where(np.isfinite(self.tau), self.tau, np.inf)) & (g.pop[:, None] > 0)
        self.clusters = []; nxt = 0
        self.cl_of = np.full((g.N, G), -1, dtype=np.int64)
        for k in range(G):
            comp, n = _components(np.flatnonzero(self.high[:, k]).tolist(), g.nbr)
            lab = np.full(g.N, -1, dtype=np.int64)
            for u, c in comp.items():
                lab[u] = c + nxt
            self.clusters.append(lab); self.cl_of[:, k] = lab; nxt += n

        # ---- directed edges (u -> v) with shared length
        E = np.array(list(g.border.keys()), dtype=np.int64).reshape(-1, 2)
        Lk = np.array(list(g.border.values()), float)
        self.du = np.concatenate([E[:, 0], E[:, 1]]); self.dv = np.concatenate([E[:, 1], E[:, 0]])
        self.dl = np.concatenate([Lk, Lk])
        self.deg = np.bincount(self.du, minlength=g.N).astype(float)
        self.outer = np.maximum(g.perim - np.bincount(self.du, self.dl, g.N), 0.0)
        # high-neighbour weights: v is high in u's main group of the family
        # border cohesion: per group w_ij = b_ij f_i f_j / sum_E b f f; family weight = mean over
        # its groups with a non-zero denominator (undirected sum over E = directed sum / 2)
        self.cohw = {}
        for f in self.families:
            W = np.zeros_like(self.dl); n = 0
            for k in self.fam_idx[f]:
                w = self.dl * self.tshare[self.du, k] * self.tshare[self.dv, k]
                Zk = w.sum() / 2
                if Zk > 0:
                    W += w / Zk; n += 1
            self.cohw[f] = W / n if n else W
        self.hiw = {}
        for f in self.families:
            gs = self.gstar[f][self.du]
            self.hiw[f] = np.where(gs >= 0, self.high[self.dv, np.maximum(gs, 0)], False).astype(float)

        # hinge weights c[g, m] = alpha^(m-1) for m <= N_g
        self.Nmax = int(self.Ng.max()) if G else 0
        m = np.arange(max(self.Nmax, 1))
        self.cw = np.where(m[None, :] < self.Ng[:, None], alpha ** m[None, :], 0.0)
        self.Z = {f: float(self.cw[ks].sum()) for f, ks in self.fam_idx.items()}
        self.radius = math.sqrt(g.area.sum() / p / math.pi)
        self._names()

    # ------------------------------------------------------------------ names
    def _names(self):
        fam = self.families
        comp = [c[0] for c in self.comp]
        ind = [self.groups[k].name for k in self.ind_idx]
        dist = lambda pre: ([f"{pre}_dev", f"{pre}_absdev_after", f"{pre}_pp", f"{pre}_log_income"]
                            + [f"{pre}_{c}" for c in comp]
                            + [f"{pre}_{f}_share_over_tau" for f in fam]
                            + [f"{pre}_{f}_in_topN" for f in fam])
        B = {"dphi": ["d_population", "d_compactness"] + [f"d_{f}" for f in fam] + ["d_maxdev", "d_excess"] + [f"d_coh_{f}" for f in fam],
             "tract": ["pop_over_ideal", "log_income"] + [f"t_{c}" for c in comp] + [f"t_{n}" for n in ind]
                      + [f"t_{f}_share_over_tau" for f in fam],
             "src": dist("src"), "dst": dist("dst"),
             "geo": ["n_nbr", "n_nbr_src", "n_nbr_dst", "n_nbr_districts", "border_frac_src",
                     "border_frac_dst", "outer_frac", "dist_src_centroid", "dist_dst_centroid"],
             "community": sum([[f"{f}_is_high", f"{f}_hi_nbr_frac_src", f"{f}_hi_nbr_frac_dst",
                                f"{f}_cluster_frac_src", f"{f}_cluster_frac_dst", f"{f}_cluster_size"]
                               for f in fam], []),
             "consequence": [f"{f}_d_span" for f in fam] + ["src_slack_after", "dst_slack_after",
                                                            "src_articulation"]}
        self.names, self.blocks, o = [], {}, 0
        for b, nm in B.items():
            self.blocks[b] = slice(o, o + len(nm)); self.names += nm; o += len(nm)
        self.dim = o

    # ------------------------------------------------------------------ state
    def state(self, a):
        return DistrictState(self.g, a, self.p, self.C, self.clusters)

    def actions(self, st):
        """All boundary flips (i, t), t != a[i] a district touching i; and a validity
        mask: src keeps >= 1 unit and stays connected, and neither district moves
        outside the population tolerance unless it moves toward it."""
        a = st.a
        cross = a[self.du] != a[self.dv]
        key = np.unique(self.du[cross] * self.p + a[self.dv[cross]])
        acts = np.column_stack([key // self.p, key % self.p])
        mask = self._valid(st, acts)
        return acts, mask

    def _valid(self, st, acts):
        i, t = acts[:, 0], acts[:, 1]; s = st.a[i]
        P = st.sums[:, 0]; pi = self.g.pop[i]
        r_s0 = np.abs(P[s] / self.ideal - 1); r_s1 = np.abs((P[s] - pi) / self.ideal - 1)
        r_t0 = np.abs(P[t] / self.ideal - 1); r_t1 = np.abs((P[t] + pi) / self.ideal - 1)
        e = self.explore_tol
        ok_pop = ((r_s1 <= e) | (r_s1 < r_s0)) & ((r_t1 <= e) | (r_t1 < r_t0))
        art = np.array([u in st.artic[d] for u, d in zip(i.tolist(), s.tolist())], bool)
        return (st.size[s] > 1) & ~art & ok_pop

    # ------------------------------------------------------------------ objectives
    def _shares(self, S, num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(S[..., den] > 0, S[..., num] / S[..., den], 0.0)

    def _pp(self, area, perim):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(perim > 0, 4 * math.pi * area / perim ** 2, 0.0)

    def _hinge(self, vals, ks):
        """vals (..., len(ks), Nmax) sorted high->low -> weighted shortfall per row."""
        tau = self.tau[ks][:, None]
        h = np.maximum(0.0, 1.0 - vals / tau)
        return (h * self.cw[ks]).sum(axis=(-1, -2))

    def phi(self, st):
        """Objective values of a state, computed from scratch (no caches)."""
        a = st.a; p = self.p
        S = np.zeros((p, self.C.shape[1])); np.add.at(S, a, self.C)
        per = np.bincount(a, self.g.perim, p)
        for (i, j), L in self.g.border.items():
            if a[i] == a[j]:
                per[a[i]] -= 2 * L
        r = S[:, 0] / self.ideal - 1
        out = {"population": -float((r ** 2).sum()), "compactness": float(self._pp(S[:, 1], per).mean()),
               "maxdev": float(np.abs(r).max()),
               "excess": -float(np.maximum(np.abs(r) - self.tol, 0).sum())}
        sh = self._shares(S, self.ix_gnum, self.ix_gden).T            # (G, p)
        srt = -np.sort(-sh, axis=1)
        for f in self.families:
            ks = self.fam_idx[f]
            if not ks or self.Z[f] == 0:
                out[f] = 0.0; continue
            v = srt[ks][:, :self.Nmax] if self.Nmax <= p else srt[ks]
            out[f] = -float(self._hinge(v, ks)) / self.Z[f]
        same = a[self.du] == a[self.dv]
        for f in self.families:
            out[f"coh_{f}"] = float(self.cohw[f][same].sum() / 2)
        return out

    def _dphi(self, st, acts):
        i, t = acts[:, 0], acts[:, 1]; s = st.a[i]; A = len(acts)
        Ci = self.C[i]
        Ss, St = st.sums[s] - Ci, st.sums[t] + Ci
        # perimeters after the move
        nb_len = self._nb(st, self.dl)
        Ls, Lt = nb_len[i, s], nb_len[i, t]
        Ps = st.perim[s] - self.g.perim[i] + 2 * Ls
        Pt = st.perim[t] + self.g.perim[i] - 2 * Lt
        r = st.sums[:, 0] / self.ideal - 1
        rs1, rt1 = Ss[:, 0] / self.ideal - 1, St[:, 0] / self.ideal - 1
        d_pop = -(rs1 ** 2 + rt1 ** 2 - r[s] ** 2 - r[t] ** 2)
        pp = self._pp(st.sums[:, 1], st.perim)
        d_comp = (self._pp(Ss[:, 1], Ps) + self._pp(St[:, 1], Pt) - pp[s] - pp[t]) / self.p
        # max |r| after: max over untouched districts and the two new values
        absr = np.abs(r); top3 = np.argsort(-absr)[:3]
        rest = np.array([max([absr[d] for d in top3 if d != a and d != b] or [0.0])
                         for a, b in zip(s.tolist(), t.tolist())])
        d_max = np.maximum(rest, np.maximum(np.abs(rs1), np.abs(rt1))) - absr.max()
        out = [d_pop, d_comp]
        sh = self._shares(st.sums, self.ix_gnum, self.ix_gden).T      # (G, p)
        shs = self._shares(Ss, self.ix_gnum, self.ix_gden)             # (A, G)
        sht = self._shares(St, self.ix_gnum, self.ix_gden)
        K = min(self.Nmax + 2, self.p)
        for f in self.families:
            ks = self.fam_idx[f]
            if not ks or self.Z[f] == 0:
                out.append(np.zeros(A)); continue
            cur = -np.sort(-sh[ks], axis=1)[:, :self.Nmax]
            L0 = self._hinge(cur, ks)
            idx = np.argsort(-sh[ks], axis=1)[:, :K]                    # (Gf, K)
            val = np.take_along_axis(sh[ks], idx, axis=1)
            V = np.broadcast_to(val, (A,) + val.shape).copy()
            drop = (idx[None] == s[:, None, None]) | (idx[None] == t[:, None, None])
            V[drop] = -np.inf
            V = np.concatenate([V, shs[:, ks, None], sht[:, ks, None]], axis=2)
            V = -np.sort(-V, axis=2)[:, :, :self.Nmax]
            L1 = self._hinge(V, ks)
            out.append(-(L1 - L0) / self.Z[f])
        out.append(d_max)
        ex = lambda x: np.maximum(np.abs(x) - self.tol, 0.0)
        out.append(-(ex(rs1) + ex(rt1) - ex(r[s]) - ex(r[t])))
        for f in self.families:                     # edges to t become internal, edges to s cut
            nb = self._nb(st, self.cohw[f])
            out.append(nb[i, t] - nb[i, s])
        return np.column_stack(out), dict(Ss=Ss, St=St, rs1=rs1, rt1=rt1, Ls=Ls, Lt=Lt, sh=sh, nb_len=nb_len)

    def _nb(self, st, w):
        """(N, p): sum over i's neighbours v in district d of w(i -> v)."""
        return np.bincount(self.du * self.p + st.a[self.dv], w, self.g.N * self.p).reshape(self.g.N, self.p)

    # ------------------------------------------------------------------ features
    def features(self, st, acts):
        i, t = acts[:, 0], acts[:, 1]; s = st.a[i]; A = len(acts); p = self.p
        dphi, aux = self._dphi(st, acts)
        cols = [dphi]

        # tract block
        fam_ratio = []
        for f in self.families:
            gs = self.gstar[f][i]
            fam_ratio.append(np.where(gs >= 0, self.tshare[i, np.maximum(gs, 0)] /
                                      np.where(gs >= 0, self.tau[np.maximum(gs, 0)], 1.0), 0.0))
        cols.append(np.column_stack([self.g.pop[i] / self.ideal, self.lz[i], self.tcomp[i],
                                     self.tshare[i][:, self.ind_idx]] + fam_ratio))

        # district blocks (before the move)
        S = st.sums; r = S[:, 0] / self.ideal - 1
        pp = self._pp(S[:, 1], st.perim)
        inc = np.where(S[:, 0] > 0, S[:, 4] / np.maximum(S[:, 0], 1e-12), 0.0)
        comp = self._shares(S, self.ix_cnum, self.ix_cden)            # (p, Q)
        sh = aux["sh"]                                                 # (G, p)
        rank = np.argsort(np.argsort(-sh, axis=1), axis=1)             # 0 = highest share
        for d, r_after in ((s, aux["rs1"]), (t, aux["rt1"])):
            fr, top = [], []
            for f in self.families:
                gs = self.gstar[f][i]; k = np.maximum(gs, 0)
                fr.append(np.where(gs >= 0, sh[k, d] / self.tau[k], 0.0))
                top.append(np.where(gs >= 0, rank[k, d] < self.Ng[k], False).astype(float))
            cols.append(np.column_stack([r[d], np.abs(r_after), pp[d], inc[d], comp[d]] + fr + top))

        # local geography
        nb_cnt = self._nb(st, np.ones_like(self.dl))
        nb_len = aux["nb_len"]
        per = np.maximum(self.g.perim[i], 1e-12)
        cen = S[:, 2:4] / np.maximum(S[:, :1], 1e-12)
        xy = self.g.xy[i]
        cols.append(np.column_stack([
            self.deg[i], nb_cnt[i, s], nb_cnt[i, t], (nb_cnt[i] > 0).sum(axis=1),
            nb_len[i, s] / per, nb_len[i, t] / per, self.outer[i] / per,
            np.linalg.norm(xy - cen[s], axis=1) / self.radius,
            np.linalg.norm(xy - cen[t], axis=1) / self.radius]))

        # community structure and consequences
        comm, cons = [], []
        for f in self.families:
            gs = self.gstar[f][i]; k = np.maximum(gs, 0); has = gs >= 0
            hn = self._nb(st, self.hiw[f])
            nh = np.maximum(hn[i].sum(axis=1), 1e-12)
            is_high = has & self.high[i, k]
            cl = np.where(is_high, self.cl_of[i, k], -1); c = np.maximum(cl, 0)
            ctot = np.maximum(st.cpop[c].sum(axis=1), 1e-12)
            on = cl >= 0
            comm += [is_high.astype(float), hn[i, s] / nh * (hn[i].sum(axis=1) > 0),
                     hn[i, t] / nh * (hn[i].sum(axis=1) > 0),
                     np.where(on, st.cpop[c, s] / ctot, 0.0), np.where(on, st.cpop[c, t] / ctot, 0.0),
                     np.where(on, st.cpop[c].sum(axis=1) / self.ideal, 0.0)]
            span = -(st.ccount[c, s] == 1).astype(float) + (st.ccount[c, t] == 0).astype(float)
            cons.append(np.where(on, span, 0.0))
        art = np.array([u in st.artic[d] for u, d in zip(i.tolist(), s.tolist())], float)
        cons += [self.tol - np.abs(aux["rs1"]), self.tol - np.abs(aux["rt1"]), art]
        cols.append(np.column_stack(comm)); cols.append(np.column_stack(cons))
        X = np.column_stack(cols)
        assert X.shape == (A, self.dim)
        return X

    # ------------------------------------------------------------------ reward
    def reward(self, X, w):
        """r = sum_k w_k * dphi_k;  w maps term names (population, compactness, excess,
        maxdev, and the family names) to weights."""
        b = self.blocks["dphi"]; names = self.names[b]
        W = np.array([w.get(n[2:], 0.0) for n in names])
        return X[:, b] @ W

    def representation(self, st):
        """Cracking metrics per group: n = districts at or above tau, N = attainable,
        cracked = N >= 1 but n = 0, shortfall = max(0, N - n).  Returns (per-group rows,
        per-family totals)."""
        sh = self._shares(st.sums, self.ix_gnum, self.ix_gden).T          # (G, p)
        rows = []
        for k, gr in enumerate(self.groups):
            n = int((sh[k] >= gr.tau).sum()) if np.isfinite(gr.tau) else 0
            rows.append(dict(group=gr.name, family=gr.family, tau=gr.tau, N=gr.N, n=n,
                             best_share=float(sh[k].max()), cracked=bool(gr.N >= 1 and n == 0),
                             shortfall=max(0, gr.N - n)))
        fam = {}
        for f in self.families:
            R = [r for r in rows if r["family"] == f and r["N"] >= 1]
            fam[f] = dict(groups_at_risk=len(R), groups_cracked=sum(r["cracked"] for r in R),
                          district_shortfall=sum(r["shortfall"] for r in R))
        return rows, fam

    def summary(self):
        rows = [(gr.name, gr.family, round(gr.tau, 4), gr.N, gr.tau_source) for gr in self.groups]
        return rows
