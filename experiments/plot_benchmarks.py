"""Histograms of each plan metric: ZDD (truth) against MCMC, one panel per metric.

Reads the per-plan statistics saved by mcmc_benchmark.py / mcmc_convergence.py and
writes outputs/figures/hist_<bench>.png.  Panel titles carry the KS distance D and
the median KS p over thinning offsets (see README, "MCMC validated against the ZDD").

    python -m experiments.plot_benchmarks --bench burl2 burl3 ia4
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcmc_benchmark import load
from redistricting import stats
from redistricting.mcmc import autocorr_time

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated: categorical slots 1-3

LABEL = {"pop_dev": "Population deviation", "mean_pp": "Compactness (mean Polsby-Popper)",
         "D_race": "D_race (minority vs white CVAP)", "D_edu": "D_education (BA+ vs not)",
         "D_pov": "D_income stand-in (poverty)", "vi": "Pairwise variation of information",
         "min1": "Minority CVAP share, rank 1", "min2": "Minority CVAP share, rank 2",
         "min3": "Minority CVAP share, rank 3", "min4": "Minority CVAP share, rank 4"}
ORDER = ["pop_dev", "mean_pp", "D_race", "D_edu", "D_pov", "vi", "min1", "min2", "min3", "min4"]


def ks_summary(ref, x, is_vi=False):
    D = ks_2samp(ref, x).statistic
    if is_vi:
        return D, ks_2samp(ref, x).pvalue
    s = max(1, int(round(3 * autocorr_time(x))))
    return D, float(np.median([ks_2samp(ref, x[o::s]).pvalue for o in range(s)]))


def data(bench, out):
    """(title, ref dict, [(label, color, linestyle, dict)])"""
    if bench == "burl2":
        d = np.load(out / "mcmc_burl2_tol0.05_stats.npz")
        ref = {k[4:]: d[k] for k in d if k.startswith("ref_")}
        mc = {k[5:]: d[k] for k in d if k.startswith("flip_")}
        return ("70 tracts around Burlington, 2 districts, within 5%  "
                "(ZDD: all 464,122 valid plans;  MCMC: 4 chains x 4M steps)",
                ref, [("MCMC, walk within 10%", ORANGE, "-", mc)])
    if bench == "burl3":
        g, p = load("burl3")
        R = np.load(out / "ref_burl3_tol0.05_100k.npz")["L"]
        ref = stats.all_stats(R, g, p)
        ref["vi"] = stats.pairwise_vi(R, g.pop, p, 20000, np.random.default_rng(5))
        d = np.load(out / "rep" / "mcmc_burl3_tol0.05_stats.npz")
        mc = {k[5:]: d[k] for k in d if k.startswith("flip_")}
        return ("70 tracts around Burlington, 3 districts, within 5%  "
                "(ZDD: 100k uniform samples;  MCMC: 4 chains x 12M steps)",
                ref, [("MCMC, walk within 10%", ORANGE, "-", mc)])
    if bench == "ia4":
        d = np.load(out / "mcmc_convergence_ia4_tol0.05_walk0.2_stats.npz")
        ref = {k[4:]: d[k] for k in d if k.startswith("ref_")}
        w20 = {k[8:]: d[k] for k in d if k.startswith("walk0.2_")}
        e = np.load(out / "mcmc_ia4_tol0.05_stats.npz")
        w10 = {k[5:]: e[k] for k in e if k.startswith("flip_")}
        return ("Iowa, 99 counties, 4 districts, within 5%  "
                "(ZDD: 5,000 uniform samples;  MCMC: 4 chains x 50M steps at 20%, x 8M at 10%)",
                ref, [("MCMC, walk within 20%", ORANGE, "-", w20),
                      ("MCMC, walk within 10% (not converged)", AQUA, "--", w10)])
    raise ValueError(bench)


def plot(bench, out):
    title, ref, series = data(bench, out)
    keys = [k for k in ORDER if k in ref and all(k in s[3] for s in series)]
    ncol = 3; nrow = -(-len(keys) // ncol)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow + 0.9),
                             facecolor=SURFACE, squeeze=False)
    for ax, k in zip(axes.flat, keys):
        r = np.asarray(ref[k], float)
        allv = np.concatenate([r] + [np.asarray(s[3][k], float) for s in series])
        lo, hi = np.quantile(allv, [0.001, 0.999])
        bins = np.linspace(lo, hi, 41)
        ax.set_facecolor(SURFACE)
        ax.hist(r, bins=bins, density=True, color=BLUE, alpha=0.28, lw=0)
        ax.hist(r, bins=bins, density=True, histtype="step", color=BLUE, lw=1.6)
        notes = []
        for lab, col, ls, s in series:
            x = np.asarray(s[k], float)
            h, _ = np.histogram(x, bins=bins, density=True)
            ax.stairs(h, bins, color=col, lw=2, linestyle=(0, (5, 2.5)) if ls == "--" else "-")
            D, pv = ks_summary(r, x, k == "vi")
            tag = (lab.split("within ")[1].split()[0] + ": ") if len(series) > 1 else ""
            notes.append(f"{tag}D={D:.3f}, p={pv:.2g}")
        ax.set_title(LABEL[k], fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, ax.get_ylim()[1] * (1.18 + 0.1 * (len(notes) - 1)))   # headroom for the label
        ax.text(0.99, 0.98, "\n".join(notes), transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK2)
        ax.set_yticks([]); ax.grid(axis="x", color=GRID, lw=0.6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.4, ec=BLUE, lw=1.6)]
    labels = ["ZDD (exact uniform truth)"]
    for lab, col, ls, _ in series:
        handles.append(plt.Line2D([0], [0], color=col, lw=2, ls=ls)); labels.append(lab)
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.01, 1 - 0.45 / (2.9 * nrow + 0.9)))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    note = ("Densities over the same bins.  D = two-sample KS distance on all MCMC states; "
            "p = median KS p over thinning offsets at 3x the autocorrelation time.")
    if bench == "ia4":
        note += ("\nThe 10% walk is so autocorrelated (29-148 effective plans for D_race and minority "
                 "ranks 2-4) that its p-values have little power; read its D values and the curves.")
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.035 if bench == "ia4" else 0.02, 1, 1 - 0.75 / (2.9 * nrow + 0.9)))
    f = out / "figures" / f"hist_{bench}.png"; f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=150, facecolor=SURFACE); plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", nargs="+", default=["burl2", "burl3", "ia4"])
    ap.add_argument("--out-dir", default="outputs")
    A = ap.parse_args()
    for b in A.bench:
        print(plot(b, ROOT / A.out_dir if not Path(A.out_dir).is_absolute() else Path(A.out_dir)))


if __name__ == "__main__":
    main()
