"""The districting objective, defined exactly once.

Maximised:   f = mean_d PolsbyPopper(d)  -  lambda * U(plan)

Polsby-Popper is 4*pi*A / P^2 (1 for a circle).  U is a population-dispersion
penalty, selected by `mode`:

  tvd      total variation distance from uniform, 0.5 * sum_d |p_d - 1/D|.
           Equals 0.5 * mean_d |r_d|: the share of the state's population that
           would have to move to reach exact equality.  Linear in deviation, so
           its gradient never vanishes.  (default)
  rmsd     sqrt(mean_d r_d^2), with r_d = (pop_d - ideal) / ideal.
  entropy  1 - H, H the normalised Shannon entropy of population shares.

Entropy is quadratically flat at uniformity (1 - H grows like delta^2 while TVD
grows like delta), so it goes blind exactly where apportionment accuracy is
decided: a 10% deviation still scores H = 0.99958.  It is reported but should not
drive a search.  Every evaluator -- cached, incremental and rebuilt -- calls
objective(), so the formula cannot drift between code paths.
"""
import math
import numpy as np


def objective(pop, area, perim, ideal, D, lam, mode):
    """f = mean Polsby-Popper - lam * population-dispersion penalty.

    The single definition of the objective; every evaluator calls this so the
    cached, incremental and rebuilt paths cannot drift apart.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        pp = np.where(perim > 0, 4 * math.pi * area / (perim ** 2), 0.0)
    tot = pop.sum()
    if mode == "tvd":
        pen = 0.5 * float(np.abs(pop / tot - 1.0 / D).sum())
    elif mode == "rmsd":
        pen = math.sqrt((((pop - ideal) / ideal) ** 2).mean())
    else:
        q = pop / tot; nz = q[q > 0]
        pen = 1.0 - float(-(nz * np.log(nz)).sum() / math.log(D))
    return float(pp.mean()) - lam * pen
