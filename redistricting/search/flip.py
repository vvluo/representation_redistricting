"""Greedy single-unit border flips.

A flip is (i, d): unit i leaves its district and joins district d.  Valid only if
d is new for i, i touches d, the old district keeps >= 1 unit, and the old
district stays connected.
"""
import random
from ..plan import Plan


def hill_climb(p: Plan, max_iter=10000, tol=1e-12):
    cur = p.score(); it = 0
    while it < max_iter:
        best, bf = tol, None
        for i, d in p.candidates():
            if not p.valid(i, d):
                continue
            gain = p.score_after(i, d) - cur
            if gain > best:
                best, bf = gain, (i, d)
        if bf is None:
            break                           # local optimum: no improving flip
        p.apply(*bf); cur = p.score(); it += 1
    return p, it


def hill_climb_first(p, rng, max_iter=10000, tol=1e-12):
    """First-improvement variant: shuffle the candidate flips, take the first one
    that strictly improves f, rescan.  Same stopping rule as hill_climb (no
    improving flip exists) but ~4x fewer objective evaluations per run."""
    cur = p.score(); it = 0
    while it < max_iter:
        cand = [c for c in p.candidates()]
        rng.shuffle(cand)
        bf = None
        for i, d in cand:
            if not p.valid(i, d):
                continue
            if p.score_after(i, d) - cur > tol:
                bf = (i, d); break
        if bf is None:
            break
        p.apply(*bf); cur = p.score(); it += 1
    return p, it
