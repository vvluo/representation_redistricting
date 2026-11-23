"""Initialisation strategies ("restarts").

Every strategy is  fn(g, rng) -> assignment array  and returns a plan with D
non-empty contiguous districts.  random and pop_greedy are contiguous by
construction; sweep, kmeans and spectral can produce disconnected labellings and
pass through repair().
"""
from .growth import seeded_growth, init_random, init_pop_greedy
from .sweep import init_sweep
from .kmeans import init_kmeans
from .spectral import init_spectral
from .repair import repair, components

INITS = {"random": init_random, "pop_greedy": init_pop_greedy,
         "sweep": init_sweep, "kmeans": init_kmeans, "spectral": init_spectral}


def make_init(name, g, rng):
    """Build a starting assignment by strategy name."""
    try:
        return INITS[name](g, rng)
    except KeyError:
        raise ValueError(f"unknown init {name!r}; choose from {sorted(INITS)}") from None
