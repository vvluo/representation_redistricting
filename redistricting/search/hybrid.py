"""Hybrid search: ReCom moves blocks, single-unit flips polish the boundary."""
from .flip import hill_climb
from .recom import recom_climb


def hybrid(p, rng, **kw):
    """ReCom to move blocks, then flips to polish the boundary, until neither helps."""
    tot_r = tot_f = 0
    while True:
        p, sr, _ = recom_climb(p, rng, **kw)
        p, sf = hill_climb(p)
        tot_r += sr; tot_f += sf
        if sr == 0 and sf == 0: break
        if sr == 0: break
    return p, tot_r, tot_f
