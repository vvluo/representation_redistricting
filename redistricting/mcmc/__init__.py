"""MCMC samplers over contiguous plans, for validation against the ZDD.

FlipChain   single-flip Metropolis-Hastings, uniform target (the sampler under test)
ReComChain  plain ReCom, NOT uniform (negative control)
"""
from .flip import FlipChain
from .recom import ReComChain
from .diagnostics import autocorr_time, ess, rhat, compare
