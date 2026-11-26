"""Reinforcement-learning interface: protected groups, flip actions, features x(s, a)
and the reward r(s, a) = w . dphi(s, a).  See features.py for the definitions."""
from .groups import Group, build_groups, default_groups, national_thresholds
from .state import DistrictState, articulation
from .features import FlipFeatures, FAMILIES
