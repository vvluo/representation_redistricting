"""Exact enumeration and uniform sampling of contiguous partitions (ZDD).

    from redistricting.enumerate import PartitionZDD, population_ok
    z = PartitionZDD(n, edges, p, order="best", xy=xy)
    z.count; z.sample(M, rng); z.all_labels()
"""
from .zdd import PartitionZDD, BuildAborted, district_pops, max_deviation, population_ok
from .ordering import (farthest_pair, order_cut, order_bfs, order_sweep, best_order,
                       edge_order, frontiers, max_frontier)
