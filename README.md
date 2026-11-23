# representation_redistricting

Compactness- and population-driven redistricting by local search on a unit graph
(counties or census tracts): one objective, five ways to start, three move sets.

```python
import random
from redistricting import Geo, Plan, make_init, hybrid

g = Geo.load("AL_tract", D=7)                      # 1,437 tracts -> 7 districts
rng = random.Random(0)
p = Plan(g, make_init("spectral", g, rng), lam=6.0, mode="tvd")
p, n_recom, n_flips = hybrid(p, rng, tree="compact", tol=0.01)
print(p.report())
```

## Objective

Maximise `f = mean_d PolsbyPopper(d) - lambda * TVD(population)`.

- **Polsby-Popper** is `4*pi*A / P^2` per district (1 for a circle).
- **TVD** is the total variation distance of district population shares from uniform,
  `0.5 * sum_d |p_d - 1/D|`: the share of the state's population sitting in the wrong
  district. It is linear in the deviation, so its gradient never vanishes.
- **lambda** is the weight on population equality relative to compactness. At 0 only
  shape matters; as it grows, equal population buys more and more shape.

`rmsd` and `entropy` are also available as `mode=`. Normalised Shannon entropy is
reported but should not drive a search: `1 - H` is quadratically flat at uniformity,
so a 10% deviation still scores H = 0.99958. Everything routes through
`redistricting.objective.objective`, so the cached, incremental and rebuilt paths cannot
drift apart.

## Initialisations ("restarts")

All are `fn(g, rng) -> assignment` and return D non-empty **contiguous** districts.

| name | idea | contiguous by construction | needs lat/lon |
|---|---|---|---|
| `random` | grow D regions from random seeds, one random unit at a time | yes | no |
| `pop_greedy` | same, always extending the least-populous district | yes | no |
| `sweep` | fill one district at a time along a random direction | no, repaired | yes |
| `kmeans` | population-weighted k-means on centroids | no, repaired | yes |
| `spectral` | recursive bisection on the border-length-weighted Laplacian | no, repaired | no |

`repair()` keeps each district's largest component, reabsorbs orphans into the neighbour
sharing the most border, and refills any emptied district from a non-articulation donor.

## Moves

- **Flip** (`hill_climb`, `hill_climb_first`): a flip is `(i, d)`, unit `i` joins
  district `d`. Valid only if `d` is new for `i`, `i` touches `d`, the old district keeps
  at least one unit, and it stays connected. `hill_climb` is steepest ascent (scan all
  flips, take the best); `hill_climb_first` shuffles and takes the first improving flip.
- **ReCom** (`recom_climb`): merge two adjacent districts, draw a spanning tree of the
  union, cut one tree edge by population. Both halves are subtrees, so contiguity needs
  no repair, and one move can shift a large block of units. A cut is accepted only if
  it raises `f`. The tree sampler is a parameter:

  | `tree=` | minimum it pursues | use for |
  |---|---|---|
  | `ust` | none; Wilson's algorithm, uniform over spanning trees | sampling (well-defined proposal distribution) |
  | `rmst` | sum of i.i.d. U(0,1) weights, a pure randomiser; not uniform | cheap default |
  | `compact` | `U(0,1) / shared_border_length`, so long borders are cheap edges | optimising (biases toward compact cuts) |

- **Hybrid** (`hybrid`): `compact` ReCom to move blocks, then flips to polish the
  ragged boundary, repeated until neither helps.

## Measured results

Alabama, `lambda = 6`, per-cut tolerance 2%, medians over random starts, seed 20261122
(`python -m experiments.compare_search`):

| tracts (1,437), 5 starts | mean PP | max pop. deviation | median f |
|---|---|---|---|
| flips only | 0.321 | 10.36% | 0.1394 |
| recom `ust` | 0.333 | 1.11% | 0.3014 |
| recom `rmst` | 0.310 | 1.84% | 0.2941 |
| recom `compact` | 0.370 | 1.07% | 0.3534 |
| recom `compact` + flips | 0.409 | 0.51% | 0.4028 |

| counties (67), 6 starts | mean PP | max pop. deviation | median f |
|---|---|---|---|
| flips only | 0.316 | 16.86% | 0.1667 |
| recom `ust` | 0.359 | 7.24% | 0.2801 |
| recom `rmst` | 0.352 | 4.89% | 0.2922 |
| recom `compact` | 0.381 | 9.55% | 0.2932 |
| recom `compact` + flips | 0.374 | 7.24% | 0.3131 |

A single flip moves about 0.5% of a district's population at tract scale against about
10% at county scale, which is why flips alone stall on tracts and ReCom does not.

## Data

Two CSVs per unit set `<name>` in `data/` (or `$REDISTRICTING_DATA`):

- `<name>_features.csv`: `geoid, pop, area_land_km2, area_water_km2, perimeter_km`,
  optionally `name, lat, lon` (lat/lon only for `sweep` and `kmeans`).
- `<name>_borders.csv`: `a, b, shared_km`, one row per adjacent pair.

`data/` ships Alabama counties and tracts (`AL_county`, `AL_tract`), reduced to those
columns. Regenerate borders from a TIGER/Line shapefile (adjacency and shared length by
hashing identical boundary segments, no geometry library):

```
python -m redistricting.borders --shp TIGER/tl_2026_01_tract/tl_2026_01_tract \
    --state 01 --out data/AL_tract_borders.csv
```

**Index discipline.** `Geo` sorts units by GEOID, and every array, neighbour list and
district label in the package follows that order. Anything drawn from another source,
such as a shapefile in file order, must be re-keyed by GEOID, never by position.

## Tests

```
pip install -e ".[dev]"
python -m pytest                     # 57 tests; 2 more run when REDISTRICTING_TIGER is set
python -m redistricting.audit --unit AL_tract
```

- `test_audit.py` runs the invariant audit: merge/split perimeter identities, incremental
  score updates against rebuilt plans, `valid()` against brute-force apply-and-test,
  union-find against BFS contiguity, every ReCom tree and cut, and a random walk of
  flips. With `REDISTRICTING_TIGER` set it also rebuilds boundaries from raw TIGER
  segments and checks them against the shipped CSVs (state outer boundary 1,789.06 km
  for Alabama at both county and tract level).
- `test_regression.py` pins final assignments at fixed seeds to values recorded from the
  original scratch implementation, so a refactor that changes behaviour fails loudly.

## Not in this repo yet

- Feature-matrix builders for the ACS/CVAP/industry columns (this package reads only
  population, area, perimeter and, optionally, centroids).
- The adapter for the `Flip_ReCom_LocalSearch` data (IL, MO, TN) and the page builders
  used to visualise results.
- The lambda sweep and the 1,000-run score-distribution analysis, which so far exist
  only as ad hoc scripts.
