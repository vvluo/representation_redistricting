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

## Exact enumeration (ZDD)

`redistricting.enumerate` builds a zero-suppressed decision diagram (ZDD) of **every**
partition of a unit graph into exactly p connected districts, following Fifield, Imai,
Kawahara & Kenny (2020), "The Essential Role of Empirical Validation in Legislative
Redistricting Simulation", sec. 2.3 (frontier-based search after Kawahara et al. 2017).

```python
from redistricting import Geo
from redistricting.enumerate import PartitionZDD, population_ok
g = Geo.load("IA_county", D=4)
z = PartitionZDD(g.N, list(g.border), 4, order="best", xy=g.xy)
z.count                                   # exact number of contiguous 4-partitions
L = z.sample(100_000, rng)                # iid uniform plans, (M, N) labels
L = L[population_ok(L, g.pop, 4, 0.05)]   # still uniform, now on plans within 5%
```

- **State.** A ZDD node stores the component labels of the frontier vertices, the
  forbidden pairs (components joined by a cut edge, which may never merge) and the
  number of finished districts. As in the paper, population is **not** in the state:
  storing it stops nodes from merging. Population limits filter enumerated or sampled
  plans afterwards, and a uniform sample filtered this way is uniform on the plans
  that pass.
- **Ordering (paper sec. 2.3.4).** `order_cut` implements the recursive minimum s-t
  vertex cut. The paper finds s, t with Floyd-Warshall; on an unweighted graph one
  BFS per vertex gives the same distances, and both are provided. Neither is a
  bottleneck: Floyd-Warshall (scipy) takes 0.5 s on Alabama's 1,437 tracts. The
  limit is the frontier: node counts grow about 8x per extra frontier vertex. No single
  ordering wins everywhere, so `best_order` also tries BFS layering and coordinate
  sweeps in 36 directions, and keeps the one with the smallest `sum_l 8^|F_l|`.

  | max frontier | cut | BFS | best sweep |
  |---|---|---|---|
  | Iowa counties (99) | 12 | 11 | 10 |
  | Alabama counties (67) | 10 | 9 | 9 |
  | Iowa tracts (896) | 46 | 51 | - |
  | Alabama tracts (1,437) | 47 | 81 | - |

- **Validation** (`tests/test_zdd.py`). The enumerated partitions equal, as sets, those
  found by an independent brute-force enumerator on grids, triangulated grids, wheels,
  random Delaunay triangulations, trees and K5 (p = 1..4, both orderings); counts match
  closed forms on paths, cycles, trees and complete graphs and a third, subset-based
  count on the 4x4 grid (627). Every plan's sampling probability is exactly 1/count.
- **How far it goes** (pure Python, about 5-10 us per node, `experiments/zdd_scaling.py`).
  On random contiguous subsets of the Iowa tract map, grown as in the paper, with an
  8M-node / 180 s budget: every 30-70 tract subset finished (at most 5.8M nodes, 82 s,
  1.5 GB); at 85 and 100 tracts one of the two subsets finished and the other ran out
  of budget. Node count tracks the frontier, not the number of units: a 40-tract
  subset with frontier 9 took 2.6M nodes while a 70-tract one with frontier 7 took 85k.
  p = 4 costs about 2.2x p = 2.
  With the C++ backend and a 150M-node budget, on random pieces of 30-200 Iowa tracts:
  **every piece with max frontier <= 12 finished** (largest: 100 tracts, frontier 12,
  17.4M nodes, 6.7 s, 0.8 GB; a 120-tract piece with frontier 10 took 0.5 s), and
  **every piece with frontier >= 13 ran out of budget**, including a 60-tract one. Size
  in units is not the limit; frontier width is. 150- and 200-tract pieces had frontiers
  of 14-18 and none finished.
- **C++ backend.** `PartitionZDD(..., backend="cpp")` builds the same diagram with
  `redistricting/enumerate/cpp/zdd_build.cpp` (compiled on first use through ctypes;
  `python -m redistricting.enumerate._cpp` compiles it explicitly). It processes states
  in the same order and canonical form as the Python builder, so the node arrays are
  identical entry for entry, not just the counts: `tests/test_zdd_cpp.py` checks this
  on grids, triangulations, wheels, trees, cycles, K6 and Iowa tract subsets for
  p = 1..5. Ordering, counting and sampling are shared Python/numpy code. It is
  30-47x faster and uses about a third of the memory.
- **Iowa, 99 counties, 4 districts** (`experiments/zdd_sample.py`): the whole map
  builds in 706 s and 4.5 GB in Python, **20 s and 1.6 GB in C++** (55.7M nodes, max
  frontier 10 with a west-east sweep).
  Exactly 944,051,457,824,193,030,004,001 (about 10^24) contiguous 4-partitions, which
  matches the paper's "approximately 10^24". Of 5M uniform samples, 349 (7.0e-5) are
  within 5% population deviation and 2 (4e-7) within 1%; the paper reports 7.2e-5 and
  6e-7 from 500M samples on 2010 populations (ours are ACS).

## MCMC validated against the ZDD

`redistricting.mcmc.FlipChain` is single-flip Metropolis-Hastings with a **uniform**
target over valid plans (p contiguous districts, each within `tol` of ideal
population): propose a uniformly random boundary move (unit u, neighbouring district
d), reject it if it breaks a constraint, else accept with probability
`min(1, |B(x)| / |B(x')|)`, where B is the set of boundary moves. Tight tolerances can
cut the valid plans into pieces that single flips cannot cross, so the chain walks
within a looser `walk_tol` and keeps only the states within `tol`; that is still
exactly uniform on the plans within `tol`. On a 3x3 grid and a 10-node triangulation,
end states of 12,000 independent chains match the ZDD's enumeration (chi-square
p = 0.76 and 0.30).

**Does the MCMC distribution of each metric match the ZDD's?** Two-sample KS tests,
MCMC against ZDD, population limit 5%. `experiments/mcmc_benchmark.py` and
`experiments/mcmc_convergence.py`; statistics in `redistricting/stats.py`. Four chains
start from extreme plans (least / most compact, lowest / highest D_race); the first
20% of each is dropped. KS D uses every kept state. Because MCMC states are
autocorrelated, the p-value is the median over all thinning offsets at a stride of
3x the autocorrelation time.

| metric | 70 tracts, 2 districts | 70 tracts, 3 districts | Iowa 99 counties, 4 districts |
|---|---|---|---|
| reference | all 464,122 valid plans (exact) | 100k iid ZDD samples | 5,000 iid ZDD samples |
| MCMC | 4 x 4M steps, walk 10% | 4 x 12M steps, walk 10% | 4 x 50M steps, walk **20%** |
| population deviation | D 0.005, p 0.55 | D 0.004, p 0.54 | D 0.011, p 0.79 |
| compactness (mean Polsby-Popper) | D 0.006, p 0.44 | D 0.004, p 0.55 | D 0.025, p 0.14 |
| D_race (minority vs white CVAP) | D 0.005, p 0.53 | D 0.005, p 0.57 | D 0.017, p 0.18 |
| minority CVAP share, rank 1 | D 0.006, p 0.40 | D 0.004, p 0.67 | D 0.014, p 0.42 |
| minority CVAP share, rank 2 | D 0.004, p 0.56 | D 0.002, p 0.46 | D 0.013, p 0.49 |
| minority CVAP share, rank 3 / 4 | - | D 0.004, p 0.77 | D 0.007 / 0.012, p 0.33 / 0.78 |
| D_education (BA+ vs not) | D 0.003, p 0.59 | D 0.006, p 0.34 | D 0.027, p 0.31 |
| D_income stand-in (poverty) | D 0.003, p 0.44 | D 0.005, p 0.57 | D 0.015, p 0.22 |
| pairwise variation of information | D 0.014, p 0.74 | D 0.010, p 0.22 | D 0.027, p 0.05 |

The MCMC matches the ZDD on every metric in all three benchmarks, **but on Iowa only
with walk_tol = 20%**. Walking within 10%, the chain does not converge: KS D for
D_race stays at 0.10-0.13 and for minority-share ranks 2 and 4 at 0.12-0.15 from 5M
to 50M steps per chain, while p-values fall towards 0 as the same bias is measured
more precisely. At 20% the distances shrink to the reference's noise floor. The price
is that only about 5% of the walk's states are within 5%, so 4 x 50M steps
(about 1.5 CPU-hours per chain) yield 5,160 usable plans.

Caveats: D_income uses people below vs above the poverty line, because a dissimilarity
index needs income-group counts (ACS B19001) and the data has only tract medians.
Race shares are CVAP, not VAP. The Iowa reference has only 5,000 plans (71M ZDD draws),
which limits how small a difference the test can see. The paper's Republican
dissimilarity index needs vote returns, which this repo does not have.

## RL features and reward (`redistricting.rl`)

Single-tract flips `a = (i, t)` move tract `i` to a neighbouring district `t`.
`FlipFeatures` builds the action features `x(s, a)` and the objective changes behind
the reward, for every boundary flip at once (about 7 ms per step for Iowa's 896 tracts
at 50 districts):

```python
F = FlipFeatures(g, p=50, tol=0.05, alpha=0.25)
st = F.state(assignment)
acts, mask = F.actions(st)             # boundary flips; mask = contiguity + population
X = F.features(st, acts)               # (A, 124) for Iowa today; names in F.names
r = F.reward(X, {"poverty": 1, "ethnicity": 1, "industry": 1})
st.apply(*acts[k])
```

**Reward.** `r(s, a) = w . dphi(s, a)`, where `dphi` is the change in each objective
term: population (`-sum r_d^2`), compactness (mean Polsby-Popper), and one
representation term per protected family that has data. Families are defined in
`redistricting/rl/groups.py`:

| Family | Groups | Data |
|---|---|---|
| poverty | below the poverty line | in the tract master |
| ethnicity | Black, Hispanic, Asian, Native CVAP | in the tract master |
| industry | the 13 ACS sectors | in the tract master |
| rurality | micropolitan, small-town and rural tracts (RUCA 4–6, 7–9, 10) | in the tract master |
| education | BA+; less than high school | BA+ in the master; less than high school needs B15003 |
| income | households under $35k; $150k and over | B19001 |
| class_of_worker | government, self-employed, non-profit workers | C24080 |
| housing | renters, rent-burdened households, vacant units | B25003, B25070, B25002 |
| health | uninsured | B27010 |
| food | SNAP households | B22003 |
| language | Spanish, other Indo-European, Asian/Pacific Island, other languages spoken at home | B16007 |
| group_quarters | adult correctional, nursing-facility and college-housing residents | B26103 |

A family switches on as soon as its count columns appear in the unit features, with
no code change. The extra ACS tables come from `redistricting.acs` (variable codes
checked against the 2024 ACS 5-year metadata). The Census API needs a free key
(https://api.census.gov/data/key_signup.html), read from `CENSUS_API_KEY`:

```
python -m redistricting.acs --state 19 --out ../extracted/acs_api/IA_2024.csv
python -m redistricting.prep --state 19 --abbr IA --shp ../TIGER/tl_2026_19_tract/tl_2026_19_tract \
    --master ../extracted/tract_master.csv --acs ../extracted/acs_api/IA_2024.csv --out data
```

The same command takes `--state all`, or `--files` with data.census.gov ZIPs split any
way (per table, per state). Pass the national output to
`python -m redistricting.rl.groups --acs ...` to set these groups' thresholds from US
tracts. Until then, their `tau` uses the same
mean + 1 SD rule over the state's own tracts (`tau_source = "state_tracts"`). The family term is `-L`, where

`L = (1/Z) sum_g sum_{m<=N_g} alpha^(m-1) max(0, 1 - s_g^(m) / tau_g)`

- `s_g^(m)` is the `m`-th highest district share of group `g`.
- `tau_g` is the share at which a district represents `g`. By default it is the US
  tract mean + 1 SD, stored in `data/group_thresholds.csv`.
- `N_g` is the number of representing districts the map can supply, a greedy lower
  bound.

Losing a group's only attainable district costs 1; merging two into one costs `alpha`.
Groups the map cannot give a district (`N_g = 0`) carry no reward, and groups that
every map represents contribute almost nothing. Because the reward is linear in the
`dphi` block, a grid search over `w` only re-weights columns.

**Feature blocks:**

| Block | Contents |
|---|---|
| `dphi` | the changes above |
| `tract` | the tract's population, income, poverty, race, education and industry shares |
| `src`, `dst` | each district's deviation, compactness and composition, and its share of the tract's main group relative to `tau` |
| `geo` | neighbours and shared-border fractions per district, outer boundary, distance to each district's centroid |
| `community` | high-tract neighbours and high-tract cluster population in each district |
| `consequence` | whether the flip fragments or reunites the tract's cluster, population slack, articulation |

**Validation.** `tests/test_rl_features.py` checks every `dphi` against a from-scratch
recomputation, the mask against BFS contiguity, and the incremental state against a
rebuilt one.

**Iowa at 50 districts.** No ethnic group is attainable at the US thresholds (34–37%
CVAP), so the ethnicity term is zero there unless `tau` is lowered. Poverty has one
attainable district. A greedy policy on `w = {poverty: 1}` raised a start's best
poverty district from 24.1% to 26.0%, past `tau` = 25.4%, in 28 flips while staying
within 5% population.

### Environment (`redistricting.rl.env`)

```python
env = RedistrictingEnv("IA", "congress")        # any state; "congress" or "house"
obs, info = env.reset(seed=0)                   # obs = {"x": (K, F) features, "mask": (K,)}
obs, r, terminated, truncated, info = env.step(k)   # k indexes a legal flip
```

- **States and districts:** `redistricting.states` holds the congressional seats
  (2020 apportionment) and the number of state-house districts (not seats). The
  tolerance is ±1% for congress and ±5% for the house. Unit files are built on
  demand from TIGER and the tract master into `data/states/`, and island components
  are bridged.
- **Actions:** only the legal flips, each one a row of `x(s, a)`. Contiguity is a hard
  constraint, and no district may move outside the population tolerance, or further
  outside it.
- **Reward:** `w . dphi`, so an episode's return equals `J(final) - J(start)`.
- **Resets:** maps are drawn from a cached starter pool (`rl/pool.py`) mixing five
  initialisations, each balanced and then flip-repaired.
- **Exploration band:** `explore_tol` (default = `tol`) is the band the mask
  enforces on intermediate maps. Set it wider, e.g. `explore_tol=0.05` for congress,
  and maps may leave `tol` mid-episode while the `excess` term, `-sum_d max(0, |r_d| -
  tol)`, pulls them back. `tol` stays the requirement for a valid map, and the tracked
  best map is always within `tol`.
- **Episode end:** at `horizon` flips (truncated) or when no legal flip remains
  (terminated). `info` carries the best map seen.
- **Evaluation:** `env.evaluate(map)` reports the objective terms, constraint status
  and the cracking table. A group counts as cracked when `N_g >= 1` but no district
  reaches `tau_g`.

State houses on tracts: 19 states have fewer than 10 tracts per house district, and
there the flip repair cannot reach ±5%. Iowa's house plans stall at 59–155% deviation.
`strict_pool=False` keeps such starts for development; block groups are the fix.

## Data

Two CSVs per unit set `<name>` in `data/` (or `$REDISTRICTING_DATA`):

- `<name>_features.csv`: `geoid, pop, area_land_km2, area_water_km2, perimeter_km`,
  optionally `name, lat, lon` (lat/lon only for `sweep` and `kmeans`).
- `<name>_borders.csv`: `a, b, shared_km`, one row per adjacent pair.

`data/` ships Alabama counties and tracts (`AL_county`, `AL_tract`), reduced to those
columns, and Iowa counties and tracts (`IA_county`, `IA_tract`), which also carry the
count columns the plan statistics need (CVAP by race, BA+ of the 25+ population, people
below poverty, employed residents and workers in each of the 13 ACS industry sectors). Iowa counties are unions of tracts, built by `redistricting.prep`:

```
python -m redistricting.prep --state 19 --abbr IA \
    --shp ../TIGER/tl_2026_19_tract/tl_2026_19_tract \
    --master ../extracted/tract_master.csv --out data
``` Regenerate borders from a TIGER/Line shapefile (adjacency and shared length by
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
pip install -e ".[dev,rl]"
python -m pytest                     # 206 tests; 2 more run when REDISTRICTING_TIGER is set
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

- The ACS tables for income, class of worker, education below high school, housing,
  health insurance and SNAP (B19001, C24080, B15003, B25003/B25070/B25002, B27010,
  B22003, B16007, B26103). Their families switch on once prep carries the columns.
- A starter-map pool for RL resets. `balance` rarely reaches 5% at 50 Iowa districts;
  a greedy flip repair on `d_population` closes the gap.
- The adapter for the `Flip_ReCom_LocalSearch` data (IL, MO, TN) and the page builders
  used to visualise results.
- The lambda sweep and the 1,000-run score-distribution analysis, which so far exist
  only as ad hoc scripts.
