"""The unit graph: populations, areas, perimeters and shared-border lengths.

Input contract, two CSVs per unit set <name> (e.g. AL_county, AL_tract):

  <name>_features.csv   geoid, pop, area_land_km2, area_water_km2, perimeter_km
                        optional: name, lat, lon   (lat/lon needed only by the
                        sweep and k-means initialisations)
  <name>_borders.csv    a, b, shared_km    one row per adjacent pair, from
                        redistricting.borders

Units are sorted by GEOID.  Every array, neighbour list and district label in this
package is indexed in that order; anything drawn from another source (a shapefile,
say) must be re-keyed by GEOID, never by position.
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from . import config


class Geo:
    def __init__(self, features_csv, borders_csv, D=7):
        C = pd.read_csv(features_csv, dtype={"geoid": str})
        C = C.sort_values("geoid").reset_index(drop=True)
        self.gid   = C.geoid.tolist()
        self.name  = C["name"].tolist() if "name" in C.columns else list(self.gid)
        self.idx   = {g: i for i, g in enumerate(self.gid)}
        self.N     = len(self.gid)
        self.D     = D
        self.pop   = C["pop"].to_numpy(float)
        self.area  = (C.area_land_km2 + C.area_water_km2).to_numpy(float)
        self.perim = C.perimeter_km.to_numpy(float)
        self.ideal = self.pop.sum() / D
        self._lat = C["lat"].to_numpy(float) if "lat" in C.columns else None
        self._lon = C["lon"].to_numpy(float) if "lon" in C.columns else None
        self._xy = None
        self.features = C                  # every column, GEOID-sorted (stats read counts here)

        B = pd.read_csv(borders_csv, dtype={"a": str, "b": str})
        self.nbr    = [[] for _ in range(self.N)]
        self.border = {}
        for a, b, L in B.itertuples(index=False):
            i, j = self.idx[a], self.idx[b]
            self.nbr[i].append(j); self.nbr[j].append(i)
            self.border[(min(i, j), max(i, j))] = L
        self.nbr = [tuple(v) for v in self.nbr]

    @classmethod
    def load(cls, name, D=7, data_dir=None):
        """Geo.load("AL_tract", D=7)  ->  <data_dir>/AL_tract_{features,borders}.csv"""
        d = Path(data_dir) if data_dir else config.data_dir()
        return cls(d / f"{name}_features.csv", d / f"{name}_borders.csv", D)

    def col(self, name):
        """A feature column as a float array in unit order (missing -> 0)."""
        return self.features[name].fillna(0).to_numpy(float)

    def subset(self, idx, D=None):
        """The sub-map induced by unit indices idx (re-indexed 0..len(idx)-1 in the
        given order).  Borders to units outside idx are dropped from `border`; the
        units' own perimeters are kept, so a district's perimeter is still its true
        outer boundary."""
        idx = list(idx); pos = {u: k for k, u in enumerate(idx)}
        h = Geo.__new__(Geo)
        h.gid = [self.gid[u] for u in idx]; h.name = [self.name[u] for u in idx]
        h.idx = {g: k for k, g in enumerate(h.gid)}; h.N = len(idx)
        h.D = D if D is not None else self.D
        h.pop = self.pop[idx]; h.area = self.area[idx]; h.perim = self.perim[idx]
        h.ideal = h.pop.sum() / h.D
        h._lat = None if self._lat is None else self._lat[idx]
        h._lon = None if self._lon is None else self._lon[idx]
        h._xy = None
        h.features = self.features.iloc[idx].reset_index(drop=True)
        h.border = {}
        for (a, b), L in self.border.items():
            if a in pos and b in pos:
                i, j = pos[a], pos[b]
                h.border[(min(i, j), max(i, j))] = L
        nb = [[] for _ in range(h.N)]
        for i, j in h.border:
            nb[i].append(j); nb[j].append(i)
        h.nbr = [tuple(v) for v in nb]
        return h

    def bl(self, i, j):
        """Shared border length (km) between units i and j; 0 if not adjacent."""
        return self.border.get((min(i, j), max(i, j)), 0.0)

    @property
    def xy(self):
        """Planar unit centroids in km (equirectangular about the mean latitude)."""
        if self._xy is None:
            if self._lat is None or self._lon is None:
                raise ValueError("this unit set has no lat/lon columns; the sweep "
                                 "and kmeans initialisations need them")
            R_EARTH = 6371.0
            lat, lon = self._lat, self._lon
            self._xy = np.column_stack(
                [np.radians(lon) * R_EARTH * math.cos(math.radians(lat.mean())),
                 np.radians(lat) * R_EARTH])
        return self._xy
