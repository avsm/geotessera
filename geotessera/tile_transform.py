"""
Computed piecewise affine CoordinateTransform for tessera zarr stores.

Each pixel's exact UTM coordinate is computed from the 0.1° WGS84 tile
grid — the same geometry used to generate the tiles.  No per-tile metadata
needs to be stored in the zarr; the transform is fully determined by the
zone's grid origin and the 0.1° cell structure.

For a given zarr pixel index:
  1. Approximate its UTM coordinate from the regular grid
  2. Convert to WGS84 to determine which 0.1° cell it belongs to
  3. Project that cell's bounds back to UTM (reproducing the tile's affine)
  4. Compute the exact coordinate from the tile's affine
"""

from __future__ import annotations

from typing import Any, Hashable

import math
import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.transform import Affine, from_origin


class TesseraTileTransform(xr.indexes.CoordinateTransform):
    """Computed piecewise affine: zarr (row, col) → exact UTM (xc, yc).

    Reproduces the tile generation logic (project 0.1° cell bounds to UTM)
    to compute exact per-pixel coordinates without stored metadata.
    """

    def __init__(
        self,
        dim_size: dict[str, int],
        grid_affine: Affine,
        epsg: int,
    ):
        super().__init__(
            coord_names=("xc", "yc"),
            dim_size=dim_size,
            dtype=np.float64,
        )
        self.grid_affine = grid_affine
        self.epsg = epsg
        self._to_wgs = Transformer.from_crs(
            f"EPSG:{epsg}",
            "EPSG:4326",
            always_xy=True,
        )
        self._to_utm = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True,
        )
        # Cache: (cell_lon, cell_lat) → tile_affine
        self._cell_cache: dict[tuple[float, float], Affine] = {}

    def _cell_centre(self, lon: float, lat: float) -> tuple[float, float]:
        """Snap a WGS84 point to the nearest 0.1° cell centre.

        Tile grid_-8.35_54.75 has centre (-8.35, 54.75) and covers ±0.05°.
        Centres are at ..., -8.35, -8.25, -8.15, -8.05, 0.05, 0.15, ...
        """
        cell_lon = round(math.floor(lon / 0.1) * 0.1 + 0.05, 2)
        cell_lat = round(math.floor(lat / 0.1) * 0.1 + 0.05, 2)
        return cell_lon, cell_lat

    def _tile_affine(self, cell_lon: float, cell_lat: float) -> Affine:
        """Compute the tile affine for a 0.1° cell, matching tile generation."""
        key = (cell_lon, cell_lat)
        if key in self._cell_cache:
            return self._cell_cache[key]

        west = cell_lon - 0.05
        east = cell_lon + 0.05
        south = cell_lat - 0.05
        north = cell_lat + 0.05

        xs, ys = self._to_utm.transform(
            [west, east, west, east],
            [north, north, south, south],
        )
        xmin = min(xs)
        ymax = max(ys)

        affine = from_origin(xmin, ymax, 10.0, 10.0)
        self._cell_cache[key] = affine
        return affine

    def _pixel_coord(self, row: int, col: int) -> tuple[float, float]:
        """Exact UTM coordinate for zarr pixel (row, col)."""
        # Approximate coordinate from regular grid
        approx_e, approx_n = self.grid_affine * (col + 0.5, row + 0.5)

        # Determine which 0.1° cell this falls in
        approx_lon, approx_lat = self._to_wgs.transform(approx_e, approx_n)
        cell_lon, cell_lat = self._cell_centre(approx_lon, approx_lat)

        # Get that cell's tile affine
        tile_aff = self._tile_affine(cell_lon, cell_lat)

        # Compute the zarr pixel's position in the tile's grid
        # The tile was placed at zarr offset = round((tile_origin - grid_origin) / px)
        tile_col0 = round((tile_aff.c - self.grid_affine.c) / 10.0)
        tile_row0 = round((self.grid_affine.f - tile_aff.f) / 10.0)

        local_col = col - tile_col0
        local_row = row - tile_row0

        # Exact coordinate from the tile's affine (pixel centre)
        exact_e, exact_n = tile_aff * (local_col + 0.5, local_row + 0.5)
        return exact_e, exact_n

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        rows = np.asarray(dim_positions["y"], dtype=np.float64)
        cols = np.asarray(dim_positions["x"], dtype=np.float64)

        # Default: regular grid
        eastings, northings = self.grid_affine * (cols + 0.5, rows + 0.5)
        eastings = np.asarray(eastings, dtype=np.float64)
        northings = np.asarray(northings, dtype=np.float64)

        flat_r = rows.ravel()
        flat_c = cols.ravel()
        flat_e = eastings.ravel()
        flat_n = northings.ravel()

        for i in range(len(flat_r)):
            e, n = self._pixel_coord(int(flat_r[i]), int(flat_c[i]))
            flat_e[i] = e
            flat_n[i] = n

        return {
            "xc": flat_e.reshape(eastings.shape),
            "yc": flat_n.reshape(northings.shape),
        }

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        """UTM → grid index, checking ±1 neighbourhood for exact match."""
        eastings = np.asarray(coord_labels["xc"], dtype=np.float64)
        northings = np.asarray(coord_labels["yc"], dtype=np.float64)

        cols_approx, rows_approx = ~self.grid_affine * (eastings, northings)
        cols_approx = np.asarray(cols_approx) - 0.5
        rows_approx = np.asarray(rows_approx) - 0.5

        flat_e = eastings.ravel()
        flat_n = northings.ravel()
        flat_r = rows_approx.ravel().copy()
        flat_c = cols_approx.ravel().copy()

        h_max = self.dim_size["y"] - 1
        w_max = self.dim_size["x"] - 1

        for i in range(len(flat_e)):
            r0 = int(round(flat_r[i]))
            c0 = int(round(flat_c[i]))
            best_dist = float("inf")
            best_r, best_c = float(r0), float(c0)

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = r0 + dr, c0 + dc
                    if r < 0 or r > h_max or c < 0 or c > w_max:
                        continue
                    px_e, px_n = self._pixel_coord(r, c)
                    dist = (px_e - flat_e[i]) ** 2 + (px_n - flat_n[i]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_r, best_c = float(r), float(c)

            flat_r[i] = best_r
            flat_c[i] = best_c

        return {
            "y": flat_r.reshape(rows_approx.shape),
            "x": flat_c.reshape(cols_approx.shape),
        }

    def equals(self, other: xr.indexes.CoordinateTransform, **kwargs) -> bool:
        if not isinstance(other, TesseraTileTransform):
            return False
        return (
            self.grid_affine == other.grid_affine
            and self.epsg == other.epsg
            and self.dim_size == other.dim_size
        )

    @classmethod
    def from_zone_attrs(cls, attrs: dict) -> TesseraTileTransform:
        """Construct from zarr zone group attributes."""
        t = attrs["spatial:transform"]
        shape = attrs["spatial:shape"]
        epsg = int(attrs["proj:code"].split(":")[1])

        grid_affine = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        return cls(
            dim_size={"y": shape[0], "x": shape[1]},
            grid_affine=grid_affine,
            epsg=epsg,
        )
