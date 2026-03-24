"""
Piecewise affine CoordinateTransform for tessera zarr stores.

Each tile in the zarr was written at a specific (row, col) offset with
its own affine transform.  This transform computes exact per-pixel UTM
coordinates by dispatching to the affine of whichever tile wrote each
pixel.

For ``reverse()`` (coordinate → pixel index), the regular grid
approximation is used — it's within ±0.5 pixel, which ``sel()`` rounds
to the correct integer index.

Tile placements are stored in the zone attrs as ``tessera:tile_transforms``
and recorded during ``zarr-fill``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np
import xarray as xr
from rasterio.transform import Affine


@dataclass(frozen=True, slots=True)
class TilePlacement:
    """One tile's position in the zarr grid + its original affine."""
    row0: int
    col0: int
    height: int
    width: int
    ox: float       # tile origin easting (top-left corner)
    oy: float       # tile origin northing (top-left corner)
    px: float        # pixel size


class TesseraTileTransform(xr.indexes.CoordinateTransform):
    """Piecewise affine: zarr (row, col) → exact UTM (easting, northing).

    Each pixel's coordinate comes from whichever tile wrote it.
    Pixels without a tile use the regular grid.
    """

    def __init__(
        self,
        dim_size: dict[str, int],
        tiles: list[TilePlacement],
        grid_affine: Affine,
    ):
        # coord_names are "xc"/"yc" (distinct from dim names "x"/"y")
        # following the rasterix convention
        super().__init__(
            coord_names=("xc", "yc"),
            dim_size=dim_size,
            dtype=np.float64,
        )
        self.tiles = sorted(tiles, key=lambda t: (t.row0, t.col0))
        self.grid_affine = grid_affine

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        """Grid (y=row, x=col) → UTM (xc=easting, yc=northing)."""
        rows = np.asarray(dim_positions["y"], dtype=np.float64)
        cols = np.asarray(dim_positions["x"], dtype=np.float64)

        # Regular grid coordinates as default (pixel centres)
        eastings, northings = self.grid_affine * (cols + 0.5, rows + 0.5)
        eastings = np.asarray(eastings, dtype=np.float64)
        northings = np.asarray(northings, dtype=np.float64)

        # Override with exact tile coordinates where applicable
        flat_r = rows.ravel()
        flat_c = cols.ravel()
        flat_e = eastings.ravel()
        flat_n = northings.ravel()

        for t in self.tiles:
            mask = (
                (flat_r >= t.row0) & (flat_r < t.row0 + t.height) &
                (flat_c >= t.col0) & (flat_c < t.col0 + t.width)
            )
            if mask.any():
                lr = flat_r[mask] - t.row0
                lc = flat_c[mask] - t.col0
                flat_e[mask] = t.ox + (lc + 0.5) * t.px
                flat_n[mask] = t.oy - (lr + 0.5) * t.px

        return {
            "xc": flat_e.reshape(eastings.shape),
            "yc": flat_n.reshape(northings.shape),
        }

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        """UTM (xc, yc) → approximate grid (y, x).

        Uses the regular grid inverse — within ±0.5 pixel, sufficient for
        ``sel(method='nearest')`` to round to the correct index.
        """
        eastings = np.asarray(coord_labels["xc"], dtype=np.float64)
        northings = np.asarray(coord_labels["yc"], dtype=np.float64)

        cols, rows = ~self.grid_affine * (eastings, northings)
        # Subtract 0.5 because affine maps pixel corners, we indexed with +0.5 in forward
        return {"y": np.asarray(rows) - 0.5, "x": np.asarray(cols) - 0.5}

    def equals(self, other: xr.indexes.CoordinateTransform, **kwargs) -> bool:
        if not isinstance(other, TesseraTileTransform):
            return False
        return (
            self.grid_affine == other.grid_affine
            and self.dim_size == other.dim_size
            and len(self.tiles) == len(other.tiles)
        )

    # -- Serialisation to/from zarr attrs ----------------------------------

    def to_attrs(self) -> list:
        """Serialise tile placements for storage in zarr zone attrs."""
        return [
            {
                "row0": t.row0, "col0": t.col0,
                "height": t.height, "width": t.width,
                "ox": t.ox, "oy": t.oy, "px": t.px,
            }
            for t in self.tiles
        ]

    @classmethod
    def from_zone_attrs(cls, attrs: dict) -> TesseraTileTransform | None:
        """Reconstruct from zarr zone group attributes, or None if absent."""
        tile_data = attrs.get("tessera:tile_transforms")
        if not tile_data:
            return None

        t = attrs["spatial:transform"]
        shape = attrs["spatial:shape"]

        tiles = [
            TilePlacement(
                row0=td["row0"], col0=td["col0"],
                height=td["height"], width=td["width"],
                ox=td["ox"], oy=td["oy"], px=td["px"],
            )
            for td in tile_data
        ]

        grid_affine = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        return cls(
            dim_size={"y": shape[0], "x": shape[1]},
            tiles=tiles,
            grid_affine=grid_affine,
        )
