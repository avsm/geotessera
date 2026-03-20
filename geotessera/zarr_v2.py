"""Tessera v2 Zarr store: single store with time dimension.

Layout:
    tessera.zarr/
        zarr.json                     # root: tessera:dataset_version="v2"
        utm{zone:02d}/                # one group per UTM zone
            embeddings                # int8    (T, B, H, W)
            scales                    # float32 (T, H, W)
            rgb                       # uint8   (T, 4, H, W)  [optional]
            time                      # int32   (T,)
            x                         # float64 (W,)
            y                         # float64 (H,)
            band                      # int32   (B,)
        _registry.parquet             # tile ingestion tracking

Dimension order: (time, band, y, x) — ML-standard NCHW.
Inner chunks: (1, 128, 32, 32), Shards: (1, 128, 4096, 4096).

Scale sentinels:
    NaN   = water (permanent, from landmask)
    +inf  = land, no data yet (set at init, replaced by real scale on fill)
    finite = valid data
"""

from __future__ import annotations

import json as _json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .zarr_zone import (
    N_BANDS,
    TESSERA_CONVENTION,
    ShardSpec,
    ShardTileOverlap,
    TileInfo,
    _load_landmask_slice,
    _zone_group_name,
    epsg_to_utm_zone,
    gather_tile_infos,
    northing_to_canonical,
    tile_pixel_offset,
    zone_canonical_epsg,
)

if TYPE_CHECKING:
    import rich.console
    import zarr

    from .registry import Registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v2 constants
# ---------------------------------------------------------------------------

V2_SHARD_SIZE = 4096   # spatial pixels per shard side
V2_INNER_CHUNK = 32    # spatial pixels per inner chunk side
V2_DEFAULT_WORKERS = 4  # fewer workers due to larger shard buffers (~2GB each)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UnifiedZoneGrid:
    """Describes the pixel grid for a UTM zone spanning all years."""

    zone: int
    years: List[int]
    canonical_epsg: int
    origin_x: float       # UTM easting of top-left corner
    origin_y: float       # UTM northing of top-left corner
    width_px: int
    height_px: int
    pixel_size: float = 10.0


def compute_unified_zone_grid(
    all_tile_infos: List[TileInfo],
    years: List[int],
) -> UnifiedZoneGrid:
    """Compute a zone grid that covers all tiles across all years.

    The bounding box is the union of all tile extents, snapped outward
    to V2_SHARD_SIZE boundaries.
    """
    if not all_tile_infos:
        raise ValueError("No tiles provided")

    zone = epsg_to_utm_zone(all_tile_infos[0].epsg)
    pixel_size = 10.0

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for ti in all_tile_infos:
        ti_zone = epsg_to_utm_zone(ti.epsg)
        if ti_zone != zone:
            raise ValueError(
                f"Mixed zones: expected {zone}, got {ti_zone} "
                f"for tile ({ti.lon}, {ti.lat})"
            )

        tile_x = ti.transform.c
        tile_y = northing_to_canonical(ti.transform.f, ti.epsg)
        tile_right = tile_x + ti.width * pixel_size
        tile_bottom = tile_y - ti.height * pixel_size

        min_x = min(min_x, tile_x)
        max_x = max(max_x, tile_right)
        min_y = min(min_y, tile_bottom)
        max_y = max(max_y, tile_y)

    # Snap to pixel grid
    origin_x = math.floor(min_x / pixel_size) * pixel_size
    origin_y = math.ceil(max_y / pixel_size) * pixel_size
    extent_right = math.ceil(max_x / pixel_size) * pixel_size
    extent_bottom = math.floor(min_y / pixel_size) * pixel_size

    width_px = round((extent_right - origin_x) / pixel_size)
    height_px = round((origin_y - extent_bottom) / pixel_size)

    # Snap to shard boundary
    width_px = math.ceil(width_px / V2_SHARD_SIZE) * V2_SHARD_SIZE
    height_px = math.ceil(height_px / V2_SHARD_SIZE) * V2_SHARD_SIZE

    return UnifiedZoneGrid(
        zone=zone,
        years=years,
        canonical_epsg=zone_canonical_epsg(zone),
        origin_x=origin_x,
        origin_y=origin_y,
        width_px=width_px,
        height_px=height_px,
        pixel_size=pixel_size,
    )


def _v2_tile_pixel_offset(
    tile_info: TileInfo, grid: UnifiedZoneGrid,
) -> Tuple[int, int]:
    """Pixel offset of a tile within the unified zone grid."""
    tile_x = tile_info.transform.c
    tile_y = northing_to_canonical(tile_info.transform.f, tile_info.epsg)
    col = round((tile_x - grid.origin_x) / grid.pixel_size)
    row = round((grid.origin_y - tile_y) / grid.pixel_size)
    return row, col


# ---------------------------------------------------------------------------
# Shard index (same logic as v1, different shard size)
# ---------------------------------------------------------------------------

def build_v2_shard_index(
    tile_infos: List[TileInfo],
    grid: UnifiedZoneGrid,
    time_index: int,
) -> List[ShardSpec]:
    """Build shard index for one year's tiles against a unified zone grid."""
    shard_map: Dict[Tuple[int, int], List[ShardTileOverlap]] = {}

    for ti in tile_infos:
        row, col = _v2_tile_pixel_offset(ti, grid)
        h, w = ti.height, ti.width

        sr_start = row // V2_SHARD_SIZE
        sr_end = (row + h - 1) // V2_SHARD_SIZE
        sc_start = col // V2_SHARD_SIZE
        sc_end = (col + w - 1) // V2_SHARD_SIZE

        for sr in range(sr_start, sr_end + 1):
            for sc in range(sc_start, sc_end + 1):
                shard_top = sr * V2_SHARD_SIZE
                shard_left = sc * V2_SHARD_SIZE

                t_row_start = max(0, shard_top - row)
                t_row_end = min(h, shard_top + V2_SHARD_SIZE - row)
                t_col_start = max(0, shard_left - col)
                t_col_end = min(w, shard_left + V2_SHARD_SIZE - col)

                s_row_start = max(0, row - shard_top)
                s_row_end = s_row_start + (t_row_end - t_row_start)
                s_col_start = max(0, col - shard_left)
                s_col_end = s_col_start + (t_col_end - t_col_start)

                ov = ShardTileOverlap(
                    embedding_path=ti.embedding_path,
                    scales_path=ti.scales_path,
                    landmask_path=ti.landmask_path,
                    t_row_start=t_row_start,
                    t_row_end=t_row_end,
                    t_col_start=t_col_start,
                    t_col_end=t_col_end,
                    s_row_start=s_row_start,
                    s_row_end=s_row_end,
                    s_col_start=s_col_start,
                    s_col_end=s_col_end,
                )
                shard_map.setdefault((sr, sc), []).append(ov)

    specs = []
    for (sr, sc), overlaps in sorted(shard_map.items()):
        specs.append(ShardSpec(
            sr=sr, sc=sc,
            row_px=sr * V2_SHARD_SIZE,
            col_px=sc * V2_SHARD_SIZE,
            time_index=time_index,
            tiles=overlaps,
        ))
    return specs


# ---------------------------------------------------------------------------
# Store initialisation (zarr-init)
# ---------------------------------------------------------------------------

def _default_zone_grid(zone: int, years: List[int]) -> UnifiedZoneGrid:
    """Compute a default grid for a UTM zone with no tiles.

    Uses the full theoretical UTM zone extent: 6 degrees wide,
    equator to 84N (the standard UTM northern limit).  This is
    generous but sparse — no storage is consumed until data is written.
    """
    from pyproj import Transformer

    epsg = zone_canonical_epsg(zone)
    west_lon = (zone - 1) * 6 - 180
    east_lon = zone * 6 - 180
    pixel_size = 10.0

    # Project zone corners to UTM
    proj = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    # Use 0N-84N for northern hemisphere canonical grid
    ul_x, ul_y = proj.transform(west_lon, 84.0)
    ur_x, ur_y = proj.transform(east_lon, 84.0)
    ll_x, ll_y = proj.transform(west_lon, 0.0)
    lr_x, lr_y = proj.transform(east_lon, 0.0)

    min_x = min(ul_x, ll_x)
    max_x = max(ur_x, lr_x)
    min_y = min(ll_y, lr_y)
    max_y = max(ul_y, ur_y)

    origin_x = math.floor(min_x / pixel_size) * pixel_size
    origin_y = math.ceil(max_y / pixel_size) * pixel_size
    extent_right = math.ceil(max_x / pixel_size) * pixel_size
    extent_bottom = math.floor(min_y / pixel_size) * pixel_size

    width_px = round((extent_right - origin_x) / pixel_size)
    height_px = round((origin_y - extent_bottom) / pixel_size)

    width_px = math.ceil(width_px / V2_SHARD_SIZE) * V2_SHARD_SIZE
    height_px = math.ceil(height_px / V2_SHARD_SIZE) * V2_SHARD_SIZE

    return UnifiedZoneGrid(
        zone=zone,
        years=years,
        canonical_epsg=epsg,
        origin_x=origin_x,
        origin_y=origin_y,
        width_px=width_px,
        height_px=height_px,
        pixel_size=pixel_size,
    )


def _gather_landmask_tiles_by_zone(
    registry: "Registry",
) -> Dict[int, List[Tuple[float, float]]]:
    """Group landmask tile coordinates by UTM zone.

    Returns dict mapping zone number to list of (lon, lat) centres.
    """
    tiles = registry.available_landmasks  # [(lon, lat), ...]
    by_zone: Dict[int, List[Tuple[float, float]]] = {}
    for lon, lat in tiles:
        zone_num = int(math.floor((lon + 180) / 6)) + 1
        zone_num = max(1, min(60, zone_num))
        by_zone.setdefault(zone_num, []).append((lon, lat))
    return by_zone


def _compute_zone_grid_from_landmask(
    zone: int,
    tile_coords: List[Tuple[float, float]],
    years: List[int],
) -> UnifiedZoneGrid:
    """Compute a unified zone grid from landmask tile coordinates.

    Projects tile bounding boxes to UTM and computes the union extent,
    snapped to V2_SHARD_SIZE boundaries.
    """
    from pyproj import Transformer

    epsg = zone_canonical_epsg(zone)
    pixel_size = 10.0
    proj = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for lon, lat in tile_coords:
        # Each tile is 0.1 degrees
        west, east = lon - 0.05, lon + 0.05
        south, north = lat - 0.05, lat + 0.05

        corners_x, corners_y = proj.transform(
            [west, east, west, east],
            [north, north, south, south],
        )
        min_x = min(min_x, min(corners_x))
        max_x = max(max_x, max(corners_x))
        min_y = min(min_y, min(corners_y))
        max_y = max(max_y, max(corners_y))

    origin_x = math.floor(min_x / pixel_size) * pixel_size
    origin_y = math.ceil(max_y / pixel_size) * pixel_size
    extent_right = math.ceil(max_x / pixel_size) * pixel_size
    extent_bottom = math.floor(min_y / pixel_size) * pixel_size

    width_px = round((extent_right - origin_x) / pixel_size)
    height_px = round((origin_y - extent_bottom) / pixel_size)

    width_px = math.ceil(width_px / V2_SHARD_SIZE) * V2_SHARD_SIZE
    height_px = math.ceil(height_px / V2_SHARD_SIZE) * V2_SHARD_SIZE

    return UnifiedZoneGrid(
        zone=zone,
        years=years,
        canonical_epsg=epsg,
        origin_x=origin_x,
        origin_y=origin_y,
        width_px=width_px,
        height_px=height_px,
        pixel_size=pixel_size,
    )


def init_v2_store(
    registry: "Registry",
    output_path: Path,
    years: List[int],
    geotessera_version: str = "unknown",
    model_version: str = "1.0",
    console: Optional["rich.console.Console"] = None,
) -> Path:
    """Create a v2 tessera store with time dimension from the landmask registry.

    Creates all UTM zones that have landmask coverage.  For each zone, the
    grid extent is computed from landmask tiles (not embeddings), so only
    the landmask registry is needed.

    The scales array is initialised with sentinels:
    - NaN  = water (permanent, from landmask)
    - +inf = land, no data yet (replaced by real scale values during fill)

    No embedding data is written.  The embeddings array stays at fill_value (0).
    """
    import zarr

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Store already exists: {output_path}")

    years = sorted(years)
    T = len(years)

    if console:
        console.print(f"Initialising v2 store at [bold]{output_path}[/bold]")
        console.print(f"  Years: {years[0]}-{years[-1]} ({T} time steps)")

    # Get landmask coverage grouped by UTM zone
    landmask_by_zone = _gather_landmask_tiles_by_zone(registry)

    if not landmask_by_zone:
        raise ValueError("No landmask tiles found in registry")

    if console:
        console.print(f"  {len(landmask_by_zone)} zone(s) with land coverage")

    # Create root group via zarr API (not manual JSON) so consolidation
    # preserves attributes correctly.
    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    root.attrs.update({
        "zarr_conventions": [TESSERA_CONVENTION],
        "tessera:dataset_version": "v2",
        "tessera:years": years,
        "tessera:model_version": model_version,
        "tessera:build_version": geotessera_version,
    })

    # Create each zone group from landmask coverage
    for zone_num in sorted(landmask_by_zone.keys()):
        tile_coords = landmask_by_zone[zone_num]
        grid = _compute_zone_grid_from_landmask(zone_num, tile_coords, years)

        if console:
            w_km = grid.width_px * grid.pixel_size / 1000
            h_km = grid.height_px * grid.pixel_size / 1000
            n_shards_x = grid.width_px // V2_SHARD_SIZE
            n_shards_y = grid.height_px // V2_SHARD_SIZE
            console.print(
                f"  Zone {zone_num} "
                f"[dim]EPSG:{grid.canonical_epsg}[/dim] "
                f"[dim]{grid.width_px}x{grid.height_px}px "
                f"({w_km:.0f}x{h_km:.0f}km) "
                f"{n_shards_x}x{n_shards_y} shards[/dim]"
            )

        _create_v2_zone_group(root, grid, output_path, model_version,
                              geotessera_version)

    # Create empty tile registry
    _init_tile_registry(output_path)

    # Consolidate metadata so HTTP readers can discover the hierarchy
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        zarr.consolidate_metadata(str(output_path))

    if console:
        console.print(f"  [green]Store initialised (metadata only, no data written)[/green]")

    return output_path


def _create_v2_zone_group(
    root: "zarr.Group",
    grid: UnifiedZoneGrid,
    store_path: Path,
    model_version: str,
    build_version: str,
) -> "zarr.Group":
    """Create a v2 zone group with empty (T, B, H, W) arrays."""
    import zarr
    from zarr.codecs import BloscCodec

    zone_group = _zone_group_name(grid.zone)

    root_reopen = zarr.open_group(str(store_path), mode="r+", zarr_format=3,
                                  use_consolidated=False)
    store = root_reopen.create_group(zone_group)

    T = len(grid.years)
    H = grid.height_px
    W = grid.width_px

    # Main data arrays — (T, B, H, W) layout
    store.create_array(
        "embeddings",
        shape=(T, N_BANDS, H, W),
        chunks=(1, N_BANDS, V2_INNER_CHUNK, V2_INNER_CHUNK),
        shards=(1, N_BANDS, V2_SHARD_SIZE, V2_SHARD_SIZE),
        dtype=np.int8, fill_value=np.int8(0),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["time", "band", "y", "x"],
    )
    # fill_value=+inf means unwritten land pixels read as "no data yet".
    # Water pixels are written as NaN during zarr-fill (from landmask).
    # Clients: isinf(scales) → land/no-data, isnan(scales) → water,
    #          isfinite(scales) → valid embedding data.
    store.create_array(
        "scales",
        shape=(T, H, W),
        chunks=(1, V2_INNER_CHUNK, V2_INNER_CHUNK),
        shards=(1, V2_SHARD_SIZE, V2_SHARD_SIZE),
        dtype=np.float32, fill_value=np.float32("inf"),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["time", "y", "x"],
    )

    # Coordinate arrays
    x_coords = (
        grid.origin_x + (np.arange(W) + 0.5) * grid.pixel_size
    )
    y_coords = (
        grid.origin_y - (np.arange(H) + 0.5) * grid.pixel_size
    )
    time_coords = np.array(grid.years, dtype=np.int32)
    band_coords = np.arange(N_BANDS, dtype=np.int32)

    for name, data, dim in [
        ("x", x_coords, "x"),
        ("y", y_coords, "y"),
        ("time", time_coords, "time"),
        ("band", band_coords, "band"),
    ]:
        store.create_array(
            name, shape=data.shape, dtype=data.dtype,
            fill_value=0, compressors=None, dimension_names=[dim],
        )
        store[name][:] = data

    # Use geozarr-toolkit for proj: and spatial: convention metadata
    from geozarr_toolkit import create_geozarr_attrs

    x_min = grid.origin_x
    x_max = grid.origin_x + W * grid.pixel_size
    y_max = grid.origin_y
    y_min = grid.origin_y - H * grid.pixel_size

    geozarr_attrs = create_geozarr_attrs(
        dimensions=["y", "x"],
        crs=f"EPSG:{grid.canonical_epsg}",
        transform=[
            grid.pixel_size, 0.0, grid.origin_x,
            0.0, -grid.pixel_size, grid.origin_y,
        ],
        bbox=[x_min, y_min, x_max, y_max],
        shape=[H, W],
        registration="pixel",
    )

    # Fix convention descriptions to match upstream schemas exactly
    # (geozarr-toolkit has a bug: "Spatial coordinate and transformation
    # information" instead of "Spatial coordinate information")
    for conv in geozarr_attrs.get("zarr_conventions", []):
        if conv.get("uuid") == "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4":
            conv["description"] = "Spatial coordinate information"

    # Add tessera convention registration alongside the geozarr ones
    geozarr_attrs.setdefault("zarr_conventions", []).insert(0, TESSERA_CONVENTION)

    # Add tessera-specific attributes
    geozarr_attrs.update({
        "tessera:dataset_version": "v2",
        "tessera:years": grid.years,
        "tessera:utm_zone": grid.zone,
        "tessera:n_bands": N_BANDS,
        "tessera:model_version": model_version,
        "tessera:build_version": build_version,
    })

    store.attrs.update(geozarr_attrs)

    return store


# ---------------------------------------------------------------------------
# Tile registry (GeoParquet tracking which tiles are written)
# ---------------------------------------------------------------------------

def _registry_path(store_path: Path) -> Path:
    return store_path / "_registry.parquet"


def _init_tile_registry(store_path: Path) -> None:
    """Create an empty tile registry parquet file."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    schema = gpd.GeoDataFrame({
        "year": pd.array([], dtype="int32"),
        "zone": pd.array([], dtype="int32"),
        "shard_row": pd.array([], dtype="int32"),
        "shard_col": pd.array([], dtype="int32"),
        "tile_lon": pd.array([], dtype="float64"),
        "tile_lat": pd.array([], dtype="float64"),
        "written_at": pd.array([], dtype="datetime64[ns, UTC]"),
        "geometry": gpd.array.GeometryArray(
            gpd.points_from_xy([], []),
        ),
    })
    schema.to_parquet(str(_registry_path(store_path)))


def _load_tile_registry(store_path: Path) -> "gpd.GeoDataFrame":
    """Load the tile registry, or create it if missing."""
    import geopandas as gpd

    path = _registry_path(store_path)
    if path.exists():
        return gpd.read_parquet(str(path))
    _init_tile_registry(store_path)
    return gpd.read_parquet(str(path))


def _save_tile_registry(store_path: Path, gdf: "gpd.GeoDataFrame") -> None:
    """Save the tile registry."""
    gdf.to_parquet(str(_registry_path(store_path)))


def _get_written_tiles(store_path: Path, year: int, zone: int) -> set:
    """Return set of (tile_lon, tile_lat) already written for a year/zone."""
    gdf = _load_tile_registry(store_path)
    if gdf.empty:
        return set()
    mask = (gdf["year"] == year) & (gdf["zone"] == zone)
    subset = gdf[mask]
    return set(zip(subset["tile_lon"], subset["tile_lat"]))


# ---------------------------------------------------------------------------
# Shard writing (NCHW layout)
# ---------------------------------------------------------------------------

_v2_worker_store = None


def _init_v2_shard_worker(store_path: str, zone_group: str) -> None:
    """Process pool initializer: open the zone group once per worker."""
    global _v2_worker_store
    import zarr
    _v2_worker_store = zarr.open_group(
        store_path, mode="r+", path=zone_group,
        zarr_format=3, use_consolidated=False,
    )


def _write_one_shard_v2(spec: ShardSpec, store: "zarr.Group") -> bool:
    """Write one shard in NCHW layout: (T, B, H, W)."""
    t = spec.time_index
    S = V2_SHARD_SIZE

    # Allocate BHW buffer (bands-first for NCHW write)
    emb_buf = np.zeros((N_BANDS, S, S), dtype=np.int8)
    # Start with +inf (land/nodata); landmask sets water to NaN,
    # valid tiles overwrite with finite scales.
    scales_buf = np.full((S, S), np.float32("inf"))

    has_data = False
    for ov in spec.tiles:
        # Read HWB tile, transpose to BHW
        emb = np.load(ov.embedding_path, mmap_mode="r")
        tile_slice = emb[
            ov.t_row_start:ov.t_row_end,
            ov.t_col_start:ov.t_col_end,
            :,
        ]
        emb_buf[
            :,
            ov.s_row_start:ov.s_row_end,
            ov.s_col_start:ov.s_col_end,
        ] = tile_slice.transpose(2, 0, 1)

        # Scales
        scales_mmap = np.load(ov.scales_path, mmap_mode="r")
        s = scales_mmap[
            ov.t_row_start:ov.t_row_end,
            ov.t_col_start:ov.t_col_end,
        ].copy()

        # Landmask
        lm = _load_landmask_slice(
            ov.landmask_path,
            ov.t_row_start, ov.t_row_end,
            ov.t_col_start, ov.t_col_end,
        )
        s[lm == 0] = np.float32("nan")
        s[~np.isfinite(s)] = np.float32("nan")

        scales_buf[ov.s_row_start:ov.s_row_end, ov.s_col_start:ov.s_col_end] = s
        has_data = True

    if not has_data:
        return False

    r, c = spec.row_px, spec.col_px
    store["embeddings"][t, :, r:r + S, c:c + S] = emb_buf
    store["scales"][t, r:r + S, c:c + S] = scales_buf
    return True


def _write_one_shard_v2_worker(spec: ShardSpec) -> bool:
    """Picklable wrapper for process pool."""
    return _write_one_shard_v2(spec, _v2_worker_store)


# ---------------------------------------------------------------------------
# Fill orchestration (zarr-fill)
# ---------------------------------------------------------------------------

def fill_v2_store(
    registry: "Registry",
    store_path: Path,
    year: Optional[int] = None,
    zones: Optional[List[int]] = None,
    with_rgb: bool = False,
    console: Optional["rich.console.Console"] = None,
    workers: Optional[int] = None,
) -> int:
    """Incrementally fill a v2 store with tile data.

    Reads the tile registry to skip already-written tiles.
    Returns the number of shards written.
    """
    import warnings
    import zarr
    from concurrent.futures import ProcessPoolExecutor, as_completed

    store_path = Path(store_path)
    if workers is None:
        workers = V2_DEFAULT_WORKERS

    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)
    root_attrs = dict(root.attrs)
    all_years = root_attrs.get("tessera:years", [])

    if not all_years:
        raise ValueError("Store has no tessera:years attribute")

    fill_years = [year] if year is not None else all_years

    if console:
        console.print(f"Filling v2 store at [bold]{store_path}[/bold]")
        console.print(f"  Years to fill: {fill_years}")

    total_shards_written = 0
    zones_filled: List[Tuple[int, int]] = []  # (zone_num, time_index) for RGB

    for fill_year in fill_years:
        if fill_year not in all_years:
            if console:
                console.print(f"  [yellow]Year {fill_year} not in store, skipping[/yellow]")
            continue

        time_index = all_years.index(fill_year)

        # Gather tiles for this year
        year_tiles = gather_tile_infos(
            registry, fill_year, zones=zones, console=console,
        )

        for zone_num, tile_infos in sorted(year_tiles.items()):
            zone_group = _zone_group_name(zone_num)
            zone_path = store_path / zone_group

            if not zone_path.exists():
                if console:
                    console.print(
                        f"  [yellow]Zone {zone_num} not initialised, skipping[/yellow]"
                    )
                continue

            # Check which tiles are already written
            written = _get_written_tiles(store_path, fill_year, zone_num)
            remaining = [
                ti for ti in tile_infos
                if (ti.lon, ti.lat) not in written
            ]

            if not remaining:
                if console:
                    console.print(
                        f"  Zone {zone_num} year {fill_year}: "
                        f"all {len(tile_infos)} tiles already written"
                    )
                continue

            if console:
                console.print(
                    f"  Zone {zone_num} year {fill_year}: "
                    f"{len(remaining)}/{len(tile_infos)} tiles to write"
                )

            # Read the zone grid from store metadata
            zone_store = zarr.open_group(
                str(store_path), mode="r", path=zone_group,
                use_consolidated=False,
            )
            zone_attrs = dict(zone_store.attrs)
            transform = zone_attrs["spatial:transform"]
            shape = zone_attrs["spatial:shape"]

            grid = UnifiedZoneGrid(
                zone=zone_num,
                years=all_years,
                canonical_epsg=int(zone_attrs["proj:code"].split(":")[1]),
                origin_x=transform[2],
                origin_y=transform[5],
                width_px=shape[1],
                height_px=shape[0],
            )

            # Build shard index
            shard_specs = build_v2_shard_index(remaining, grid, time_index)

            if console:
                console.print(
                    f"    {len(shard_specs)} shards to write "
                    f"({workers} workers)"
                )

            # Write shards via process pool
            zone_store_path = str(store_path)
            written_count = 0
            n_shards = len(shard_specs)

            if console:
                from rich.progress import (
                    Progress, BarColumn, TextColumn,
                    MofNCompleteColumn, TimeElapsedColumn,
                    TimeRemainingColumn, SpinnerColumn,
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"    Zone {zone_num} y{fill_year}",
                        total=n_shards,
                    )
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_init_v2_shard_worker,
                        initargs=(zone_store_path, zone_group),
                    ) as pool:
                        futures = {
                            pool.submit(_write_one_shard_v2_worker, spec): spec
                            for spec in shard_specs
                        }
                        for future in as_completed(futures):
                            try:
                                if future.result():
                                    written_count += 1
                            except Exception as e:
                                spec = futures[future]
                                logger.warning(
                                    f"Shard ({spec.sr},{spec.sc}) failed: {e}"
                                )
                            progress.advance(task)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_v2_shard_worker,
                    initargs=(zone_store_path, zone_group),
                ) as pool:
                    futures = {
                        pool.submit(_write_one_shard_v2_worker, spec): spec
                        for spec in shard_specs
                    }
                    for future in as_completed(futures):
                        try:
                            if future.result():
                                written_count += 1
                        except Exception as e:
                            spec = futures[future]
                            logger.warning(
                                f"Shard ({spec.sr},{spec.sc}) failed: {e}"
                            )

            total_shards_written += written_count

            if console:
                console.print(
                    f"    [green]{written_count}/{n_shards} shards written[/green]"
                )

            # Update tile registry
            _record_written_tiles(store_path, remaining, fill_year, zone_num)

            if written_count > 0:
                zones_filled.append((zone_num, time_index))

    # Generate RGB previews for filled zones
    if with_rgb and zones_filled:
        if console:
            console.print(f"\n  Generating RGB previews...")
        for zone_num, time_index in zones_filled:
            add_v2_rgb_preview(
                store_path, zone_num, time_index,
                workers=workers, console=console,
            )

    # Re-consolidate metadata after filling
    if total_shards_written > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata")
            zarr.consolidate_metadata(str(store_path))

    return total_shards_written


def _record_written_tiles(
    store_path: Path,
    tile_infos: List[TileInfo],
    year: int,
    zone: int,
) -> None:
    """Append newly written tiles to the registry."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for ti in tile_infos:
        rows.append({
            "year": np.int32(year),
            "zone": np.int32(zone),
            "shard_row": np.int32(0),  # TODO: compute from tile offset
            "shard_col": np.int32(0),
            "tile_lon": ti.lon,
            "tile_lat": ti.lat,
            "written_at": now,
            "geometry": Point(ti.lon, ti.lat),
        })

    new_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    existing = _load_tile_registry(store_path)

    combined = gpd.GeoDataFrame(
        pd.concat([existing, new_gdf], ignore_index=True),
        crs="EPSG:4326",
    )
    _save_tile_registry(store_path, combined)


# ---------------------------------------------------------------------------
# RGB preview generation (v2 — NCHW layout)
# ---------------------------------------------------------------------------

RGB_PREVIEW_BANDS = (0, 1, 2)


def _ensure_v2_rgb_array(store: "zarr.Group") -> None:
    """Create the rgb array in a v2 zone group if it doesn't exist.

    Shape: (T, 4, H, W) matching the NCHW layout.
    """
    from zarr.codecs import BloscCodec

    try:
        _ = store["rgb"]
        return
    except KeyError:
        pass

    emb_shape = store["embeddings"].shape  # (T, B, H, W)
    T, _, H, W = emb_shape
    store.create_array(
        "rgb",
        shape=(T, 4, H, W),
        chunks=(1, 4, V2_INNER_CHUNK, V2_INNER_CHUNK),
        shards=(1, 4, V2_SHARD_SIZE, V2_SHARD_SIZE),
        dtype=np.uint8, fill_value=np.uint8(0),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["time", "rgba", "y", "x"],
    )


def _compute_rgb_chunk_v2(
    emb_bhw: np.ndarray,
    scales_hw: np.ndarray,
    band_indices: tuple,
    stretch_min: List[float],
    stretch_max: List[float],
) -> np.ndarray:
    """Compute RGBA preview from NCHW-layout embedding + scales.

    Args:
        emb_bhw: int8 (B, H, W)
        scales_hw: float32 (H, W)

    Returns:
        uint8 (4, H, W) — RGBA in channels-first layout.
    """
    h, w = scales_hw.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    valid = np.isfinite(scales_hw)
    scales_safe = np.where(valid, scales_hw, 0.0)

    for i, band_idx in enumerate(band_indices):
        raw = emb_bhw[band_idx].astype(np.float32)
        dequant = raw * scales_safe
        lo, hi = stretch_min[i], stretch_max[i]
        normalised = np.clip((dequant - lo) / max(hi - lo, 1e-10), 0.0, 1.0)
        rgba[i] = (normalised * 255).astype(np.uint8)

    rgba[:3, ~valid] = 0
    rgba[3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _sample_v2_chunk_stats(
    emb_arr, scales_arr,
    time_index: int,
    ci: int, cj: int,
    shard_size: int,
    spatial_shape: Tuple[int, int],
    band_indices: tuple = RGB_PREVIEW_BANDS,
    max_per_chunk: int = 10_000,
) -> Optional[np.ndarray]:
    """Sample dequantised values from one shard for stretch estimation.

    Reads from NCHW layout: emb_arr[t, bands, r0:r1, c0:c1].
    """
    H, W = spatial_shape
    r0, r1 = ci * shard_size, min(ci * shard_size + shard_size, H)
    c0, c1 = cj * shard_size, min(cj * shard_size + shard_size, W)

    scales_chunk = np.asarray(scales_arr[time_index, r0:r1, c0:c1])
    valid = np.isfinite(scales_chunk)
    if not np.any(valid):
        return None

    # Read only the RGB bands
    band_list = list(band_indices)
    emb_chunk = np.asarray(
        emb_arr[time_index, band_list[0]:band_list[-1] + 1, r0:r1, c0:c1]
    )  # (n_rgb_bands, h, w)

    # Dequantise only valid pixels (avoid inf/nan multiply warnings)
    scales_safe = np.where(valid, scales_chunk, 0.0)
    vals_all = emb_chunk.astype(np.float32) * scales_safe[np.newaxis, :, :]
    # Reshape to (n_pixels, n_bands) and keep only valid
    n_bands = vals_all.shape[0]
    vals_flat = vals_all.reshape(n_bands, -1).T  # (n_pixels, n_bands)
    valid_flat = valid.ravel()
    vals = vals_flat[valid_flat]

    if vals.shape[0] > max_per_chunk:
        rng = np.random.default_rng(ci * 10007 + cj)
        idx = rng.choice(vals.shape[0], max_per_chunk, replace=False)
        vals = vals[idx]

    return vals


def compute_v2_stretch(
    store: "zarr.Group",
    time_index: int,
    p_low: float = 2,
    p_high: float = 98,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
    sample_fraction: float = 0.1,
) -> dict:
    """Compute percentile stretch for RGB bands at one time step."""
    from .zarr_zone import _run_parallel

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    _, _, H, W = emb_arr.shape

    n_rows = math.ceil(H / V2_SHARD_SIZE)
    n_cols = math.ceil(W / V2_SHARD_SIZE)
    all_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]

    n_sample = max(1, int(len(all_indices) * sample_fraction))
    rng = np.random.default_rng(42)
    sample_indices = [
        all_indices[i] for i in rng.choice(len(all_indices), n_sample, replace=False)
    ]

    results = _run_parallel(
        lambda idx: _sample_v2_chunk_stats(
            emb_arr, scales_arr, time_index,
            idx[0], idx[1], V2_SHARD_SIZE, (H, W),
        ),
        sample_indices, workers, console,
        label=f"Sampling stretch ({n_sample}/{len(all_indices)} shards)",
    )

    samples = [r for _, r in results if r is not None]
    if not samples:
        return {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}

    all_rgb = np.concatenate(samples, axis=0)
    stretch_min = [float(np.percentile(all_rgb[:, i], p_low)) for i in range(3)]
    stretch_max = [float(np.percentile(all_rgb[:, i], p_high)) for i in range(3)]

    for i in range(3):
        if stretch_max[i] <= stretch_min[i]:
            stretch_max[i] = stretch_min[i] + 1.0

    return {"min": stretch_min, "max": stretch_max}


def _write_v2_rgb_pass(
    store: "zarr.Group",
    time_index: int,
    stretch: dict,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Write the RGB preview for one time step, shard by shard."""
    from .zarr_zone import _run_parallel

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    rgb_arr = store["rgb"]
    _, _, H, W = emb_arr.shape

    n_rows = math.ceil(H / V2_SHARD_SIZE)
    n_cols = math.ceil(W / V2_SHARD_SIZE)

    def _process_shard(idx):
        ci, cj = idx
        r0 = ci * V2_SHARD_SIZE
        r1 = min(r0 + V2_SHARD_SIZE, H)
        c0 = cj * V2_SHARD_SIZE
        c1 = min(c0 + V2_SHARD_SIZE, W)

        scales_chunk = np.asarray(scales_arr[time_index, r0:r1, c0:c1])
        if not np.any(np.isfinite(scales_chunk)):
            return False

        emb_chunk = np.asarray(emb_arr[time_index, :, r0:r1, c0:c1])  # (B, h, w)
        rgba = _compute_rgb_chunk_v2(
            emb_chunk, scales_chunk,
            RGB_PREVIEW_BANDS, stretch["min"], stretch["max"],
        )
        rgb_arr[time_index, :, r0:r1, c0:c1] = rgba
        return True

    shard_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]

    results = _run_parallel(
        _process_shard, shard_indices, workers, console,
        label=f"Writing RGB preview ({workers} threads)",
    )

    return sum(1 for _, wrote in results if wrote)


def add_v2_rgb_preview(
    store_path: Path,
    zone_num: int,
    time_index: int,
    workers: Optional[int] = None,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add or update the RGB preview for one zone at one time step."""
    import zarr
    from .zarr_zone import _default_workers

    if workers is None:
        workers = _default_workers()

    zone_group = _zone_group_name(zone_num)
    store = zarr.open_group(
        str(store_path), mode="r+", path=zone_group,
        zarr_format=3, use_consolidated=False,
    )

    _ensure_v2_rgb_array(store)

    if console:
        console.print(f"  Zone {zone_num} t={time_index}: sampling stretch...")

    stretch = compute_v2_stretch(
        store, time_index, workers=workers, console=console,
    )

    if console:
        console.print(
            f"    Stretch: min={[f'{v:.3f}' for v in stretch['min']]}, "
            f"max={[f'{v:.3f}' for v in stretch['max']]}"
        )

    written = _write_v2_rgb_pass(
        store, time_index, stretch, workers=workers, console=console,
    )

    store.attrs.update({"tessera:has_rgb_preview": True})

    if console:
        console.print(f"    [green]RGB: {written} shards written[/green]")


# ---------------------------------------------------------------------------
# Global RGB preview pyramid (v2)
# ---------------------------------------------------------------------------
# Reprojects per-zone UTM RGB (NCHW) into a single EPSG:4326 pyramid.
# The global pyramid stays HWC (H, W, 4) since it's purely for web tiles.
# Parallelisation follows v1: ProcessPoolExecutor for reprojection,
# ThreadPoolExecutor for pyramid coarsening.

from .zarr_zone import (
    GLOBAL_BOUNDS, GLOBAL_BASE_RES, GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W,
    GLOBAL_CHUNK, GLOBAL_NUM_BANDS, GLOBAL_DEFAULT_LEVELS,
    SPATIAL_CONVENTION, MULTISCALES_CONVENTION, PROJ_CONVENTION,
    _zone_output_bounds, _coarsen_zone_pyramid,
)


def _ensure_v2_global_store(store_path: Path, num_levels: int) -> None:
    """Create the global_rgb/ pyramid group within a v2 store."""
    import zarr
    from zarr.codecs import BloscCodec

    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3,
                           use_consolidated=False)

    # Check if already exists with correct shape
    if "global_rgb/0/rgb" in root:
        shape = root["global_rgb/0/rgb"].shape
        if shape == (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS):
            return
        import shutil
        shutil.rmtree(str(store_path / "global_rgb"))
        root = zarr.open_group(str(store_path), mode="r+", zarr_format=3,
                               use_consolidated=False)

    # Create pyramid levels via zarr API
    global_grp = root.create_group("global_rgb")
    h, w = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    band_data = np.arange(GLOBAL_NUM_BANDS, dtype=np.int32)

    for lvl in range(num_levels):
        if h < 1 or w < 1:
            break
        lvl_grp = global_grp.create_group(str(lvl))
        lvl_grp.create_array(
            "rgb",
            shape=(h, w, GLOBAL_NUM_BANDS),
            chunks=(GLOBAL_CHUNK, GLOBAL_CHUNK, GLOBAL_NUM_BANDS),
            dtype=np.uint8, fill_value=np.uint8(0),
            compressors=BloscCodec(cname="zstd", clevel=3),
            dimension_names=["lat", "lon", "band"],
        )
        lvl_grp.create_array(
            "band", data=band_data,
            chunks=(GLOBAL_NUM_BANDS,),
            dimension_names=["band"],
        )
        h //= 2
        w //= 2

    # Multiscale metadata
    from topozarr.metadata import create_multiscale_metadata
    actual_levels = len([k for k in global_grp.keys() if k.isdigit()])
    ms_attrs = create_multiscale_metadata(actual_levels, "EPSG:4326", "mean")

    # Fix convention descriptions (same geozarr-toolkit bug workaround)
    if "zarr_conventions" in ms_attrs:
        for conv in ms_attrs["zarr_conventions"]:
            if conv.get("uuid") == "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4":
                conv["description"] = "Spatial coordinate information"
        ms_attrs["zarr_conventions"].append(SPATIAL_CONVENTION)
    else:
        ms_attrs["zarr_conventions"] = [
            MULTISCALES_CONVENTION, PROJ_CONVENTION, SPATIAL_CONVENTION,
        ]

    west, south, east, north_ = GLOBAL_BOUNDS
    ms_attrs["spatial:dimensions"] = ["lat", "lon"]
    ms_attrs["spatial:bbox"] = [west, south, east, north_]

    h_lvl, w_lvl = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    res = GLOBAL_BASE_RES
    for item in ms_attrs.get("multiscales", {}).get("layout", []):
        item["spatial:shape"] = [h_lvl, w_lvl]
        item["spatial:transform"] = [res, 0.0, west, 0.0, -res, north_]
        h_lvl //= 2
        w_lvl //= 2
        res *= 2.0

    global_grp.attrs.update(ms_attrs)


# Per-worker state for v2 reprojection
_v2_reproj_global_arr = None
_v2_reproj_src_arr = None
_v2_reproj_to_utm = None
_v2_reproj_time_index = None


def _init_v2_reproj_worker(
    store_path: str, zone_group: str, zone_epsg: int, time_index: int,
) -> None:
    """Process pool initializer: open stores and create transformer."""
    global _v2_reproj_global_arr, _v2_reproj_src_arr
    global _v2_reproj_to_utm, _v2_reproj_time_index
    import zarr
    from pyproj import Transformer

    root = zarr.open_group(store_path, mode="r+", zarr_format=3,
                           use_consolidated=False)
    _v2_reproj_global_arr = root["global_rgb/0/rgb"]
    _v2_reproj_src_arr = root[zone_group + "/rgb"]  # (T, 4, H, W)
    _v2_reproj_to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{zone_epsg}", always_xy=True,
    )
    _v2_reproj_time_index = time_index


def _v2_reproject_chunk_worker(args) -> bool:
    """Process pool worker for v2 reprojection."""
    chunk_row, chunk_col, src_epsg, src_pixel, src_origin_e, src_origin_n, src_h, src_w = args
    return _v2_reproject_chunk(
        _v2_reproj_global_arr, chunk_row, chunk_col,
        _v2_reproj_src_arr, _v2_reproj_time_index,
        src_epsg, src_pixel, src_origin_e, src_origin_n, src_h, src_w,
        _v2_reproj_to_utm,
    )


def _v2_reproject_chunk(
    global_arr,
    chunk_row: int, chunk_col: int,
    src_arr, time_index: int,
    src_epsg: int, src_pixel: float,
    src_origin_e: float, src_origin_n: float,
    src_h: int, src_w: int,
    to_utm,
) -> bool:
    """Reproject one 512x512 global chunk from a v2 NCHW source."""
    import warnings
    from affine import Affine
    from rasterio.enums import Resampling
    import rasterio.warp
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    west, south, east, north_ = GLOBAL_BOUNDS
    row0 = chunk_row * GLOBAL_CHUNK
    col0 = chunk_col * GLOBAL_CHUNK
    tile_h = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_H - row0)
    tile_w = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_W - col0)
    if tile_h <= 0 or tile_w <= 0:
        return False

    tile_west = west + col0 * GLOBAL_BASE_RES
    tile_north = north_ - row0 * GLOBAL_BASE_RES
    tile_east = tile_west + tile_w * GLOBAL_BASE_RES
    tile_south = tile_north - tile_h * GLOBAL_BASE_RES

    dst_transform = Affine(
        GLOBAL_BASE_RES, 0, tile_west,
        0, -GLOBAL_BASE_RES, tile_north,
    )

    # Sample corners to check zone coverage
    sample_lons = [tile_west, tile_east, tile_west, tile_east,
                   (tile_west + tile_east) / 2]
    sample_lats = [tile_north, tile_north, tile_south, tile_south,
                   (tile_north + tile_south) / 2]
    try:
        utm_xs, utm_ys = to_utm.transform(sample_lons, sample_lats)
    except Exception:
        return False

    if any(not math.isfinite(v) for v in list(utm_xs) + list(utm_ys)):
        return False

    # Compute source window in UTM pixel coords
    pad = 16
    r_min = max(0, int((src_origin_n - max(utm_ys)) / src_pixel) - pad)
    r_max = min(src_h, int(math.ceil(
        (src_origin_n - min(utm_ys)) / src_pixel
    )) + pad)
    c_min = max(0, int((min(utm_xs) - src_origin_e) / src_pixel) - pad)
    c_max = min(src_w, int(math.ceil(
        (max(utm_xs) - src_origin_e) / src_pixel
    )) + pad)

    if r_max <= r_min or c_max <= c_min:
        return False

    # Read from NCHW source: src_arr[t, :4, r_min:r_max, c_min:c_max]
    # Result is (4, h, w) — already band-first for rasterio
    window = np.asarray(src_arr[time_index, :, r_min:r_max, c_min:c_max])
    if not window.any():
        return False

    src_data = window.astype(np.float32)  # (4, h, w) already band-first
    del window

    win_transform = Affine(
        src_pixel, 0, src_origin_e + c_min * src_pixel,
        0, -src_pixel, src_origin_n - r_min * src_pixel,
    )

    # Mask invalid pixels (alpha < 128) as NaN
    alpha_band = src_data[3]
    invalid = alpha_band < 128
    for b in range(3):
        src_data[b][invalid] = np.nan
    rgb_src = src_data[:3]
    del src_data

    rgb_dst = np.full((3, tile_h, tile_w), np.nan, dtype=np.float32)

    try:
        rasterio.warp.reproject(
            source=rgb_src,
            destination=rgb_dst,
            src_transform=win_transform,
            src_crs=f"EPSG:{src_epsg}",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    except Exception:
        return False

    del rgb_src

    # Derive alpha from valid reprojected RGB
    has_data = np.any(np.isfinite(rgb_dst) & (rgb_dst != 0), axis=0)
    rgb_dst = np.nan_to_num(rgb_dst, nan=0.0)
    rgb_dst = np.clip(rgb_dst, 0, 255).astype(np.uint8)
    rgb_out = np.transpose(rgb_dst, (1, 2, 0))  # (h, w, 3)
    del rgb_dst

    out = np.zeros((tile_h, tile_w, GLOBAL_NUM_BANDS), dtype=np.uint8)
    out[:, :, :3] = rgb_out
    out[:, :, 3] = np.where(has_data, 255, 0).astype(np.uint8)
    del rgb_out

    if not out.any():
        return False

    # Composite: only overwrite pixels where new zone has data
    mask = out.any(axis=2)
    if mask.all():
        global_arr[row0:row0 + tile_h, col0:col0 + tile_w, :] = out
    else:
        existing = np.asarray(
            global_arr[row0:row0 + tile_h, col0:col0 + tile_w, :]
        )
        existing[mask] = out[mask]
        global_arr[row0:row0 + tile_h, col0:col0 + tile_w, :] = existing
    return True


def _reproject_v2_zone(
    store_path: Path,
    zone_num: int,
    zone_group: str,
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
    time_index: int,
    workers: int,
    console: Optional["rich.console.Console"] = None,
    force: bool = False,
) -> Tuple[int, int, int, int, bool]:
    """Reproject one zone's RGB into global level 0 (v2 NCHW source)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    src_pixel = zone_transform[0]
    src_origin_e = zone_transform[2]
    src_origin_n = zone_transform[5]
    src_h, src_w = zone_shape[:2]

    row_start, row_end, col_start, col_end = _zone_output_bounds(
        zone_epsg=zone_epsg,
        zone_transform=zone_transform,
        zone_shape=(src_h, src_w),
    )

    if col_end <= col_start or row_end <= row_start:
        if console:
            console.print(f"    [yellow]Zone {zone_num}: no output region[/yellow]")
        return (0, 0, 0, 0, False)

    n_chunk_rows = (row_end - row_start) // GLOBAL_CHUNK
    n_chunk_cols = (col_end - col_start) // GLOBAL_CHUNK
    chunk_row_start = row_start // GLOBAL_CHUNK
    chunk_col_start = col_start // GLOBAL_CHUNK

    # Resume check
    marker = store_path / f".zone_{zone_num}_done"
    if marker.exists():
        if force:
            marker.unlink()
        else:
            if console:
                console.print(
                    f"    Zone {zone_num:02d}: already complete, skipping"
                )
            return (row_start, row_end, col_start, col_end, False)

    chunks_total = n_chunk_rows * n_chunk_cols
    if console:
        console.print(
            f"    Zone {zone_num:02d}: {n_chunk_rows}x{n_chunk_cols} "
            f"= {chunks_total} chunks"
        )

    work_items = [
        (chunk_row_start + cr, chunk_col_start + cc,
         zone_epsg, src_pixel, src_origin_e, src_origin_n, src_h, src_w)
        for cr in range(n_chunk_rows)
        for cc in range(n_chunk_cols)
    ]

    chunks_written = 0

    if console:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task(
                f"Reprojecting zone {zone_num:02d}", total=len(work_items),
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_v2_reproj_worker,
                initargs=(str(store_path), zone_group, zone_epsg, time_index),
            ) as pool:
                futures = {
                    pool.submit(_v2_reproject_chunk_worker, item): item
                    for item in work_items
                }
                for future in as_completed(futures):
                    try:
                        if future.result():
                            chunks_written += 1
                    except Exception as e:
                        logger.warning(f"Reproject chunk failed: {e}")
                    progress.advance(ptask)
        console.print(f"    {chunks_written} chunks with data")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_v2_reproj_worker,
            initargs=(str(store_path), zone_group, zone_epsg, time_index),
        ) as pool:
            futures = {
                pool.submit(_v2_reproject_chunk_worker, item): item
                for item in work_items
            }
            for future in as_completed(futures):
                try:
                    if future.result():
                        chunks_written += 1
                except Exception:
                    pass

    marker.write_text(
        f"zone={zone_num} chunks={chunks_total} written={chunks_written}\n"
    )

    return (row_start, row_end, col_start, col_end, True)


def build_v2_global_preview(
    store_path: Path,
    time_index: int,
    zones: Optional[List[int]] = None,
    num_levels: int = GLOBAL_DEFAULT_LEVELS,
    workers: int = 4,
    console: Optional["rich.console.Console"] = None,
    force: bool = False,
) -> None:
    """Build the global EPSG:4326 RGB pyramid from v2 zone-level RGB.

    For each zone that has RGB data at the given time_index, reprojects
    from UTM to geographic coordinates and composites into the pyramid.
    Uses ProcessPoolExecutor for reprojection (CPU-bound) and
    ThreadPoolExecutor for pyramid coarsening (I/O-bound).
    """
    import re
    import warnings
    import zarr
    import gc

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)
    root_attrs = dict(root.attrs)
    all_years = root_attrs.get("tessera:years", [])

    if time_index >= len(all_years):
        raise ValueError(f"time_index {time_index} out of range (store has {len(all_years)} years)")

    year = all_years[time_index]

    if console:
        console.print(
            f"Building global preview for year {year} (t={time_index})"
        )

    # Discover zones with RGB data
    zone_pattern = re.compile(r"^utm(\d{2})$")
    zone_infos: Dict[int, dict] = {}

    for name in sorted(root.keys()):
        m = zone_pattern.match(name)
        if not m:
            continue
        zone_num = int(m.group(1))
        if zones is not None and zone_num not in zones:
            continue

        zone_store = root[name]
        attrs = dict(zone_store.attrs)

        if not attrs.get("tessera:has_rgb_preview", False):
            continue

        # Check the rgb array exists and has data for this time index
        try:
            rgb_arr = zone_store["rgb"]
            # Shape is (T, 4, H, W)
            _, _, zone_h, zone_w = rgb_arr.shape
        except (KeyError, ValueError):
            continue

        zone_infos[zone_num] = {
            "zone_group": name,
            "epsg": int(attrs["proj:code"].split(":")[1]),
            "transform": list(attrs["spatial:transform"]),
            "shape": (zone_h, zone_w),
        }

    if not zone_infos:
        if console:
            console.print("[yellow]No zones with RGB data found[/yellow]")
        return

    if console:
        console.print(f"  {len(zone_infos)} zone(s) with RGB data")

    # Ensure global pyramid structure exists
    _ensure_v2_global_store(store_path, num_levels)

    # Reproject each zone and build pyramid
    for zone_num, info in sorted(zone_infos.items()):
        if console:
            console.print(f"\n  Zone {zone_num:02d}:")

        row_start, row_end, col_start, col_end, did_work = _reproject_v2_zone(
            store_path=store_path,
            zone_num=zone_num,
            zone_group=info["zone_group"],
            zone_epsg=info["epsg"],
            zone_transform=info["transform"],
            zone_shape=info["shape"],
            time_index=time_index,
            workers=workers,
            console=console,
            force=force,
        )

        if did_work:
            if console:
                console.print(f"    Building pyramid...")
            _coarsen_zone_pyramid(
                store_path=store_path,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                num_levels=num_levels,
                workers=workers,
                console=console,
            )

        gc.collect()

    # Consolidate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        zarr.consolidate_metadata(str(store_path))

    if console:
        console.print(f"\n  [green]Global preview complete[/green]")
