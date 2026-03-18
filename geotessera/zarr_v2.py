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
    PROJ_CONVENTION,
    SPATIAL_CONVENTION,
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

    # Create root group
    os.makedirs(str(output_path), exist_ok=True)
    root_meta = output_path / "zarr.json"
    with open(str(root_meta), "w") as f:
        _json.dump({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "zarr_conventions": [TESSERA_CONVENTION],
                "tessera:dataset_version": "v2",
                "tessera:years": years,
                "tessera:model_version": model_version,
                "tessera:build_version": geotessera_version,
            },
        }, f, indent=2)

    root = zarr.open_group(str(output_path), mode="r+", zarr_format=3,
                           use_consolidated=False)

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
    zone_dir = store_path / zone_group

    os.makedirs(str(zone_dir), exist_ok=True)
    zone_meta = zone_dir / "zarr.json"
    with open(str(zone_meta), "w") as f:
        _json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)

    root_reopen = zarr.open_group(str(store_path), mode="r+", zarr_format=3,
                                  use_consolidated=False)
    store = root_reopen[zone_group]

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

    # Spatial metadata
    x_min = grid.origin_x
    x_max = grid.origin_x + W * grid.pixel_size
    y_max = grid.origin_y
    y_min = grid.origin_y - H * grid.pixel_size

    store.attrs.update({
        "zarr_conventions": [
            TESSERA_CONVENTION, PROJ_CONVENTION, SPATIAL_CONVENTION,
        ],
        # proj:
        "proj:code": f"EPSG:{grid.canonical_epsg}",
        # spatial:
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": [
            grid.pixel_size, 0.0, grid.origin_x,
            0.0, -grid.pixel_size, grid.origin_y,
        ],
        "spatial:shape": [H, W],
        "spatial:bbox": [x_min, y_min, x_max, y_max],
        "spatial:registration": "pixel",
        # tessera:
        "tessera:dataset_version": "v2",
        "tessera:years": grid.years,
        "tessera:utm_zone": grid.zone,
        "tessera:n_bands": N_BANDS,
        "tessera:model_version": model_version,
        "tessera:build_version": build_version,
    })

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
    import zarr
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from shapely.geometry import Point

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
                    f"    [green]{written_count} shards written[/green]"
                )

            # Update tile registry
            _record_written_tiles(store_path, remaining, fill_year, zone_num)

    # Re-consolidate metadata after filling
    if total_shards_written > 0:
        import warnings
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
