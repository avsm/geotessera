"""Tessera Zarr store: single store with time dimension.

Layout:
    tessera.zarr/
        zarr.json                     # root: geoemb: convention attrs
        utm{zone:02d}/                # one group per UTM zone
            embeddings                # int8    (T, B, H, W)
            scales                    # float32 (T, H, W)
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

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import geopandas
    import rich.console
    import zarr
    from rasterio.transform import Affine

    from .registry import Registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BANDS = 128

# Global preview grid (fixed extent, never changes)
GLOBAL_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
GLOBAL_BASE_RES = 0.0001  # degrees (~10m at equator)
GLOBAL_LEVEL0_W = 3_600_000  # ceil(360 / 0.0001)
GLOBAL_LEVEL0_H = 1_800_000  # ceil(180 / 0.0001)
GLOBAL_CHUNK = 512
GLOBAL_NUM_BANDS = 4
GLOBAL_DEFAULT_LEVELS = 10

# GeoZarr convention registration entries
GEOEMB_CONVENTION = {
    "uuid": "61c12cc5-0e28-4056-999a-480cf3fb7e4c",
    "name": "geoemb:",
    "description": "Geoembeddings convention for geospatial embedding arrays with model provenance",
    "spec_url": "https://github.com/geo-embeddings/embeddings-zarr-convention/blob/v1/README.md",
    "schema_url": "https://raw.githubusercontent.com/geo-embeddings/embeddings-zarr-convention/refs/tags/v1/schema.json",
}


# ---------------------------------------------------------------------------
# Data types (shared)
# ---------------------------------------------------------------------------


@dataclass
class TileInfo:
    """Metadata for a single tile to be placed in a zone store."""

    lon: float
    lat: float
    year: int
    epsg: int
    transform: Affine
    height: int
    width: int
    landmask_path: str
    embedding_path: str
    scales_path: str


@dataclass
class ShardTileOverlap:
    """One tile's contribution to one shard — precomputed slice coordinates."""

    embedding_path: str
    scales_path: str
    landmask_path: str
    # Tile-local region to read
    t_row_start: int
    t_row_end: int
    t_col_start: int
    t_col_end: int
    # Shard-buffer region to write into
    s_row_start: int
    s_row_end: int
    s_col_start: int
    s_col_end: int


@dataclass
class ShardSpec:
    """Everything a shard worker needs to write one complete shard."""

    sr: int  # shard row index
    sc: int  # shard col index
    row_px: int  # pixel row in zone grid (sr * SHARD_SIZE)
    col_px: int  # pixel col in zone grid (sc * SHARD_SIZE)
    tiles: List[ShardTileOverlap]
    time_index: int = 0


# ---------------------------------------------------------------------------
# UTM helpers
# ---------------------------------------------------------------------------


def epsg_to_utm_zone(epsg: int) -> int:
    """Extract UTM zone number from an EPSG code (326xx north, 327xx south)."""
    if 32601 <= epsg <= 32660:
        return epsg - 32600
    if 32701 <= epsg <= 32760:
        return epsg - 32700
    raise ValueError(f"EPSG {epsg} is not a UTM zone (expected 326xx or 327xx)")


def epsg_is_south(epsg: int) -> bool:
    """Check if an EPSG code is a southern hemisphere UTM zone."""
    return 32701 <= epsg <= 32760


def zone_canonical_epsg(zone: int) -> int:
    """Get the canonical (northern hemisphere) EPSG code for a UTM zone."""
    return 32600 + zone


def northing_to_canonical(northing: float, epsg: int) -> float:
    """Convert a northing to canonical coordinates.

    Southern hemisphere tiles use a false northing of 10,000,000m.
    We subtract this for a continuous axis.
    """
    if epsg_is_south(epsg):
        return northing - 10_000_000.0
    return northing


def _zone_group_name(zone: int) -> str:
    """Return the group name for a UTM zone within a store."""
    return f"utm{zone:02d}"


# ---------------------------------------------------------------------------
# Landmask handling
# ---------------------------------------------------------------------------


def _load_landmask_slice(
    landmask_path: str,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    """Load a sub-region of a landmask GeoTIFF using a rasterio window.

    Returns a 2D uint8 array where 0 = water.  If the landmask cannot be
    read (missing file, shape mismatch, etc.) returns all-ones (all land)
    so that no pixels are masked.
    """
    import rasterio
    from rasterio.windows import Window

    try:
        with rasterio.open(landmask_path) as src:
            window = Window.from_slices(
                (row_start, row_end),
                (col_start, col_end),
            )
            return src.read(1, window=window)
    except Exception as e:
        logger.warning(f"Failed to read landmask slice from {landmask_path}: {e}")
        return np.ones((row_end - row_start, col_end - col_start), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tile info gathering
# ---------------------------------------------------------------------------


def gather_tile_infos(
    registry: "Registry",
    year: int,
    zones: Optional[List[int]] = None,
    console: Optional["rich.console.Console"] = None,
) -> Dict[int, List[TileInfo]]:
    """Gather tile metadata and group by UTM zone.

    Computes grid info deterministically from coordinates (no file I/O).
    """
    from rasterio.transform import Affine
    from .registry import (
        EMBEDDINGS_DIR_NAME,
        LANDMASKS_DIR_NAME,
        tile_to_embedding_paths,
        tile_to_landmask_filename,
    )

    # Get tiles for this year from MultiIndex, filtering to those with data
    gdf = registry._registry_gdf
    try:
        year_slice = gdf.loc[year]
        # Filter to tiles that have actual embedding data in the registry
        valid = year_slice["file_size"] > 0
        if "scales_size" in year_slice.columns:
            valid = valid & (year_slice["scales_size"] > 0)
        year_slice = year_slice[valid]
        tiles = [
            (year, lon_i / 100.0, lat_i / 100.0)
            for lon_i, lat_i in year_slice.index.unique()
        ]
    except KeyError:
        tiles = []

    if console is not None:
        console.print(f"  Found {len(tiles):,} tiles for year {year}")

    zone_set = set(zones) if zones is not None else None

    # Pre-filter by UTM zone (deterministic from longitude)
    if zone_set is not None:
        before = len(tiles)
        tiles = [
            (y, lon, lat)
            for y, lon, lat in tiles
            if int(math.floor((lon + 180.0) / 6.0)) + 1 in zone_set
        ]
        if console is not None:
            console.print(
                f"  Filtered to {len(tiles):,} tiles in zone(s) "
                f"{','.join(str(z) for z in sorted(zone_set))} "
                f"(skipped {before - len(tiles):,})"
            )

    # Build TileInfos using computed grid (no file I/O)
    from pyproj import Transformer as ProjTransformer

    base_emb = str(registry._embeddings_dir / EMBEDDINGS_DIR_NAME)
    base_lm = str(registry._embeddings_dir / LANDMASKS_DIR_NAME)
    zones_dict: Dict[int, List[TileInfo]] = {}
    transformer_cache: Dict[int, ProjTransformer] = {}
    pixel_size = 10.0

    for tile_year, tile_lon, tile_lat in tiles:
        emb_rel, scales_rel = tile_to_embedding_paths(tile_lon, tile_lat, tile_year)
        emb_path = os.path.join(base_emb, emb_rel)
        scales_path = os.path.join(base_emb, scales_rel)
        landmask_path = os.path.join(
            base_lm, tile_to_landmask_filename(tile_lon, tile_lat)
        )

        # Compute EPSG and zone from coordinates
        zone_num = int(math.floor((tile_lon + 180) / 6)) + 1
        zone_num = max(1, min(60, zone_num))
        if zone_set is not None and zone_num not in zone_set:
            continue
        is_south = tile_lat < 0
        epsg = 32700 + zone_num if is_south else 32600 + zone_num

        # Reuse cached transformer for this EPSG
        if epsg not in transformer_cache:
            transformer_cache[epsg] = ProjTransformer.from_crs(
                "EPSG:4326", f"EPSG:{epsg}", always_xy=True
            )
        proj = transformer_cache[epsg]

        # Project tile corners to UTM
        west, east = tile_lon - 0.05, tile_lon + 0.05
        south, north = tile_lat - 0.05, tile_lat + 0.05
        ul_e, ul_n = proj.transform(west, north)
        ur_e, ur_n = proj.transform(east, north)
        ll_e, ll_n = proj.transform(west, south)
        lr_e, lr_n = proj.transform(east, south)

        origin_e = min(ul_e, ll_e)
        origin_n = max(ul_n, ur_n)
        max_e = max(ur_e, lr_e)
        min_n = min(ll_n, lr_n)

        width = round((max_e - origin_e) / pixel_size)
        height = round((origin_n - min_n) / pixel_size)
        tf_tuple = (pixel_size, 0.0, origin_e, 0.0, -pixel_size, origin_n)

        ti = TileInfo(
            lon=tile_lon,
            lat=tile_lat,
            year=tile_year,
            epsg=epsg,
            transform=Affine(*tf_tuple),
            height=height,
            width=width,
            landmask_path=landmask_path,
            embedding_path=emb_path,
            scales_path=scales_path,
        )
        zones_dict.setdefault(zone_num, []).append(ti)

    if console is not None:
        total_matched = sum(len(t) for t in zones_dict.values())
        zone_summary = ", ".join(
            f"zone {z}: {len(t)}" for z, t in sorted(zones_dict.items())
        )
        console.print(
            f"  {total_matched} tiles in {len(zones_dict)} zone(s): {zone_summary}"
        )

    return zones_dict


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def _run_parallel(
    fn, items, workers, console=None, label="Processing", progress_callback=None
):
    """Run fn(item) in a ThreadPoolExecutor, with optional Rich progress.

    Args:
        fn: Callable that takes one item and returns a result.
        items: Iterable of items to process.
        workers: Number of threads.
        console: Optional Rich Console for progress display.
        label: Description for the progress bar.
        progress_callback: Optional callable(completed, total) called after
            each item completes.  Used for cross-process progress reporting
            when ``console`` is not available.

    Returns:
        List of (item, result) tuples for successful calls.
        Failed calls are logged and skipped.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = list(items)
    results = []
    completed = 0

    def _execute(pool):
        nonlocal completed
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append((item, future.result()))
            except Exception as e:
                logger.warning(f"{label} failed for {item}: {e}")
            completed += 1
            yield item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if console is not None:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                MofNCompleteColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(label, total=len(items))
                for _ in _execute(pool):
                    progress.advance(task)
        else:
            for _ in _execute(pool):
                if progress_callback is not None:
                    progress_callback(completed, len(items))

    return results


# ---------------------------------------------------------------------------
# Global preview helpers
# ---------------------------------------------------------------------------


def _zone_output_bounds(
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
) -> Tuple[int, int, int, int]:
    """Compute the chunk-aligned output bounds for a zone in global grid pixels.

    Returns (row_start, row_end, col_start, col_end).
    """
    from pyproj import Transformer

    src_pixel = zone_transform[0]
    src_origin_e = zone_transform[2]
    src_origin_n = zone_transform[5]
    src_h, src_w = zone_shape[:2]

    west, _south, _east, north = GLOBAL_BOUNDS

    to_4326 = Transformer.from_crs(
        f"EPSG:{zone_epsg}",
        "EPSG:4326",
        always_xy=True,
    )
    corners_utm = [
        (src_origin_e, src_origin_n),
        (src_origin_e + src_w * src_pixel, src_origin_n),
        (src_origin_e, src_origin_n - src_h * src_pixel),
        (src_origin_e + src_w * src_pixel, src_origin_n - src_h * src_pixel),
    ]
    mid_e = src_origin_e + src_w * src_pixel / 2
    mid_n = src_origin_n - src_h * src_pixel / 2
    corners_utm += [
        (mid_e, src_origin_n),
        (mid_e, src_origin_n - src_h * src_pixel),
        (src_origin_e, mid_n),
        (src_origin_e + src_w * src_pixel, mid_n),
    ]
    corners_4326 = [to_4326.transform(e, n) for e, n in corners_utm]
    lons = [c[0] for c in corners_4326]
    lats = [c[1] for c in corners_4326]

    zlon_min, zlon_max = min(lons), max(lons)
    zlat_min, zlat_max = min(lats), max(lats)

    col_start = max(
        0,
        (
            int(math.floor((zlon_min - west) / GLOBAL_BASE_RES))
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )
    col_end = min(
        GLOBAL_LEVEL0_W,
        (
            (int(math.ceil((zlon_max - west) / GLOBAL_BASE_RES)) + GLOBAL_CHUNK - 1)
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )
    row_start = max(
        0,
        (
            int(math.floor((north - zlat_max) / GLOBAL_BASE_RES))
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )
    row_end = min(
        GLOBAL_LEVEL0_H,
        (
            (int(math.ceil((north - zlat_min) / GLOBAL_BASE_RES)) + GLOBAL_CHUNK - 1)
            // GLOBAL_CHUNK
            * GLOBAL_CHUNK
        ),
    )

    return (row_start, row_end, col_start, col_end)


def _coarsen_zone_pyramid(
    store_path: Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    num_levels: int,
    workers: int,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Update pyramid levels 1 through num_levels-1 for the affected region.

    Reads from the previous level and writes coarsened data to the current
    level, processing in 2D tiles parallelised with a thread pool.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import zarr

    root = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )

    prev_row_start, prev_row_end = row_start, row_end
    prev_col_start, prev_col_end = col_start, col_end

    for lvl in range(1, num_levels):
        prev_arr_path = f"global_rgb/{lvl - 1}/rgb"
        cur_arr_path = f"global_rgb/{lvl}/rgb"

        if prev_arr_path not in root or cur_arr_path not in root:
            break

        prev_arr = root[prev_arr_path]
        cur_arr = root[cur_arr_path]
        cur_h, cur_w = cur_arr.shape[:2]

        lr_start = max(0, (prev_row_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lr_end = min(
            cur_h,
            ((prev_row_end // 2 + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK),
        )
        lc_start = max(0, (prev_col_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lc_end = min(
            cur_w,
            ((prev_col_end // 2 + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK),
        )

        if lr_end <= lr_start or lc_end <= lc_start:
            break

        if console is not None:
            console.print(
                f"    Level {lvl}: rows {lr_start}-{lr_end}, cols {lc_start}-{lc_end}"
            )

        tile_size = GLOBAL_CHUNK  # output tile dimension

        def _coarsen_tile(
            r0, c0, _prev_arr=prev_arr, _cur_arr=cur_arr, _cur_h=cur_h, _cur_w=cur_w
        ):
            r1 = min(r0 + tile_size, _cur_h)
            c1 = min(c0 + tile_size, _cur_w)
            sr0 = r0 * 2
            sr1 = min(sr0 + (r1 - r0) * 2, _prev_arr.shape[0])
            sc0 = c0 * 2
            sc1 = min(sc0 + (c1 - c0) * 2, _prev_arr.shape[1])
            tile = np.asarray(_prev_arr[sr0:sr1, sc0:sc1, :]).astype(np.float32)
            th = tile.shape[0] // 2
            tw = tile.shape[1] // 2
            if th == 0 or tw == 0:
                return
            coarsened = (
                tile[: th * 2, : tw * 2, :]
                .reshape(th, 2, tw, 2, GLOBAL_NUM_BANDS)
                .mean(axis=(1, 3))
            )
            result = np.clip(coarsened, 0, 255).astype(np.uint8)
            _cur_arr[r0 : r0 + th, c0 : c0 + tw, :] = result

        tile_args = [
            (r0, c0)
            for r0 in range(lr_start, lr_end, tile_size)
            for c0 in range(lc_start, lc_end, tile_size)
        ]

        if console is not None:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                MofNCompleteColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
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
                ptask = progress.add_task(
                    f"Pyramid level {lvl}",
                    total=len(tile_args),
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_coarsen_tile, r0, c0): (r0, c0)
                        for r0, c0 in tile_args
                    }
                    for future in as_completed(futures):
                        future.result()
                        progress.advance(ptask)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_coarsen_tile, r0, c0) for r0, c0 in tile_args]
                for future in as_completed(futures):
                    future.result()

        prev_row_start, prev_row_end = lr_start, lr_end
        prev_col_start, prev_col_end = lc_start, lc_end


# ---------------------------------------------------------------------------
# Store constants
# ---------------------------------------------------------------------------

SHARD_SIZE = 4096  # spatial pixels per shard side
INNER_CHUNK = 32  # spatial pixels per inner chunk side
DEFAULT_WORKERS = 4  # fewer workers due to larger shard buffers (~2GB each)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class UnifiedZoneGrid:
    """Describes the pixel grid for a UTM zone spanning all years."""

    zone: int
    years: List[int]
    canonical_epsg: int
    origin_x: float  # UTM easting of top-left corner
    origin_y: float  # UTM northing of top-left corner
    width_px: int
    height_px: int
    pixel_size: float = 10.0


def _tile_pixel_offset(
    tile_info: TileInfo,
    grid: UnifiedZoneGrid,
) -> Tuple[int, int]:
    """Pixel offset of a tile within the unified zone grid."""
    tile_x = tile_info.transform.c
    tile_y = northing_to_canonical(tile_info.transform.f, tile_info.epsg)
    col = round((tile_x - grid.origin_x) / grid.pixel_size)
    row = round((grid.origin_y - tile_y) / grid.pixel_size)
    return row, col


# ---------------------------------------------------------------------------
# Shard index
# ---------------------------------------------------------------------------


def build_shard_index(
    tile_infos: List[TileInfo],
    grid: UnifiedZoneGrid,
    time_index: int,
) -> List[ShardSpec]:
    """Build shard index for one year's tiles against a unified zone grid."""
    shard_map: Dict[Tuple[int, int], List[ShardTileOverlap]] = {}

    for ti in tile_infos:
        row, col = _tile_pixel_offset(ti, grid)
        h, w = ti.height, ti.width

        sr_start = row // SHARD_SIZE
        sr_end = (row + h - 1) // SHARD_SIZE
        sc_start = col // SHARD_SIZE
        sc_end = (col + w - 1) // SHARD_SIZE

        for sr in range(sr_start, sr_end + 1):
            for sc in range(sc_start, sc_end + 1):
                shard_top = sr * SHARD_SIZE
                shard_left = sc * SHARD_SIZE

                t_row_start = max(0, shard_top - row)
                t_row_end = min(h, shard_top + SHARD_SIZE - row)
                t_col_start = max(0, shard_left - col)
                t_col_end = min(w, shard_left + SHARD_SIZE - col)

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
        specs.append(
            ShardSpec(
                sr=sr,
                sc=sc,
                row_px=sr * SHARD_SIZE,
                col_px=sc * SHARD_SIZE,
                time_index=time_index,
                tiles=overlaps,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Store initialisation (zarr-init)
# ---------------------------------------------------------------------------


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
    snapped to SHARD_SIZE boundaries.
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

    width_px = math.ceil(width_px / SHARD_SIZE) * SHARD_SIZE
    height_px = math.ceil(height_px / SHARD_SIZE) * SHARD_SIZE

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


def init_store(
    registry: "Registry",
    output_path: Path,
    years: List[int],
    geotessera_version: str = "unknown",
    model_version: str = "1.0",
    console: Optional["rich.console.Console"] = None,
) -> Path:
    """Create a tessera store with time dimension from the landmask registry.

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
        console.print(f"Initialising store at [bold]{output_path}[/bold]")
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
    root.attrs.update(
        {
            "zarr_conventions": [GEOEMB_CONVENTION],
            "geoemb:type": "pixel",
            "geoemb:dimensions": N_BANDS,
            "geoemb:model": f"https://geotessera.org/model/{model_version}",
            "geoemb:source_data": [
                "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
                "https://sentinel.esa.int/web/sentinel/missions/sentinel-2",
            ],
            "geoemb:data_type": "int8",
            "geoemb:gsd": 10.0,
            "geoemb:spatial_layout": "utm_zones",
            "geoemb:build_version": geotessera_version,
            "geoemb:quantization": {
                "method": "per_pixel_scale",
                "original_dtype": "float32",
                "quantized_dtype": "int8",
                "scale": {
                    "type": "array",
                    "array_name": "scales",
                    "nodata": "+inf",
                },
            },
        }
    )

    # Create each zone group from landmask coverage
    for zone_num in sorted(landmask_by_zone.keys()):
        tile_coords = landmask_by_zone[zone_num]
        grid = _compute_zone_grid_from_landmask(zone_num, tile_coords, years)

        if console:
            w_km = grid.width_px * grid.pixel_size / 1000
            h_km = grid.height_px * grid.pixel_size / 1000
            n_shards_x = grid.width_px // SHARD_SIZE
            n_shards_y = grid.height_px // SHARD_SIZE
            console.print(
                f"  Zone {zone_num} "
                f"[dim]EPSG:{grid.canonical_epsg}[/dim] "
                f"[dim]{grid.width_px}x{grid.height_px}px "
                f"({w_km:.0f}x{h_km:.0f}km) "
                f"{n_shards_x}x{n_shards_y} shards[/dim]"
            )

        _create_zone_group(grid, output_path)

    # Create empty tile registry
    _init_tile_registry(output_path)

    # Consolidate metadata so HTTP readers can discover the hierarchy
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        zarr.consolidate_metadata(str(output_path))

    if console:
        console.print(
            "  [green]Store initialised (metadata only, no data written)[/green]"
        )

    return output_path


def _create_zone_group(
    grid: UnifiedZoneGrid,
    store_path: Path,
) -> "zarr.Group":
    """Create a zone group with empty (T, B, H, W) arrays."""
    import zarr
    from zarr.codecs import BloscCodec

    zone_group = _zone_group_name(grid.zone)

    root_reopen = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )
    store = root_reopen.create_group(zone_group)

    T = len(grid.years)
    H = grid.height_px
    W = grid.width_px

    # Main data arrays — (T, B, H, W) layout
    store.create_array(
        "embeddings",
        shape=(T, N_BANDS, H, W),
        chunks=(1, N_BANDS, INNER_CHUNK, INNER_CHUNK),
        shards=(1, N_BANDS, SHARD_SIZE, SHARD_SIZE),
        dtype=np.int8,
        fill_value=np.int8(0),
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
        chunks=(1, INNER_CHUNK, INNER_CHUNK),
        shards=(1, SHARD_SIZE, SHARD_SIZE),
        dtype=np.float32,
        fill_value=np.float32("inf"),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["time", "y", "x"],
    )

    # Coordinate arrays
    x_coords = grid.origin_x + (np.arange(W) + 0.5) * grid.pixel_size
    y_coords = grid.origin_y - (np.arange(H) + 0.5) * grid.pixel_size
    time_coords = np.array(grid.years, dtype=np.int32)
    band_coords = np.arange(N_BANDS, dtype=np.int32)

    for name, data, dim in [
        ("x", x_coords, "x"),
        ("y", y_coords, "y"),
        ("time", time_coords, "time"),
        ("band", band_coords, "band"),
    ]:
        store.create_array(
            name,
            shape=data.shape,
            dtype=data.dtype,
            fill_value=0,
            compressors=None,
            dimension_names=[dim],
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
            grid.pixel_size,
            0.0,
            grid.origin_x,
            0.0,
            -grid.pixel_size,
            grid.origin_y,
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

    # Zone groups only carry proj: and spatial: conventions (geoemb: is on root)

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

    schema = gpd.GeoDataFrame(
        {
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
        }
    )
    schema.to_parquet(str(_registry_path(store_path)))


def _load_tile_registry(store_path: Path) -> "geopandas.GeoDataFrame":
    """Load the tile registry, or create it if missing."""
    import geopandas as gpd

    path = _registry_path(store_path)
    if path.exists():
        return gpd.read_parquet(str(path))
    _init_tile_registry(store_path)
    return gpd.read_parquet(str(path))


def _save_tile_registry(store_path: Path, gdf: "geopandas.GeoDataFrame") -> None:
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

_worker_store = None


def _init_shard_worker(store_path: str, zone_group: str) -> None:
    """Process pool initializer: open the zone group once per worker."""
    global _worker_store
    import zarr

    _worker_store = zarr.open_group(
        store_path,
        mode="r+",
        path=zone_group,
        zarr_format=3,
        use_consolidated=False,
    )


def _write_one_shard(spec: ShardSpec, store: "zarr.Group") -> bool:
    """Write one shard in NCHW layout: (T, B, H, W)."""
    t = spec.time_index
    S = SHARD_SIZE

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
            ov.t_row_start : ov.t_row_end,
            ov.t_col_start : ov.t_col_end,
            :,
        ]
        emb_buf[
            :,
            ov.s_row_start : ov.s_row_end,
            ov.s_col_start : ov.s_col_end,
        ] = tile_slice.transpose(2, 0, 1)

        # Scales
        scales_mmap = np.load(ov.scales_path, mmap_mode="r")
        s = scales_mmap[
            ov.t_row_start : ov.t_row_end,
            ov.t_col_start : ov.t_col_end,
        ].copy()

        # Landmask
        lm = _load_landmask_slice(
            ov.landmask_path,
            ov.t_row_start,
            ov.t_row_end,
            ov.t_col_start,
            ov.t_col_end,
        )
        s[lm == 0] = np.float32("nan")
        s[~np.isfinite(s)] = np.float32("nan")

        scales_buf[ov.s_row_start : ov.s_row_end, ov.s_col_start : ov.s_col_end] = s
        has_data = True

    if not has_data:
        return False

    r, c = spec.row_px, spec.col_px
    store["embeddings"][t, :, r : r + S, c : c + S] = emb_buf
    store["scales"][t, r : r + S, c : c + S] = scales_buf
    return True


def _write_one_shard_worker(spec: ShardSpec) -> bool:
    """Picklable wrapper for process pool."""
    return _write_one_shard(spec, _worker_store)


# ---------------------------------------------------------------------------
# Fill orchestration (zarr-fill)
# ---------------------------------------------------------------------------


def fill_store(
    registry: "Registry",
    store_path: Path,
    year: Optional[int] = None,
    zones: Optional[List[int]] = None,
    console: Optional["rich.console.Console"] = None,
    workers: Optional[int] = None,
) -> int:
    """Incrementally fill a store with tile data.

    Reads the tile registry to skip already-written tiles.
    Returns the number of shards written.
    """
    import warnings
    import zarr
    from concurrent.futures import ProcessPoolExecutor, as_completed

    store_path = Path(store_path)
    if workers is None:
        workers = DEFAULT_WORKERS

    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)
    # Derive years from the first zone's time coordinate array
    all_years: list[int] = []
    for member_name in sorted(root.keys()):
        if member_name.startswith("utm"):
            try:
                time_arr = root[member_name]["time"][:]
                all_years = [int(v) for v in time_arr]
                break
            except Exception:
                continue

    if not all_years:
        raise ValueError("Store has no years (checked root attrs and zone time coords)")

    fill_years = [year] if year is not None else all_years

    if console:
        console.print(f"Filling store at [bold]{store_path}[/bold]")
        console.print(f"  Years to fill: {fill_years}")

    total_shards_written = 0

    for fill_year in fill_years:
        if fill_year not in all_years:
            if console:
                console.print(
                    f"  [yellow]Year {fill_year} not in store, skipping[/yellow]"
                )
            continue

        time_index = all_years.index(fill_year)

        # Gather tiles for this year
        year_tiles = gather_tile_infos(
            registry,
            fill_year,
            zones=zones,
            console=console,
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
            remaining = [ti for ti in tile_infos if (ti.lon, ti.lat) not in written]

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
                str(store_path),
                mode="r",
                path=zone_group,
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
            shard_specs = build_shard_index(remaining, grid, time_index)

            if console:
                console.print(
                    f"    {len(shard_specs)} shards to write ({workers} workers)"
                )

            # Write shards via process pool
            zone_store_path = str(store_path)
            written_count = 0
            n_shards = len(shard_specs)

            if console:
                from rich.progress import (
                    Progress,
                    BarColumn,
                    TextColumn,
                    MofNCompleteColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                    SpinnerColumn,
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
                        initializer=_init_shard_worker,
                        initargs=(zone_store_path, zone_group),
                    ) as pool:
                        futures = {
                            pool.submit(_write_one_shard_worker, spec): spec
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
                    initializer=_init_shard_worker,
                    initargs=(zone_store_path, zone_group),
                ) as pool:
                    futures = {
                        pool.submit(_write_one_shard_worker, spec): spec
                        for spec in shard_specs
                    }
                    for future in as_completed(futures):
                        try:
                            if future.result():
                                written_count += 1
                        except Exception as e:
                            spec = futures[future]
                            logger.warning(f"Shard ({spec.sr},{spec.sc}) failed: {e}")

            total_shards_written += written_count

            if console:
                console.print(
                    f"    [green]{written_count}/{n_shards} shards written[/green]"
                )

            # Update tile registry
            _record_written_tiles(store_path, remaining, fill_year, zone_num)

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
        rows.append(
            {
                "year": np.int32(year),
                "zone": np.int32(zone),
                "shard_row": np.int32(0),  # TODO: compute from tile offset
                "shard_col": np.int32(0),
                "tile_lon": ti.lon,
                "tile_lat": ti.lat,
                "written_at": now,
                "geometry": Point(ti.lon, ti.lat),
            }
        )

    new_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    existing = _load_tile_registry(store_path)

    combined = gpd.GeoDataFrame(
        pd.concat([existing, new_gdf], ignore_index=True),
        crs="EPSG:4326",
    )
    _save_tile_registry(store_path, combined)


# ---------------------------------------------------------------------------
# RGB preview generation (NCHW layout)
# ---------------------------------------------------------------------------

RGB_PREVIEW_BANDS = (0, 1, 2)


def _compute_rgb_chunk(
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


def _sample_chunk_stats(
    emb_arr,
    scales_arr,
    time_index: int,
    ci: int,
    cj: int,
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
        emb_arr[time_index, band_list[0] : band_list[-1] + 1, r0:r1, c0:c1]
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


def compute_stretch(
    store: "zarr.Group",
    time_index: int,
    p_low: float = 2,
    p_high: float = 98,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
    sample_fraction: float = 0.1,
) -> dict:
    """Compute percentile stretch for RGB bands at one time step."""

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    _, _, H, W = emb_arr.shape

    n_rows = math.ceil(H / SHARD_SIZE)
    n_cols = math.ceil(W / SHARD_SIZE)
    all_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]

    n_sample = max(1, int(len(all_indices) * sample_fraction))
    rng = np.random.default_rng(42)
    sample_indices = [
        all_indices[i] for i in rng.choice(len(all_indices), n_sample, replace=False)
    ]

    results = _run_parallel(
        lambda idx: _sample_chunk_stats(
            emb_arr,
            scales_arr,
            time_index,
            idx[0],
            idx[1],
            SHARD_SIZE,
            (H, W),
        ),
        sample_indices,
        workers,
        console,
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


# ---------------------------------------------------------------------------
# Global RGB preview pyramid
# ---------------------------------------------------------------------------
# Reprojects per-zone UTM embeddings into a single EPSG:4326 RGB pyramid,
# computing RGB on the fly from bands 0-2 + scales (no stored rgb array).
# ProcessPoolExecutor for reprojection, ThreadPoolExecutor for pyramid
# coarsening.


def _ensure_global_store(store_path: Path, num_levels: int) -> None:
    """Create the global_rgb/ pyramid group within the store."""
    import zarr
    from zarr.codecs import BloscCodec

    root = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )

    # Check if already exists with correct shape
    if "global_rgb/0/rgb" in root:
        shape = root["global_rgb/0/rgb"].shape
        if shape == (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS):
            return
        import shutil

        shutil.rmtree(str(store_path / "global_rgb"))
        root = zarr.open_group(
            str(store_path), mode="r+", zarr_format=3, use_consolidated=False
        )

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
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=BloscCodec(cname="zstd", clevel=3),
            dimension_names=["lat", "lon", "band"],
        )
        lvl_grp.create_array(
            "band",
            data=band_data,
            chunks=(GLOBAL_NUM_BANDS,),
            dimension_names=["band"],
        )
        h //= 2
        w //= 2

    # Re-open the global_rgb group to ensure attrs write to the correct handle
    root = zarr.open_group(
        str(store_path), mode="r+", zarr_format=3, use_consolidated=False
    )
    global_grp = root["global_rgb"]

    # Build multiscale + spatial + proj metadata directly
    # (avoids depending on unstable topozarr API)
    from geozarr_toolkit import (
        create_geozarr_attrs,
        create_multiscales_layout,
    )
    from geozarr_toolkit.conventions.multiscales import MultiscalesConventionMetadata

    west, south, east, north_ = GLOBAL_BOUNDS
    actual_levels = len([k for k in global_grp.keys() if k.isdigit()])

    # Build multiscale layout
    h_lvl, w_lvl = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    res = GLOBAL_BASE_RES
    levels = []
    for lvl in range(actual_levels):
        entry: Dict[str, Any] = {"asset": str(lvl)}
        if lvl > 0:
            entry["derived_from"] = str(lvl - 1)
            entry["transform"] = {"scale": [2.0, 2.0], "translation": [0.0, 0.0]}
            entry["resampling_method"] = "mean"
        else:
            entry["transform"] = {"scale": [1.0, 1.0], "translation": [0.0, 0.0]}
        entry["spatial:shape"] = [h_lvl, w_lvl]
        entry["spatial:transform"] = [res, 0.0, west, 0.0, -res, north_]
        levels.append(entry)
        h_lvl //= 2
        w_lvl //= 2
        res *= 2.0

    ms_layout = create_multiscales_layout(levels, resampling_method="mean")

    # Geospatial attrs (proj + spatial)
    geozarr_attrs = create_geozarr_attrs(
        dimensions=["lat", "lon"],
        crs="EPSG:4326",
        bbox=[west, south, east, north_],
    )

    # Fix spatial description bug in geozarr-toolkit
    for conv in geozarr_attrs.get("zarr_conventions", []):
        if conv.get("uuid") == "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4":
            conv["description"] = "Spatial coordinate information"

    # Add multiscales convention registration
    ms_conv = MultiscalesConventionMetadata()
    geozarr_attrs["zarr_conventions"].insert(0, ms_conv.model_dump(exclude_none=True))

    # Merge all attrs
    geozarr_attrs.update(ms_layout)
    global_grp.attrs.update(geozarr_attrs)


# Per-worker state for reprojection
_reproj_global_arr = None
_reproj_emb_arr = None
_reproj_scales_arr = None
_reproj_to_utm = None
_reproj_time_index = None
_reproj_stretch = None


def _init_reproj_worker(
    store_path: str,
    zone_group: str,
    zone_epsg: int,
    time_index: int,
    stretch: dict,
) -> None:
    """Process pool initializer: open stores and create transformer."""
    global _reproj_global_arr, _reproj_emb_arr, _reproj_scales_arr
    global _reproj_to_utm, _reproj_time_index, _reproj_stretch
    import zarr
    from pyproj import Transformer

    root = zarr.open_group(store_path, mode="r+", zarr_format=3, use_consolidated=False)
    _reproj_global_arr = root["global_rgb/0/rgb"]
    zone = root[zone_group]
    _reproj_emb_arr = zone["embeddings"]
    _reproj_scales_arr = zone["scales"]
    _reproj_to_utm = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{zone_epsg}",
        always_xy=True,
    )
    _reproj_time_index = time_index
    _reproj_stretch = stretch


def _reproject_chunk_worker(args) -> bool:
    """Process pool worker for reprojection."""
    (
        chunk_row,
        chunk_col,
        src_epsg,
        src_pixel,
        src_origin_e,
        src_origin_n,
        src_h,
        src_w,
    ) = args
    return _reproject_chunk(
        _reproj_global_arr,
        chunk_row,
        chunk_col,
        _reproj_emb_arr,
        _reproj_scales_arr,
        _reproj_time_index,
        _reproj_stretch,
        src_epsg,
        src_pixel,
        src_origin_e,
        src_origin_n,
        src_h,
        src_w,
        _reproj_to_utm,
    )


def _reproject_chunk(
    global_arr,
    chunk_row: int,
    chunk_col: int,
    emb_arr,
    scales_arr,
    time_index: int,
    stretch: dict,
    src_epsg: int,
    src_pixel: float,
    src_origin_e: float,
    src_origin_n: float,
    src_h: int,
    src_w: int,
    to_utm,
) -> bool:
    """Reproject one 512x512 global chunk, computing RGB from embeddings on the fly."""
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
        GLOBAL_BASE_RES,
        0,
        tile_west,
        0,
        -GLOBAL_BASE_RES,
        tile_north,
    )

    # Sample corners to check zone coverage
    sample_lons = [
        tile_west,
        tile_east,
        tile_west,
        tile_east,
        (tile_west + tile_east) / 2,
    ]
    sample_lats = [
        tile_north,
        tile_north,
        tile_south,
        tile_south,
        (tile_north + tile_south) / 2,
    ]
    try:
        utm_xs, utm_ys = to_utm.transform(sample_lons, sample_lats)
    except Exception:
        return False

    if any(not math.isfinite(v) for v in list(utm_xs) + list(utm_ys)):
        return False

    # Compute source window in UTM pixel coords
    pad = 16
    r_min = max(0, int((src_origin_n - max(utm_ys)) / src_pixel) - pad)
    r_max = min(src_h, int(math.ceil((src_origin_n - min(utm_ys)) / src_pixel)) + pad)
    c_min = max(0, int((min(utm_xs) - src_origin_e) / src_pixel) - pad)
    c_max = min(src_w, int(math.ceil((max(utm_xs) - src_origin_e) / src_pixel)) + pad)

    if r_max <= r_min or c_max <= c_min:
        return False

    # Compute RGB on the fly from embeddings + scales (no stored rgb array needed)
    scales_chunk = np.asarray(scales_arr[time_index, r_min:r_max, c_min:c_max])
    valid = np.isfinite(scales_chunk)
    if not np.any(valid):
        return False

    b0, b1 = RGB_PREVIEW_BANDS[0], RGB_PREVIEW_BANDS[-1] + 1
    emb_chunk = np.asarray(emb_arr[time_index, b0:b1, r_min:r_max, c_min:c_max])
    rgba = _compute_rgb_chunk(
        emb_chunk,
        scales_chunk,
        tuple(range(b1 - b0)),
        stretch["min"],
        stretch["max"],
    )  # (4, h, w) uint8

    src_data = rgba.astype(np.float32)
    del emb_chunk, scales_chunk

    win_transform = Affine(
        src_pixel,
        0,
        src_origin_e + c_min * src_pixel,
        0,
        -src_pixel,
        src_origin_n - r_min * src_pixel,
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
        global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :] = out
    else:
        existing = np.asarray(global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :])
        existing[mask] = out[mask]
        global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :] = existing
    return True


def _reproject_zone(
    store_path: Path,
    zone_num: int,
    zone_group: str,
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
    time_index: int,
    stretch: dict,
    workers: int,
    console: Optional["rich.console.Console"] = None,
    force: bool = False,
) -> Tuple[int, int, int, int, bool]:
    """Reproject one zone's embeddings into global level 0 (computing RGB on the fly)."""
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
                console.print(f"    Zone {zone_num:02d}: already complete, skipping")
            return (row_start, row_end, col_start, col_end, False)

    chunks_total = n_chunk_rows * n_chunk_cols
    if console:
        console.print(
            f"    Zone {zone_num:02d}: {n_chunk_rows}x{n_chunk_cols} "
            f"= {chunks_total} chunks"
        )

    work_items = [
        (
            chunk_row_start + cr,
            chunk_col_start + cc,
            zone_epsg,
            src_pixel,
            src_origin_e,
            src_origin_n,
            src_h,
            src_w,
        )
        for cr in range(n_chunk_rows)
        for cc in range(n_chunk_cols)
    ]

    chunks_written = 0

    if console:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            MofNCompleteColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
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
            ptask = progress.add_task(
                f"Reprojecting zone {zone_num:02d}",
                total=len(work_items),
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_reproj_worker,
                initargs=(str(store_path), zone_group, zone_epsg, time_index, stretch),
            ) as pool:
                futures = {
                    pool.submit(_reproject_chunk_worker, item): item
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
            initializer=_init_reproj_worker,
            initargs=(str(store_path), zone_group, zone_epsg, time_index, stretch),
        ) as pool:
            futures = {
                pool.submit(_reproject_chunk_worker, item): item for item in work_items
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


def build_global_preview(
    store_path: Path,
    year: int = 2024,
    zones: Optional[List[int]] = None,
    num_levels: int = GLOBAL_DEFAULT_LEVELS,
    workers: int = 4,
    console: Optional["rich.console.Console"] = None,
    force: bool = False,
) -> None:
    """Build the global EPSG:4326 RGB pyramid from zone-level embeddings.

    Computes RGB from embeddings+scales (bands 0-2) for the specified year,
    reprojects from UTM to geographic coordinates and composites into the
    pyramid. No pre-computed rgb array needed.
    """
    import re
    import warnings
    import zarr
    import gc

    warnings.filterwarnings("ignore", message="Object at .* is not recognized")

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)

    # Derive years from first zone's time coordinate
    all_years: list[int] = []
    for member_name in sorted(root.keys()):
        if member_name.startswith("utm"):
            try:
                time_arr = root[member_name]["time"][:]
                all_years = [int(v) for v in time_arr]
                break
            except Exception:
                continue

    if not all_years:
        if console:
            console.print("[red]Error: no years found in store[/red]")
        return

    if year not in all_years:
        if console:
            console.print(
                f"[red]Error: year {year} not in store (available: {all_years})[/red]"
            )
        return

    time_index = all_years.index(year)

    if console:
        console.print(f"Building global preview for year {year} (t={time_index})")

    # Discover zones with embedding data
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

        try:
            emb_arr = zone_store["embeddings"]
            _, _, zone_h, zone_w = emb_arr.shape
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
            console.print("[yellow]No zones with embedding data found[/yellow]")
        return

    if console:
        console.print(f"  {len(zone_infos)} zone(s) with data")

    # Ensure global pyramid structure exists
    _ensure_global_store(store_path, num_levels)

    # Reproject each zone and build pyramid
    for zone_num, info in sorted(zone_infos.items()):
        if console:
            console.print(f"\n  Zone {zone_num:02d}:")

        # Compute stretch for this zone+time
        zone_store = zarr.open_group(
            str(store_path),
            mode="r",
            path=info["zone_group"],
            zarr_format=3,
            use_consolidated=False,
        )
        if console:
            console.print("    Sampling stretch...")
        stretch = compute_stretch(
            zone_store,
            time_index,
            workers=workers,
            console=console,
        )
        if console:
            console.print(
                f"    Stretch: min={[f'{v:.3f}' for v in stretch['min']]}, "
                f"max={[f'{v:.3f}' for v in stretch['max']]}"
            )

        row_start, row_end, col_start, col_end, did_work = _reproject_zone(
            store_path=store_path,
            zone_num=zone_num,
            zone_group=info["zone_group"],
            zone_epsg=info["epsg"],
            zone_transform=info["transform"],
            zone_shape=info["shape"],
            time_index=time_index,
            stretch=stretch,
            workers=workers,
            console=console,
            force=force,
        )

        if did_work:
            if console:
                console.print("    Building pyramid...")
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
        console.print("\n  [green]Global preview complete[/green]")
