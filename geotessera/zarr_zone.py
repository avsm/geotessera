"""Zone-wide Zarr format for consolidated Tessera embeddings.

This module provides tools for building and reading Zarr v3 stores that
consolidate all tiles within a UTM zone into a single store per year.
This enables efficient spatial subsetting and cloud-native access.

Store layout (uncompressed):
    utm{zone:02d}_{year}.zarr/
        embeddings        # int8    (northing, easting, band)  chunks=(1024, 1024, 128)
        scales            # float32 (northing, easting)        chunks=(1024, 1024)
        rgb               # uint8   (northing, easting, rgba)  chunks=(1024, 1024, 4)  [optional]

NaN in scales indicates no-data (water or no coverage).
Embeddings are high-entropy quantised values; compression gives negligible
benefit so we store uncompressed.
"""

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_BANDS = 128
RGB_PREVIEW_BANDS = (0, 1, 2)

# Global preview grid (fixed extent, never changes)
GLOBAL_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
GLOBAL_BASE_RES = 0.0001  # degrees (~10m at equator)
GLOBAL_LEVEL0_W = 3_600_000  # ceil(360 / 0.0001)
GLOBAL_LEVEL0_H = 1_800_000  # ceil(180 / 0.0001)
GLOBAL_CHUNK = 512
GLOBAL_NUM_BANDS = 4
GLOBAL_DEFAULT_LEVELS = 7
GLOBAL_BATCH_CHUNK_ROWS = 64  # chunk-rows per dask compute batch


# =============================================================================
# Data types
# =============================================================================


@dataclass
class TileInfo:
    """Metadata for a single tile to be placed in a zone store."""

    lon: float
    lat: float
    year: int
    epsg: int
    transform: "rasterio.transform.Affine"
    height: int
    width: int
    landmask_path: str
    embedding_path: str
    scales_path: str


@dataclass
class ZoneGrid:
    """Describes the pixel grid for a single UTM zone store."""

    zone: int
    year: int
    canonical_epsg: int
    origin_easting: float
    origin_northing: float
    width_px: int
    height_px: int
    pixel_size: float = 10.0
    tiles: List[TileInfo] = field(default_factory=list)


# =============================================================================
# Progress helpers — eliminate repeated console/no-console branching
# =============================================================================


def _run_parallel(fn, items, workers, console=None, label="Processing"):
    """Run fn(item) in a ThreadPoolExecutor, with optional Rich progress.

    Args:
        fn: Callable that takes one item and returns a result.
        items: Iterable of items to process.
        workers: Number of threads.
        console: Optional Rich Console for progress display.
        label: Description for the progress bar.

    Returns:
        List of (item, result) tuples for successful calls.
        Failed calls are logged and skipped.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = list(items)
    results = []

    def _execute(pool):
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append((item, future.result()))
            except Exception as e:
                logger.warning(f"{label} failed for {item}: {e}")
            yield item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if console is not None:
            from rich.progress import (
                Progress, SpinnerColumn, BarColumn, TextColumn,
                MofNCompleteColumn, TimeElapsedColumn,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(label, total=len(items))
                for _ in _execute(pool):
                    progress.advance(task)
        else:
            for _ in _execute(pool):
                pass

    return results


# =============================================================================
# UTM helpers
# =============================================================================


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


# =============================================================================
# Grid computation
# =============================================================================


def _snap_to_grid(value: float, pixel_size: float, snap_floor: bool) -> float:
    """Snap a coordinate to the pixel grid."""
    if snap_floor:
        return math.floor(value / pixel_size) * pixel_size
    else:
        return math.ceil(value / pixel_size) * pixel_size


def compute_zone_grid(tile_infos: List[TileInfo], year: int) -> ZoneGrid:
    """Compute the zone-wide grid that encompasses all tiles."""
    if not tile_infos:
        raise ValueError("No tiles provided")

    zone = epsg_to_utm_zone(tile_infos[0].epsg)
    pixel_size = 10.0

    min_easting = float("inf")
    max_easting = float("-inf")
    min_northing = float("inf")
    max_northing = float("-inf")

    for ti in tile_infos:
        ti_zone = epsg_to_utm_zone(ti.epsg)
        if ti_zone != zone:
            raise ValueError(
                f"Mixed zones: expected {zone}, got {ti_zone} "
                f"for tile ({ti.lon}, {ti.lat})"
            )

        tile_easting = ti.transform.c
        tile_northing = northing_to_canonical(ti.transform.f, ti.epsg)
        tile_right = tile_easting + ti.width * pixel_size
        tile_bottom = tile_northing - ti.height * pixel_size

        min_easting = min(min_easting, tile_easting)
        max_easting = max(max_easting, tile_right)
        min_northing = min(min_northing, tile_bottom)
        max_northing = max(max_northing, tile_northing)

    origin_easting = _snap_to_grid(min_easting, pixel_size, snap_floor=True)
    origin_northing = _snap_to_grid(max_northing, pixel_size, snap_floor=False)
    extent_right = _snap_to_grid(max_easting, pixel_size, snap_floor=False)
    extent_bottom = _snap_to_grid(min_northing, pixel_size, snap_floor=True)

    return ZoneGrid(
        zone=zone,
        year=year,
        canonical_epsg=zone_canonical_epsg(zone),
        origin_easting=origin_easting,
        origin_northing=origin_northing,
        width_px=round((extent_right - origin_easting) / pixel_size),
        height_px=round((origin_northing - extent_bottom) / pixel_size),
        pixel_size=pixel_size,
        tiles=tile_infos,
    )


def tile_pixel_offset(
    tile_info: TileInfo, zone_grid: ZoneGrid
) -> Tuple[int, int]:
    """Compute the (row, col) pixel offset of a tile within the zone grid."""
    tile_easting = tile_info.transform.c
    tile_northing = northing_to_canonical(tile_info.transform.f, tile_info.epsg)

    col_start = round((tile_easting - zone_grid.origin_easting) / zone_grid.pixel_size)
    row_start = round(
        (zone_grid.origin_northing - tile_northing) / zone_grid.pixel_size
    )
    return row_start, col_start


def compute_tile_grid(lon: float, lat: float, pixel_size: float = 10.0):
    """Compute expected UTM EPSG, transform, and pixel dimensions for a tile.

    Derives the UTM zone from longitude, projects the tile's WGS84 corners
    to UTM, and returns the expected grid parameters.
    """
    from pyproj import Transformer

    zone = int(math.floor((lon + 180) / 6)) + 1
    zone = max(1, min(60, zone))
    is_south = lat < 0
    epsg = 32700 + zone if is_south else 32600 + zone

    west, east = lon - 0.05, lon + 0.05
    south, north = lat - 0.05, lat + 0.05

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    ul_e, ul_n = transformer.transform(west, north)
    ur_e, ur_n = transformer.transform(east, north)
    ll_e, ll_n = transformer.transform(west, south)
    lr_e, lr_n = transformer.transform(east, south)

    origin_e = min(ul_e, ll_e)
    origin_n = max(ul_n, ur_n)
    max_e = max(ur_e, lr_e)
    min_n = min(ll_n, lr_n)

    width = round((max_e - origin_e) / pixel_size)
    height = round((origin_n - min_n) / pixel_size)

    transform_tuple = (pixel_size, 0.0, origin_e, 0.0, -pixel_size, origin_n)
    return epsg, transform_tuple, height, width


# =============================================================================
# Store creation & writing
# =============================================================================


def _store_name(zone: int, year: int) -> str:
    return f"utm{zone:02d}_{year}.zarr"


def create_zone_store(
    zone_grid: ZoneGrid,
    output_dir: Path,
    geotessera_version: str = "unknown",
    dataset_version: str = "v1",
    include_rgb: bool = False,
) -> "zarr.Group":
    """Create a new Zarr v3 store for a UTM zone."""
    import zarr

    store_path = output_dir / _store_name(zone_grid.zone, zone_grid.year)

    if store_path.exists():
        import shutil
        shutil.rmtree(store_path)

    store = zarr.open_group(store_path, mode="w", zarr_format=3)

    store.create_array(
        "embeddings",
        shape=(zone_grid.height_px, zone_grid.width_px, N_BANDS),
        chunks=(1024, 1024, N_BANDS),
        dtype=np.int8,
        fill_value=np.int8(0),
        compressors=None,
        dimension_names=["northing", "easting", "band"],
    )
    store.create_array(
        "scales",
        shape=(zone_grid.height_px, zone_grid.width_px),
        chunks=(1024, 1024),
        dtype=np.float32,
        fill_value=np.float32("nan"),
        compressors=None,
        dimension_names=["northing", "easting"],
    )

    # Optional preview arrays
    for name in (["rgb"] if include_rgb else []):
        store.create_array(
            name,
            shape=(zone_grid.height_px, zone_grid.width_px, 4),
            chunks=(1024, 1024, 4),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=None,
            dimension_names=["northing", "easting", "rgba"],
        )

    # Coordinate arrays
    easting_coords = (
        zone_grid.origin_easting
        + (np.arange(zone_grid.width_px) + 0.5) * zone_grid.pixel_size
    )
    northing_coords = (
        zone_grid.origin_northing
        - (np.arange(zone_grid.height_px) + 0.5) * zone_grid.pixel_size
    )

    for name, data, dim in [
        ("easting", easting_coords, "easting"),
        ("northing", northing_coords, "northing"),
        ("band", np.arange(N_BANDS, dtype=np.int32), "band"),
    ]:
        store.create_array(
            name, shape=data.shape, dtype=data.dtype,
            fill_value=0, compressors=None, dimension_names=[dim],
        )
        store[name][:] = data

    # CRS WKT
    try:
        from pyproj import CRS
        crs_wkt = CRS.from_epsg(zone_grid.canonical_epsg).to_wkt()
    except ImportError:
        crs_wkt = ""

    store.attrs.update({
        "utm_zone": zone_grid.zone,
        "year": zone_grid.year,
        "crs_epsg": zone_grid.canonical_epsg,
        "crs_wkt": crs_wkt,
        "transform": [
            zone_grid.pixel_size, 0.0, zone_grid.origin_easting,
            0.0, -zone_grid.pixel_size, zone_grid.origin_northing,
        ],
        "pixel_size_m": zone_grid.pixel_size,
        "geotessera_version": geotessera_version,
        "tessera_dataset_version": dataset_version,
        "n_tiles": len(zone_grid.tiles),
    })

    return store


def write_tile_to_store(
    store: "zarr.Group",
    embedding_int8: np.ndarray,
    scales: np.ndarray,
    row_start: int,
    col_start: int,
) -> None:
    """Write a single tile's embedding and scales into the zone store."""
    h, w = scales.shape[:2]
    store["embeddings"][row_start : row_start + h, col_start : col_start + w, :] = (
        embedding_int8
    )
    store["scales"][row_start : row_start + h, col_start : col_start + w] = scales


# =============================================================================
# Landmask handling
# =============================================================================


def apply_landmask_to_scales(
    scales: np.ndarray, landmask_path: str
) -> np.ndarray:
    """Apply landmask to scales, setting water pixels to NaN.

    If scales is 3D (H, W, 128), reduces to 2D (H, W) via nanmax.
    """
    import rasterio

    if scales.ndim == 3:
        scales = np.nanmax(scales, axis=2)

    scales = scales.astype(np.float32, copy=True)

    with rasterio.open(landmask_path) as src:
        landmask = src.read(1)

    water_mask = landmask == 0

    if water_mask.shape != scales.shape:
        logger.warning(
            f"Landmask shape {water_mask.shape} != scales shape {scales.shape}, "
            f"skipping landmask for {landmask_path}"
        )
        return scales

    scales[water_mask] = np.float32("nan")
    return scales


# =============================================================================
# Tile reading
# =============================================================================


def _read_single_tile(
    tile_info: TileInfo,
    zone_grid: ZoneGrid,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Read a single tile from disk and return (embedding, scales, row, col)."""
    embedding = np.load(tile_info.embedding_path)
    if embedding.ndim != 3 or embedding.shape[2] != N_BANDS:
        raise ValueError(
            f"Unexpected embedding shape {embedding.shape} for "
            f"({tile_info.lon}, {tile_info.lat})"
        )

    scales = np.load(tile_info.scales_path)
    scales = apply_landmask_to_scales(scales, tile_info.landmask_path)

    # Treat inf scales as no-data (same as NaN/water)
    scales[~np.isfinite(scales)] = np.float32("nan")

    # Zero out embeddings where scales are NaN (water/no-data)
    embedding[np.isnan(scales)] = 0

    row_start, col_start = tile_pixel_offset(tile_info, zone_grid)

    h, w = embedding.shape[:2]
    if (
        row_start < 0
        or col_start < 0
        or row_start + h > zone_grid.height_px
        or col_start + w > zone_grid.width_px
    ):
        raise ValueError(
            f"Tile ({tile_info.lon}, {tile_info.lat}) at offset "
            f"({row_start}, {col_start}) with size ({h}, {w}) "
            f"exceeds zone grid ({zone_grid.height_px}, {zone_grid.width_px})"
        )

    return embedding, scales, row_start, col_start


# =============================================================================
# Tile info gathering
# =============================================================================


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
        EMBEDDINGS_DIR_NAME, LANDMASKS_DIR_NAME,
        tile_to_embedding_paths, tile_to_landmask_filename,
    )

    # Get tiles for this year from MultiIndex
    gdf = registry._registry_gdf
    try:
        year_slice = gdf.loc[year]
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
            (y, lon, lat) for y, lon, lat in tiles
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
        ur_e, _ = proj.transform(east, north)
        ll_e, ll_n = proj.transform(west, south)
        _, ur_n = proj.transform(east, north)
        lr_e, lr_n = proj.transform(east, south)

        origin_e = min(ul_e, ll_e)
        origin_n = max(ul_n, ur_n)
        max_e = max(ur_e, lr_e)
        min_n = min(ll_n, lr_n)

        width = round((max_e - origin_e) / pixel_size)
        height = round((origin_n - min_n) / pixel_size)
        tf_tuple = (pixel_size, 0.0, origin_e, 0.0, -pixel_size, origin_n)

        ti = TileInfo(
            lon=tile_lon, lat=tile_lat, year=tile_year, epsg=epsg,
            transform=Affine(*tf_tuple),
            height=height, width=width,
            landmask_path=landmask_path,
            embedding_path=emb_path, scales_path=scales_path,
        )
        zones_dict.setdefault(zone_num, []).append(ti)

    if console is not None:
        total_matched = sum(len(t) for t in zones_dict.values())
        zone_summary = ", ".join(
            f"zone {z}: {len(t)}" for z, t in sorted(zones_dict.items())
        )
        console.print(f"  {total_matched} tiles in {len(zones_dict)} zone(s): {zone_summary}")

    return zones_dict


# =============================================================================
# Orchestration
# =============================================================================


def _default_workers() -> int:
    """Return a sensible default thread count."""
    import multiprocessing
    return min(multiprocessing.cpu_count(), 16)


def build_zone_stores(
    registry: "Registry",
    output_dir: Path,
    year: int,
    zones: Optional[List[int]] = None,
    dry_run: bool = False,
    geotessera_version: str = "unknown",
    dataset_version: str = "v1",
    console: Optional["rich.console.Console"] = None,
    rgb: bool = True,
    workers: Optional[int] = None,
) -> List[Path]:
    """Build zone-wide Zarr stores from local tile data.

    Iterates all tiles for the given year, groups by UTM zone, and writes
    one Zarr store per zone.
    """
    if workers is None:
        workers = _default_workers()

    output_dir = Path(output_dir)

    zones_dict = gather_tile_infos(
        registry, year, zones=zones, console=console,
    )

    if not zones_dict:
        if console is not None:
            console.print("  [yellow]No tiles found matching criteria[/yellow]")
        return []

    if dry_run:
        if console is not None:
            from rich.table import Table
            table = Table(show_header=True)
            table.add_column("Zone", style="cyan", justify="right")
            table.add_column("EPSG", style="dim")
            table.add_column("Tiles", justify="right")
            for zone_num, tile_infos in sorted(zones_dict.items()):
                table.add_row(
                    str(zone_num),
                    str(zone_canonical_epsg(zone_num)),
                    str(len(tile_infos)),
                )
            console.print(table)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created_stores: List[Path] = []

    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        MofNCompleteColumn, TimeElapsedColumn,
    )

    for zone_num, tile_infos in sorted(zones_dict.items()):
        zone_grid = compute_zone_grid(tile_infos, year)
        store_name = _store_name(zone_grid.zone, zone_grid.year)

        if console is not None:
            grid_w_km = zone_grid.width_px * zone_grid.pixel_size / 1000
            grid_h_km = zone_grid.height_px * zone_grid.pixel_size / 1000
            console.print(
                f"  Zone {zone_num} "
                f"[dim]EPSG:{zone_grid.canonical_epsg}[/dim] "
                f"[dim]{zone_grid.width_px}x{zone_grid.height_px}px "
                f"({grid_w_km:.0f}x{grid_h_km:.0f}km)[/dim] "
                f"-> {store_name}"
            )

        store = create_zone_store(
            zone_grid, output_dir,
            geotessera_version=geotessera_version,
            dataset_version=dataset_version,
            include_rgb=rgb,
        )

        # Write tiles: read each tile from disk, write directly to store
        errors = 0
        tiles_written = 0

        if console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                TextColumn("•"),
                TextColumn("[dim]{task.fields[status]}", justify="left"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Writing tiles", total=len(tile_infos), status="starting...",
                )
                for ti in tile_infos:
                    try:
                        emb, scales, row, col = _read_single_tile(ti, zone_grid)
                        write_tile_to_store(store, emb, scales, row, col)
                        tiles_written += 1
                    except Exception as e:
                        logger.warning(f"Failed tile ({ti.lon}, {ti.lat}): {e}")
                        errors += 1
                    progress.update(task, status=f"({ti.lon:.2f}, {ti.lat:.2f})")
                    progress.advance(task)
                status = f"done ({tiles_written} tiles)"
                if errors:
                    status += f" ({errors} errors)"
                progress.update(task, status=status)
        else:
            for ti in tile_infos:
                try:
                    emb, scales, row, col = _read_single_tile(ti, zone_grid)
                    write_tile_to_store(store, emb, scales, row, col)
                    tiles_written += 1
                except Exception as e:
                    logger.warning(f"Failed tile ({ti.lon}, {ti.lat}): {e}")
                    errors += 1

        # Optional preview passes
        if rgb:
            stretch = compute_stretch_from_store(store, workers=workers, console=console)
            if console is not None:
                console.print(
                    f"  RGB stretch: min={[f'{v:.2f}' for v in stretch['min']]}, "
                    f"max={[f'{v:.2f}' for v in stretch['max']]}"
                )
            written = write_preview_pass(
                store, "rgb",
                lambda emb, sc: compute_rgb_chunk(
                    emb, sc, RGB_PREVIEW_BANDS, stretch["min"], stretch["max"]
                ),
                workers=workers, console=console, label="Writing RGB preview",
            )
            store.attrs.update({
                "has_rgb_preview": True,
                "rgb_bands": list(RGB_PREVIEW_BANDS),
                "rgb_stretch": stretch,
            })
            if console is not None:
                console.print(f"  [green]RGB preview: {written} chunks written[/green]")

        created_stores.append(output_dir / store_name)

    return created_stores


# =============================================================================
# RGB preview
# =============================================================================


def compute_rgb_chunk(
    embedding_int8: np.ndarray,
    scales: np.ndarray,
    band_indices: tuple,
    stretch_min: List[float],
    stretch_max: List[float],
) -> np.ndarray:
    """Compute an RGBA uint8 preview from embedding + scales."""
    h, w = scales.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(scales) & (scales != 0)
    scales_safe = np.where(valid, scales, 0.0)

    for i, band_idx in enumerate(band_indices):
        raw = embedding_int8[:, :, band_idx].astype(np.float32)
        dequant = raw * scales_safe
        lo, hi = stretch_min[i], stretch_max[i]
        normalised = (dequant - lo) / (hi - lo)
        rgba[:, :, i] = np.clip(normalised * 255, 0, 255).astype(np.uint8)

    inv = ~valid
    rgba[inv, :3] = 0
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _sample_chunk_stats(
    emb_arr, scales_arr,
    ci: int, cj: int,
    chunk_h: int, chunk_w: int,
    emb_shape: tuple,
    band_slice=None,
    max_per_chunk: int = 10_000,
) -> Optional[np.ndarray]:
    """Read one chunk and return subsampled dequantised values.

    Args:
        band_slice: If given, read only these bands (e.g. slice(0,3) for RGB).
                    If None, read all bands.

    Returns:
        (N, n_bands) float32 array, or None if chunk is empty.
    """
    r0, r1 = ci * chunk_h, min(ci * chunk_h + chunk_h, emb_shape[0])
    c0, c1 = cj * chunk_w, min(cj * chunk_w + chunk_w, emb_shape[1])

    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
    valid = np.isfinite(scales_chunk) & (scales_chunk != 0)
    if not np.any(valid):
        return None

    if band_slice is not None:
        emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, band_slice])
    else:
        emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])

    vals = emb_chunk[valid].astype(np.float32) * scales_chunk[valid][:, np.newaxis]

    if vals.shape[0] > max_per_chunk:
        rng = np.random.default_rng(ci * 10007 + cj)
        idx = rng.choice(vals.shape[0], max_per_chunk, replace=False)
        vals = vals[idx]

    return vals


def compute_stretch_from_store(
    store: "zarr.Group",
    p_low: float = 2,
    p_high: float = 98,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
) -> dict:
    """Compute percentile stretch for RGB bands from an existing store."""
    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)

    band_slice = slice(RGB_PREVIEW_BANDS[0], RGB_PREVIEW_BANDS[-1] + 1)
    chunk_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]

    results = _run_parallel(
        lambda idx: _sample_chunk_stats(
            emb_arr, scales_arr, idx[0], idx[1],
            chunk_h, chunk_w, emb_shape,
            band_slice=band_slice,
        ),
        chunk_indices, workers, console,
        label=f"Computing stretch ({workers} threads)",
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



# =============================================================================
# Generic preview pass
# =============================================================================


def write_preview_pass(
    store: "zarr.Group",
    array_name: str,
    compute_fn,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
    label: str = "Writing preview",
) -> int:
    """Write a preview array by iterating over chunks in parallel.

    For each chunk, reads embeddings + scales, calls compute_fn(emb, scales)
    to get an RGBA array, and writes it to store[array_name].

    Args:
        store: Zarr group with embeddings, scales, and the target array.
        array_name: Name of the output array (e.g. "rgb").
        compute_fn: Callable(emb_int8, scales_f32) -> rgba_uint8.
        workers: Number of threads.
        console: Optional Rich Console for progress.
        label: Description for the progress bar.

    Returns:
        Number of chunks written.
    """
    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    out_arr = store[array_name]

    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)

    def _process_chunk(idx):
        ci, cj = idx
        r0, r1 = ci * chunk_h, min(ci * chunk_h + chunk_h, emb_shape[0])
        c0, c1 = cj * chunk_w, min(cj * chunk_w + chunk_w, emb_shape[1])

        scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
        if np.all(~np.isfinite(scales_chunk) | (scales_chunk == 0)):
            return False

        emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
        rgba = compute_fn(emb_chunk, scales_chunk)
        out_arr[r0:r1, c0:c1, :] = rgba
        return True

    chunk_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]

    results = _run_parallel(
        _process_chunk, chunk_indices, workers, console,
        label=f"{label} ({workers} threads)",
    )

    return sum(1 for _, wrote in results if wrote)


# =============================================================================
# Standalone preview commands (--rgb-only)
# =============================================================================


def add_rgb_to_existing_store(
    store_path: Path,
    workers: Optional[int] = None,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add RGB preview array to an existing Zarr store."""
    import zarr

    store = zarr.open_group(str(store_path), mode="r+")

    try:
        _ = store["rgb"]
    except KeyError:
        emb_shape = store["embeddings"].shape
        store.create_array(
            "rgb",
            shape=(emb_shape[0], emb_shape[1], 4),
            chunks=(1024, 1024, 4),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=None,
            dimension_names=["northing", "easting", "rgba"],
        )

    if workers is None:
        workers = _default_workers()

    if console is not None:
        console.print(f"  Pass 1: Computing band statistics ({workers} threads)...")

    stretch = compute_stretch_from_store(store, workers=workers, console=console)
    if console is not None:
        console.print(f"  Stretch: min={stretch['min']}, max={stretch['max']}")
        console.print("  Pass 2: Writing RGB preview...")

    written = write_preview_pass(
        store, "rgb",
        lambda emb, sc: compute_rgb_chunk(
            emb, sc, RGB_PREVIEW_BANDS, stretch["min"], stretch["max"]
        ),
        workers=workers, console=console, label="Writing RGB preview",
    )

    store.attrs.update({
        "has_rgb_preview": True,
        "rgb_bands": list(RGB_PREVIEW_BANDS),
        "rgb_stretch": stretch,
    })

    if console is not None:
        console.print(f"  [green]RGB preview: {written} chunks written[/green]")



def _utm_array_to_xarray(
    store: "zarr.Group", array_name: str
) -> "xarray.DataArray":
    """Wrap a UTM zarr preview array as a georeferenced xarray DataArray.

    Uses dask to lazily reference the zarr array without loading it all
    into memory.  Constructs pixel-centre coordinates from the store's
    affine transform and attaches CRS metadata via rioxarray.

    The returned DataArray has dims ``('band', 'y', 'x')`` — the order
    required by rioxarray / rasterio for reprojection.

    Args:
        store: Zarr group containing the array and transform/CRS attrs.
        array_name: Name of the preview array (e.g. ``"rgb"``).

    Returns:
        Dask-backed georeferenced xarray DataArray ready for reprojection.
    """
    import dask.array as da
    import rioxarray  # noqa: F401 — registers .rio accessor
    import xarray as xr
    from affine import Affine

    arr = store[array_name]
    # Lazy dask array — no data loaded into memory yet
    dask_arr = da.from_zarr(arr)  # (northing, easting, band)

    store_attrs = dict(store.attrs)
    transform = list(store_attrs["transform"])
    epsg = int(store_attrs["crs_epsg"])

    pixel_size = transform[0]
    origin_e = transform[2]
    origin_n = transform[5]

    h, w, nc = arr.shape

    # Pixel-centre coordinates
    x_coords = origin_e + (np.arange(w) + 0.5) * pixel_size
    y_coords = origin_n - (np.arange(h) + 0.5) * pixel_size

    # Transpose to (band, y, x) as required by rioxarray
    dask_byx = da.transpose(dask_arr, (2, 0, 1))

    xda = xr.DataArray(
        dask_byx,
        dims=["band", "y", "x"],
        coords={
            "y": y_coords,
            "x": x_coords,
        },
    )
    xda = xda.rio.write_crs(f"EPSG:{epsg}")
    xda = xda.rio.write_transform(Affine(*transform))
    xda = xda.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return xda


# =============================================================================
# Reading support
# =============================================================================


def open_zone_store(path) -> "xarray.Dataset":
    """Open a zone Zarr store as an xarray Dataset."""
    import xarray as xr
    return xr.open_zarr(str(path))


def read_region_from_zone(
    path,
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read a spatial subset from a zone store.

    Args:
        path: Path to the .zarr directory
        bbox: (min_easting, min_northing, max_easting, max_northing) in UTM

    Returns:
        (embeddings_int8, scales_float32, attrs)
    """
    import zarr

    store = zarr.open_group(str(path), mode="r")
    attrs = dict(store.attrs)

    transform = attrs["transform"]
    pixel_size = transform[0]
    origin_easting = transform[2]
    origin_northing = transform[5]

    min_e, min_n, max_e, max_n = bbox

    col_start = max(0, int(math.floor((min_e - origin_easting) / pixel_size)))
    col_end = min(
        store["scales"].shape[1],
        int(math.ceil((max_e - origin_easting) / pixel_size)),
    )
    row_start = max(0, int(math.floor((origin_northing - max_n) / pixel_size)))
    row_end = min(
        store["scales"].shape[0],
        int(math.ceil((origin_northing - min_n) / pixel_size)),
    )

    embeddings = store["embeddings"][row_start:row_end, col_start:col_end, :]
    scales = store["scales"][row_start:row_end, col_start:col_end]

    return np.asarray(embeddings), np.asarray(scales), attrs


def _ensure_global_store(store_path: Path, num_levels: int) -> None:
    """Create the global preview store with fixed dimensions if it doesn't exist.

    If the store already exists with correct dimensions, this is a no-op.
    Creates level 0 through level (num_levels-1), each with an 'rgb' array
    and a 'band' coordinate array.
    """
    import json as _json
    import zarr
    from zarr.codecs import BloscCodec

    if store_path.exists():
        root = zarr.open_group(str(store_path), mode="r")
        if "0/rgb" in root:
            shape = root["0/rgb"].shape
            if shape == (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS):
                return
            # Old-format store with different dimensions — replace it
            import shutil
            logger.warning(
                "Existing store %s has shape %s (expected %s), "
                "replacing with fixed global grid",
                store_path, shape,
                (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS),
            )
            shutil.rmtree(str(store_path))

    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    h, w = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    band_data = np.arange(GLOBAL_NUM_BANDS, dtype=np.int32)

    for lvl in range(num_levels):
        if h < 1 or w < 1:
            break
        level_dir = os.path.join(str(store_path), str(lvl))
        os.makedirs(level_dir, exist_ok=True)
        group_meta = os.path.join(level_dir, "zarr.json")
        if not os.path.exists(group_meta):
            with open(group_meta, "w") as f:
                _json.dump(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}},
                    f,
                )
        root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
        root.create_array(
            f"{lvl}/rgb",
            shape=(h, w, GLOBAL_NUM_BANDS),
            chunks=(GLOBAL_CHUNK, GLOBAL_CHUNK, GLOBAL_NUM_BANDS),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=BloscCodec(cname="zstd", clevel=3),
            dimension_names=["lat", "lon", "band"],
        )
        root.create_array(
            f"{lvl}/band",
            data=band_data,
            chunks=(GLOBAL_NUM_BANDS,),
        )
        h //= 2
        w //= 2

    from topozarr.metadata import create_multiscale_metadata
    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
    actual_levels = len([k for k in root.keys() if k.isdigit()])
    ms_attrs = create_multiscale_metadata(actual_levels, "EPSG:4326", "mean")
    ms_attrs["multiscales"]["crs"] = "EPSG:4326"
    west, south, east, north = GLOBAL_BOUNDS
    ms_attrs["spatial"] = {
        "bounds": [west, south, east, north],
        "resolution": GLOBAL_BASE_RES,
    }
    root.attrs.update(ms_attrs)
    zarr.consolidate_metadata(str(store_path))


def _reproject_chunk(
    global_arr,
    chunk_row: int,
    chunk_col: int,
    src_arr,
    src_epsg: int,
    src_pixel: float,
    src_origin_e: float,
    src_origin_n: float,
    src_h: int,
    src_w: int,
    to_utm,
) -> bool:
    """Reproject one 512x512 output chunk from UTM source and write to global_arr.

    Each call writes to a unique (chunk_row, chunk_col) position, so concurrent
    calls to different positions are safe.

    Returns True if any non-zero data was written.
    """
    from affine import Affine
    from rasterio.enums import Resampling
    import rasterio.warp

    west, south, east, north = GLOBAL_BOUNDS
    row0 = chunk_row * GLOBAL_CHUNK
    col0 = chunk_col * GLOBAL_CHUNK
    tile_h = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_H - row0)
    tile_w = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_W - col0)
    if tile_h <= 0 or tile_w <= 0:
        return False

    tile_west = west + col0 * GLOBAL_BASE_RES
    tile_north = north - row0 * GLOBAL_BASE_RES
    tile_east = tile_west + tile_w * GLOBAL_BASE_RES
    tile_south = tile_north - tile_h * GLOBAL_BASE_RES

    dst_transform = Affine(
        GLOBAL_BASE_RES, 0, tile_west,
        0, -GLOBAL_BASE_RES, tile_north,
    )

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

    window = np.asarray(src_arr[r_min:r_max, c_min:c_max, :])
    if not window.any():
        return False

    src_data = np.transpose(window.astype(np.float32), (2, 0, 1))
    del window

    win_transform = Affine(
        src_pixel, 0, src_origin_e + c_min * src_pixel,
        0, -src_pixel, src_origin_n - r_min * src_pixel,
    )

    dst_data = np.full(
        (GLOBAL_NUM_BANDS, tile_h, tile_w), np.nan, dtype=np.float32,
    )

    try:
        rasterio.warp.reproject(
            source=src_data,
            destination=dst_data,
            src_transform=win_transform,
            src_crs=f"EPSG:{src_epsg}",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
    except Exception:
        return False

    del src_data
    dst_data = np.nan_to_num(dst_data, nan=0.0)
    dst_data = np.clip(dst_data, 0, 255).astype(np.uint8)
    out = np.transpose(dst_data, (1, 2, 0))
    del dst_data

    if not out.any():
        return False

    global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :] = out
    return True


def _reproject_zone(
    store_path: Path,
    zone_num: int,
    zone_store_path: Path,
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
    workers: int,
    console: Optional["rich.console.Console"] = None,
) -> Tuple[int, int, int, int]:
    """Reproject one zone's RGB into level 0 of the global store.

    Uses dask.delayed to parallelise chunk-level reprojection tasks.
    Each task writes to a unique chunk position, eliminating write races.

    Returns (row_start, row_end, col_start, col_end) in pixel coords,
    snapped to chunk boundaries.
    """
    import dask
    import zarr
    from pyproj import Transformer

    src_pixel = zone_transform[0]
    src_origin_e = zone_transform[2]
    src_origin_n = zone_transform[5]
    src_h, src_w = zone_shape[:2]

    west, south, east, north = GLOBAL_BOUNDS

    # Compute zone's WGS84 bounding box
    to_4326 = Transformer.from_crs(
        f"EPSG:{zone_epsg}", "EPSG:4326", always_xy=True,
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

    # Snap to chunk boundaries (expand outward)
    col_start = max(0, (int(math.floor((zlon_min - west) / GLOBAL_BASE_RES))
                        // GLOBAL_CHUNK * GLOBAL_CHUNK))
    col_end = min(GLOBAL_LEVEL0_W,
                  ((int(math.ceil((zlon_max - west) / GLOBAL_BASE_RES))
                    + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK))
    row_start = max(0, (int(math.floor((north - zlat_max) / GLOBAL_BASE_RES))
                        // GLOBAL_CHUNK * GLOBAL_CHUNK))
    row_end = min(GLOBAL_LEVEL0_H,
                  ((int(math.ceil((north - zlat_min) / GLOBAL_BASE_RES))
                    + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK))

    if col_end <= col_start or row_end <= row_start:
        if console is not None:
            console.print(f"    [yellow]Zone {zone_num}: no output region[/yellow]")
        return (0, 0, 0, 0)

    n_chunk_rows = (row_end - row_start) // GLOBAL_CHUNK
    n_chunk_cols = (col_end - col_start) // GLOBAL_CHUNK
    chunk_row_start = row_start // GLOBAL_CHUNK
    chunk_col_start = col_start // GLOBAL_CHUNK

    if console is not None:
        total_chunks = n_chunk_rows * n_chunk_cols
        console.print(
            f"    Zone {zone_num:02d}: rows {row_start}-{row_end}, "
            f"cols {col_start}-{col_end} "
            f"({n_chunk_rows}x{n_chunk_cols} = {total_chunks} chunks)"
        )

    global_root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
    global_arr = global_root["0/rgb"]
    zone_store = zarr.open_group(str(zone_store_path), mode="r")
    src_arr = zone_store["rgb"]

    to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{zone_epsg}", always_xy=True,
    )

    chunks_written = 0
    chunks_total = n_chunk_rows * n_chunk_cols

    for batch_start in range(0, n_chunk_rows, GLOBAL_BATCH_CHUNK_ROWS):
        batch_end = min(batch_start + GLOBAL_BATCH_CHUNK_ROWS, n_chunk_rows)

        tasks = []
        for cr_offset in range(batch_start, batch_end):
            cr = chunk_row_start + cr_offset
            for cc_offset in range(n_chunk_cols):
                cc = chunk_col_start + cc_offset
                task = dask.delayed(_reproject_chunk)(
                    global_arr=global_arr,
                    chunk_row=cr,
                    chunk_col=cc,
                    src_arr=src_arr,
                    src_epsg=zone_epsg,
                    src_pixel=src_pixel,
                    src_origin_e=src_origin_e,
                    src_origin_n=src_origin_n,
                    src_h=src_h,
                    src_w=src_w,
                    to_utm=to_utm,
                )
                tasks.append(task)

        results = dask.compute(*tasks, scheduler="threads",
                               num_workers=workers)
        batch_written = sum(1 for r in results if r)
        chunks_written += batch_written

        if console is not None:
            done = min((batch_end) * n_chunk_cols, chunks_total)
            pct = int(100 * done / chunks_total)
            console.print(
                f"      [{pct:3d}%] {done}/{chunks_total} chunks "
                f"({chunks_written} with data)"
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
    level, processing in row-strips parallelised with dask.delayed.
    """
    import dask
    import zarr

    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)

    prev_row_start, prev_row_end = row_start, row_end
    prev_col_start, prev_col_end = col_start, col_end

    for lvl in range(1, num_levels):
        prev_arr_path = f"{lvl - 1}/rgb"
        cur_arr_path = f"{lvl}/rgb"

        if prev_arr_path not in root or cur_arr_path not in root:
            break

        prev_arr = root[prev_arr_path]
        cur_arr = root[cur_arr_path]
        cur_h, cur_w = cur_arr.shape[:2]

        lr_start = max(0, (prev_row_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lr_end = min(cur_h, ((prev_row_end // 2 + GLOBAL_CHUNK - 1)
                             // GLOBAL_CHUNK * GLOBAL_CHUNK))
        lc_start = max(0, (prev_col_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lc_end = min(cur_w, ((prev_col_end // 2 + GLOBAL_CHUNK - 1)
                             // GLOBAL_CHUNK * GLOBAL_CHUNK))

        if lr_end <= lr_start or lc_end <= lc_start:
            break

        if console is not None:
            console.print(
                f"    Level {lvl}: rows {lr_start}-{lr_end}, "
                f"cols {lc_start}-{lc_end}"
            )

        strip_h = GLOBAL_CHUNK

        def _coarsen_strip(r0, _prev_arr=prev_arr, _cur_arr=cur_arr,
                           _lc_start=lc_start, _lc_end=lc_end,
                           _cur_h=cur_h):
            r1 = min(r0 + strip_h, _cur_h)
            sr0 = r0 * 2
            sr1 = min(sr0 + (r1 - r0) * 2, _prev_arr.shape[0])
            sc0 = _lc_start * 2
            sc1 = min(sc0 + (_lc_end - _lc_start) * 2, _prev_arr.shape[1])
            strip = np.asarray(
                _prev_arr[sr0:sr1, sc0:sc1, :]
            ).astype(np.float32)
            th = strip.shape[0] // 2
            tw = strip.shape[1] // 2
            if th == 0 or tw == 0:
                return
            coarsened = (
                strip[: th * 2, : tw * 2, :]
                .reshape(th, 2, tw, 2, GLOBAL_NUM_BANDS)
                .mean(axis=(1, 3))
            )
            result = np.clip(coarsened, 0, 255).astype(np.uint8)
            _cur_arr[r0 : r0 + th, _lc_start : _lc_start + tw, :] = result

        strip_starts = list(range(lr_start, lr_end, strip_h))
        tasks = [dask.delayed(_coarsen_strip)(r0) for r0 in strip_starts]
        dask.compute(*tasks, scheduler="threads", num_workers=workers)

        prev_row_start, prev_row_end = lr_start, lr_end
        prev_col_start, prev_col_end = lc_start, lc_end


def build_global_preview(
    zarr_dir: Path,
    year: int,
    zones: Optional[List[int]] = None,
    num_levels: int = GLOBAL_DEFAULT_LEVELS,
    workers: int = 4,
    console: Optional["rich.console.Console"] = None,
) -> Path:
    """Create or update global EPSG:4326 preview store from per-zone UTM stores.

    The output store is always ``{zarr_dir}/global_rgb_{year}.zarr``.
    If the store already exists, only the specified zones are re-processed.
    """
    import gc
    import zarr

    if console is not None:
        console.print(f"[bold]Building global preview (year={year})[/bold]")

    # 1. Discover zone stores
    pattern = re.compile(rf"^utm(\d{{2}})_{year}\.zarr$")
    zone_stores: Dict[int, Path] = {}

    for entry in sorted(zarr_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m is None:
            continue
        zone_num = int(m.group(1))
        if zones is not None and zone_num not in zones:
            continue
        zone_stores[zone_num] = entry

    if not zone_stores:
        msg = f"No zone stores found in {zarr_dir} for year {year}"
        if zones is not None:
            msg += f" (zones filter: {zones})"
        raise FileNotFoundError(msg)

    if console is not None:
        console.print(
            f"  Found {len(zone_stores)} zone store(s): "
            f"{sorted(zone_stores.keys())}"
        )

    # 2. Read zone metadata
    zone_infos: Dict[int, dict] = {}
    for zone_num, store_path in sorted(zone_stores.items()):
        store = zarr.open_group(str(store_path), mode="r")
        attrs = dict(store.attrs)
        if "rgb" not in store:
            if console is not None:
                console.print(
                    f"  [yellow]Zone {zone_num}: no rgb array, skipping[/yellow]"
                )
            continue
        zone_infos[zone_num] = {
            "store_path": store_path,
            "epsg": int(attrs["crs_epsg"]),
            "transform": list(attrs["transform"]),
            "shape": store["rgb"].shape,
        }

    if not zone_infos:
        raise FileNotFoundError("No zone stores with rgb arrays found")

    # 3. Ensure global store exists
    output_path = zarr_dir / f"global_rgb_{year}.zarr"
    if console is not None:
        console.print(f"  Output: {output_path}")
    _ensure_global_store(output_path, num_levels)
    if console is not None:
        console.print(
            f"  Global grid: {GLOBAL_LEVEL0_W}x{GLOBAL_LEVEL0_H} "
            f"@ {GLOBAL_BASE_RES} deg, {num_levels} levels"
        )

    # 4. Reproject each zone and update pyramid
    for zone_num, zinfo in sorted(zone_infos.items()):
        if console is not None:
            console.print(
                f"\n  [bold]Processing zone {zone_num:02d} "
                f"(EPSG:{zinfo['epsg']})[/bold]"
            )
        row_start, row_end, col_start, col_end = _reproject_zone(
            store_path=output_path,
            zone_num=zone_num,
            zone_store_path=zinfo["store_path"],
            zone_epsg=zinfo["epsg"],
            zone_transform=zinfo["transform"],
            zone_shape=zinfo["shape"],
            workers=workers,
            console=console,
        )
        if row_end <= row_start or col_end <= col_start:
            continue
        if console is not None:
            console.print(f"    Building pyramid...")
        _coarsen_zone_pyramid(
            store_path=output_path,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            num_levels=num_levels,
            workers=workers,
            console=console,
        )
        gc.collect()

    # 5. Re-consolidate metadata
    zarr.consolidate_metadata(str(output_path))
    if console is not None:
        console.print(
            f"\n  [bold green]Global store updated: {output_path}[/bold green]"
        )
        console.print(f"  Zones processed: {sorted(zone_infos.keys())}")

    return output_path
