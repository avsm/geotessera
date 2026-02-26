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


def build_global_preview(
    zarr_dir: Path,
    output_path: Path,
    year: int,
    zones: Optional[List[int]] = None,
    num_levels: int = 7,
    preview_names: Optional[List[str]] = None,
    console: Optional["rich.console.Console"] = None,
) -> Path:
    """Build global EPSG:4326 preview store from per-zone UTM stores.

    Reprojects each zone's rgb array from UTM to WGS84,
    writes into a single global zarr store with zarr-conventions/multiscales
    metadata for use with @carbonplan/zarr-layer.

    Steps:
        1. Discover zone stores matching ``utm{ZZ}_{year}.zarr`` in *zarr_dir*.
        2. Compute the union WGS84 bounding box across all discovered zones.
        3. Determine global array dimensions at ~0.0001 deg resolution.
        4. Delegate to :func:`_write_global_store` to create the output.

    Args:
        zarr_dir: Directory containing per-zone ``.zarr`` stores.
        output_path: Path for the output global zarr store.
        year: Year to filter zone store filenames.
        zones: Optional list of UTM zone numbers to include. If *None*,
            all matching stores are used.
        num_levels: Number of resolution levels in the output pyramid.
        preview_names: List of preview array names to include (e.g.
            ``["rgb"]``). Defaults to ``["rgb"]``.
        console: Optional Rich Console for status messages.

    Returns:
        Path to the created global zarr store.
    """
    import zarr
    from pyproj import Transformer

    if preview_names is None:
        preview_names = ["rgb"]

    if console is not None:
        console.print(
            f"[bold]Scanning {zarr_dir} for zone stores (year={year})...[/bold]"
        )

    # ------------------------------------------------------------------
    # 1. Discover zone stores
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Compute global WGS84 bounding box
    # ------------------------------------------------------------------
    global_lon_min = float("inf")
    global_lon_max = float("-inf")
    global_lat_min = float("inf")
    global_lat_max = float("-inf")

    zone_infos: Dict[int, dict] = {}

    for zone_num, store_path in sorted(zone_stores.items()):
        store = zarr.open_group(str(store_path), mode="r")
        attrs = dict(store.attrs)

        epsg = int(attrs["crs_epsg"])
        transform = list(attrs["transform"])
        pixel_size = transform[0]
        origin_e = transform[2]
        origin_n = transform[5]

        # Find the first available preview array to get the shape
        preview_shape = None
        for pname in preview_names:
            if pname in store:
                preview_shape = store[pname].shape
                break

        if preview_shape is None:
            if console is not None:
                console.print(
                    f"  [yellow]Zone {zone_num}: no preview arrays found, "
                    f"skipping[/yellow]"
                )
            continue

        h, w = preview_shape[:2]

        # UTM corner coordinates
        corners_utm = [
            (origin_e, origin_n),                             # top-left
            (origin_e + w * pixel_size, origin_n),            # top-right
            (origin_e, origin_n - h * pixel_size),            # bottom-left
            (origin_e + w * pixel_size, origin_n - h * pixel_size),  # bottom-right
        ]

        # Add mid-edge points for better bounding at high latitudes
        mid_e = origin_e + w * pixel_size / 2
        mid_n = origin_n - h * pixel_size / 2
        corners_utm += [
            (mid_e, origin_n),                   # top-centre
            (mid_e, origin_n - h * pixel_size),  # bottom-centre
            (origin_e, mid_n),                   # left-centre
            (origin_e + w * pixel_size, mid_n),  # right-centre
        ]

        # Reproject corners to WGS84
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )
        corners_4326 = [
            transformer.transform(e, n) for e, n in corners_utm
        ]

        lons = [c[0] for c in corners_4326]
        lats = [c[1] for c in corners_4326]

        zone_lon_min, zone_lon_max = min(lons), max(lons)
        zone_lat_min, zone_lat_max = min(lats), max(lats)

        # Union into the global bounding box
        global_lon_min = min(global_lon_min, zone_lon_min)
        global_lon_max = max(global_lon_max, zone_lon_max)
        global_lat_min = min(global_lat_min, zone_lat_min)
        global_lat_max = max(global_lat_max, zone_lat_max)

        zone_infos[zone_num] = {
            "store_path": store_path,
            "epsg": epsg,
            "transform": transform,
            "shape": preview_shape,
            "bounds_4326": (
                zone_lon_min, zone_lat_min, zone_lon_max, zone_lat_max,
            ),
        }

        if console is not None:
            console.print(
                f"  Zone {zone_num:02d} (EPSG:{epsg}): "
                f"{h}x{w} px, "
                f"lon [{zone_lon_min:.4f}, {zone_lon_max:.4f}], "
                f"lat [{zone_lat_min:.4f}, {zone_lat_max:.4f}]"
            )

    if not zone_infos:
        raise FileNotFoundError(
            "No zone stores with valid preview arrays found"
        )

    global_bounds = (
        global_lon_min, global_lat_min, global_lon_max, global_lat_max,
    )

    if console is not None:
        console.print(
            f"  [bold]Global extent:[/bold] "
            f"lon [{global_lon_min:.4f}, {global_lon_max:.4f}], "
            f"lat [{global_lat_min:.4f}, {global_lat_max:.4f}]"
        )

    # ------------------------------------------------------------------
    # 3. Compute global array dimensions
    # ------------------------------------------------------------------
    base_res = 0.0001  # ~10 m at equator

    global_w = int(
        math.ceil((global_lon_max - global_lon_min) / base_res)
    )
    global_h = int(
        math.ceil((global_lat_max - global_lat_min) / base_res)
    )

    if console is not None:
        console.print(
            f"  [dim]Base resolution: {base_res}° "
            f"→ {global_w}x{global_h} pixels[/dim]"
        )
        console.print(
            f"  [dim]Pyramid levels: {num_levels}[/dim]"
        )

    # ------------------------------------------------------------------
    # 4. Create the global store
    # ------------------------------------------------------------------
    _write_global_store(
        output_path=output_path,
        zone_infos=zone_infos,
        preview_names=preview_names,
        global_bounds=global_bounds,
        base_res=base_res,
        num_levels=num_levels,
        console=console,
    )

    return output_path


def _write_global_store(
    output_path: Path,
    zone_infos: dict,
    preview_names: List[str],
    global_bounds: Tuple[float, float, float, float],
    base_res: float,
    num_levels: int,
    console: "rich.console.Console",
) -> None:
    """Create global 4326 zarr store and reproject each zone into it.

    Two phases:
      1. Reproject each zone's preview arrays from UTM to WGS84 using
         windowed rasterio.warp.reproject() into a temporary level-0
         zarr store.  Peak memory ≈ one strip of source + destination.
      2. Open level-0 as a dask-backed xarray Dataset, use topozarr
         to build coarsened pyramid levels, and write the final store.
    """
    import gc
    import shutil
    import tempfile

    import dask.array as da
    import rasterio.warp
    import xarray as xr
    import xproj  # noqa: F401 — registers .proj accessor
    import zarr
    from affine import Affine
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from topozarr import create_pyramid
    from zarr.codecs import BloscCodec

    west, south, east, north = global_bounds
    num_bands = 4
    level0_w = int(math.ceil((east - west) / base_res))
    level0_h = int(math.ceil((north - south) / base_res))

    # ==================================================================
    # Phase 1: Reproject zones into a temporary level-0 zarr store
    # ==================================================================
    tmp_dir = tempfile.mkdtemp(prefix="geotessera_global_")
    tmp_store_path = os.path.join(tmp_dir, "level0.zarr")

    if console is not None:
        console.print(
            f"  [dim]Phase 1: Reprojecting zones into level-0 "
            f"({level0_w}x{level0_h} @ {base_res}°)[/dim]"
        )

    tmp_root = zarr.open_group(tmp_store_path, mode="w", zarr_format=3)

    for preview_name in preview_names:
        if console is not None:
            console.print(
                f"\n  [bold]Reprojecting: {preview_name}[/bold]"
            )

        tmp_arr = tmp_root.create_array(
            preview_name,
            shape=(level0_h, level0_w, num_bands),
            chunks=(512, 512, num_bands),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=BloscCodec(cname="zstd", clevel=3),
        )

        for zone_num, zinfo in sorted(zone_infos.items()):
            store_path = zinfo["store_path"]
            zone_store = zarr.open_group(str(store_path), mode="r")

            if preview_name not in zone_store:
                continue

            src_arr = zone_store[preview_name]
            src_h, src_w, _ = src_arr.shape
            src_epsg = zinfo["epsg"]
            src_transform = zinfo["transform"]
            src_pixel = src_transform[0]
            src_origin_e = src_transform[2]
            src_origin_n = src_transform[5]

            to_utm = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src_epsg}", always_xy=True,
            )

            zlon_min, zlat_min, zlon_max, zlat_max = (
                zinfo["bounds_4326"]
            )
            zone_col_start = max(
                0, int(math.floor((zlon_min - west) / base_res))
            )
            zone_col_end = min(
                level0_w,
                int(math.ceil((zlon_max - west) / base_res)),
            )
            zone_strip_w = zone_col_end - zone_col_start
            if zone_strip_w <= 0:
                continue

            zone_row_start = max(
                0, int(math.floor((north - zlat_max) / base_res))
            )
            zone_row_end = min(
                level0_h,
                int(math.ceil((north - zlat_min) / base_res)),
            )
            if zone_row_end <= zone_row_start:
                continue

            if console is not None:
                console.print(
                    f"    Zone {zone_num:02d} "
                    f"(src {src_h}x{src_w}, "
                    f"dst rows {zone_row_start}-{zone_row_end}, "
                    f"cols {zone_col_start}-{zone_col_end})"
                )

            strip_height = 2048
            strips_written = 0

            for strip_start in range(
                zone_row_start, zone_row_end, strip_height
            ):
                strip_end = min(
                    strip_start + strip_height, zone_row_end
                )
                strip_h = strip_end - strip_start

                dst_north = north - strip_start * base_res
                dst_west = west + zone_col_start * base_res

                dst_transform = Affine(
                    base_res, 0, dst_west,
                    0, -base_res, dst_north,
                )

                # Back-project corners + midpoint to UTM to find
                # the source window we need
                dst_east = west + zone_col_end * base_res
                dst_south = north - strip_end * base_res
                sample_lons = [
                    dst_west, dst_east, dst_west, dst_east,
                    (dst_west + dst_east) / 2,
                ]
                sample_lats = [
                    dst_north, dst_north, dst_south, dst_south,
                    (dst_north + dst_south) / 2,
                ]
                try:
                    utm_xs, utm_ys = to_utm.transform(
                        sample_lons, sample_lats
                    )
                except Exception:
                    continue

                if any(
                    not math.isfinite(v)
                    for v in list(utm_xs) + list(utm_ys)
                ):
                    continue

                pad = 16
                r_min = max(0, int(
                    (src_origin_n - max(utm_ys)) / src_pixel
                ) - pad)
                r_max = min(src_h, int(math.ceil(
                    (src_origin_n - min(utm_ys)) / src_pixel
                )) + pad)
                c_min = max(0, int(
                    (min(utm_xs) - src_origin_e) / src_pixel
                ) - pad)
                c_max = min(src_w, int(math.ceil(
                    (max(utm_xs) - src_origin_e) / src_pixel
                )) + pad)

                if r_max <= r_min or c_max <= c_min:
                    continue

                window = np.asarray(
                    src_arr[r_min:r_max, c_min:c_max, :]
                )
                if not window.any():
                    del window
                    continue

                src_data = np.transpose(
                    window.astype(np.float32), (2, 0, 1)
                )
                del window

                win_transform = Affine(
                    src_pixel, 0,
                    src_origin_e + c_min * src_pixel,
                    0, -src_pixel,
                    src_origin_n - r_min * src_pixel,
                )

                dst_data = np.full(
                    (num_bands, strip_h, zone_strip_w),
                    np.nan,
                    dtype=np.float32,
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
                except Exception as exc:
                    if console is not None:
                        console.print(
                            f"      [yellow]strip {strip_start}-"
                            f"{strip_end}: {exc}[/yellow]"
                        )
                    del src_data, dst_data
                    continue

                del src_data
                dst_data = np.nan_to_num(dst_data, nan=0.0)
                dst_data = np.clip(dst_data, 0, 255).astype(np.uint8)
                out = np.transpose(dst_data, (1, 2, 0))
                del dst_data

                mask = out.any(axis=2)
                if not mask.any():
                    del out, mask
                    continue

                tmp_arr[
                    strip_start:strip_end,
                    zone_col_start:zone_col_end,
                    :,
                ] = out

                strips_written += 1
                del out, mask
                gc.collect()

            if console is not None:
                console.print(
                    f"      {strips_written} strips written"
                )

    # ==================================================================
    # Phase 2: Use topozarr to build pyramid from level-0
    # ==================================================================
    if console is not None:
        console.print(
            f"\n  [bold]Phase 2: Building {num_levels}-level pyramid "
            f"with topozarr[/bold]"
        )

    # Open the temporary level-0 as dask-backed xarray
    for preview_name in preview_names:
        level0_zarr = zarr.open_array(
            os.path.join(tmp_store_path, preview_name), mode="r"
        )

        dask_arr = da.from_zarr(level0_zarr, chunks=(512, 512, num_bands))

        lat_coords = north - (np.arange(level0_h) + 0.5) * base_res
        lon_coords = west + (np.arange(level0_w) + 0.5) * base_res

        ds = xr.Dataset(
            {
                preview_name: xr.DataArray(
                    dask_arr,
                    dims=["lat", "lon", "band"],
                    coords={
                        "lat": lat_coords,
                        "lon": lon_coords,
                    },
                ),
            },
        )
        ds = ds.proj.assign_crs(spatial_ref_crs={"EPSG": 4326})

        pyramid = create_pyramid(
            ds,
            levels=num_levels,
            x_dim="lon",
            y_dim="lat",
            method="mean",
            target_chunk_bytes=512 * 512 * num_bands,  # ~1 MB chunks
            target_shard_bytes=None,  # no sharding
        )

        # topozarr sets non-spatial dims to chunk=1, but for 4-band
        # RGBA that means 4x the fetches.  Override encoding + dask
        # chunks to keep band=num_bands.
        for path, var_enc in pyramid.encoding.items():
            for var_name, enc in var_enc.items():
                if "chunks" in enc:
                    chunks = list(enc["chunks"])
                    lvl_ds = pyramid.dt[path].dataset
                    if var_name in lvl_ds:
                        da_var = lvl_ds[var_name]
                        for i, dim in enumerate(da_var.dims):
                            if dim == "band":
                                chunks[i] = int(da_var.sizes["band"])
                        enc["chunks"] = tuple(chunks)
                enc["compressors"] = BloscCodec(cname="zstd", clevel=3)

            # Rechunk DataTree node so dask chunks match encoding
            node = pyramid.dt[path]
            rechunked = node.to_dataset().chunk({"band": num_bands})
            pyramid.dt[path] = rechunked

        if console is not None:
            for idx in range(num_levels):
                lvl_ds = pyramid.dt[f"/{idx}"].dataset
                shape = lvl_ds[preview_name].shape
                console.print(
                    f"  [dim]Level {idx}: "
                    f"{shape[1]}x{shape[0]}x{shape[2]}[/dim]"
                )

        # Patch CRS into multiscales for zarr-layer compatibility
        attrs = dict(pyramid.dt.attrs)
        if "multiscales" in attrs:
            attrs["multiscales"]["crs"] = "EPSG:4326"
            pyramid.dt.attrs = attrs

        if console is not None:
            console.print(
                f"  [dim]Writing pyramid to {output_path}[/dim]"
            )

        pyramid.dt.to_zarr(
            str(output_path),
            mode="w",
            encoding=pyramid.encoding,
            zarr_format=3,
        )

    # Consolidate metadata
    zarr.consolidate_metadata(str(output_path))

    # Re-apply root attributes after consolidation (consolidate_metadata
    # may overwrite the root zarr.json, dropping custom attributes like
    # multiscales that were set on the datatree).
    root = zarr.open_group(str(output_path), mode="r+", zarr_format=3)
    if not root.attrs.get("multiscales") and attrs.get("multiscales"):
        root.attrs.update(attrs)
        logger.info("Re-applied multiscales attributes after consolidation")

    # Clean up temporary store
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    if console is not None:
        console.print(
            f"\n  [bold green]Global store written to {output_path}[/bold green]"
        )
        console.print(
            f"  [dim]{num_levels} levels, "
            f"{len(preview_names)} preview(s): {preview_names}[/dim]"
        )
        console.print(
            f"  [dim]Bounds: lon [{west:.4f}, {east:.4f}], "
            f"lat [{south:.4f}, {north:.4f}][/dim]"
        )
        console.print(
            f"  [dim]Base resolution: {base_res}deg, "
            f"zones: {sorted(zone_infos.keys())}[/dim]"
        )
