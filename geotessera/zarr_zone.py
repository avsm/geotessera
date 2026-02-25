"""Zone-wide Zarr format for consolidated Tessera embeddings.

This module provides tools for building and reading Zarr v3 stores that
consolidate all tiles within a UTM zone into a single store per year.
This enables efficient spatial subsetting and cloud-native access.

Store layout (uncompressed):
    utm{zone:02d}_{year}.zarr/
        embeddings    # int8    (northing, easting, band)  chunks=(1024, 1024, 128)
        scales        # float32 (northing, easting)        chunks=(1024, 1024)

NaN in scales indicates no-data (water or no coverage).
Embeddings are high-entropy quantised values; compression gives negligible
benefit so we store uncompressed.
"""

import logging
import math
import os
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
    include_pca: bool = False,
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
    for name in (["rgb"] if include_rgb else []) + (["pca_rgb"] if include_pca else []):
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
    pca: bool = True,
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
            include_rgb=rgb, include_pca=pca,
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

        if pca:
            pca_basis = compute_pca_basis(store, workers=workers, console=console)
            if console is not None:
                evr = pca_basis["explained_variance_ratio"]
                console.print(
                    f"  PCA explained variance: "
                    f"[{evr[0]:.1%}, {evr[1]:.1%}, {evr[2]:.1%}] "
                    f"(total {evr.sum():.1%})"
                )
            pca_written = write_preview_pass(
                store, "pca_rgb",
                lambda emb, sc: compute_pca_chunk(emb, sc, pca_basis),
                workers=workers, console=console, label="Writing PCA preview",
            )
            store.attrs.update({
                "has_pca_preview": True,
                "pca_explained_variance": pca_basis["explained_variance_ratio"].tolist(),
                "pca_components": pca_basis["components"].tolist(),
                "pca_mean": pca_basis["mean"].tolist(),
                "pca_medians": pca_basis["pc_medians"].tolist(),
                "pca_iqrs": pca_basis["pc_iqrs"].tolist(),
                "pca_stretch": {
                    "min": pca_basis["p_low"].tolist(),
                    "max": pca_basis["p_high"].tolist(),
                },
            })
            if console is not None:
                console.print(f"  [green]PCA preview: {pca_written} chunks written[/green]")

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
# PCA preview
# =============================================================================


def _randomized_svd(
    data: np.ndarray,
    n_components: int,
    n_oversamples: int = 10,
    n_power_iter: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD (Halko-Martinsson-Tropp) for low-rank approximation."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples, n_features = data.shape
    k = n_components + n_oversamples

    omega = rng.standard_normal((n_features, k)).astype(np.float32)
    Y = data @ omega

    for _ in range(n_power_iter):
        Y = data @ (data.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B = Q.T @ data
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :n_components], S[:n_components], Vt[:n_components]


def compute_pca_basis(
    store: "zarr.Group",
    n_components: int = 3,
    max_per_chunk: int = 5000,
    max_total_samples: int = 200_000,
    chunk_sample_fraction: float = 0.25,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
) -> dict:
    """Compute PCA basis from an existing store using parallel reads."""
    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)
    total_chunks = n_rows * n_cols

    # Sample a subset of chunks — PCA on 128 dims converges with ~50-100K samples
    all_chunk_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]
    rng = np.random.default_rng(42)

    n_to_sample = max(4, int(total_chunks * chunk_sample_fraction))
    if n_to_sample < total_chunks:
        sampled = rng.choice(len(all_chunk_indices), n_to_sample, replace=False)
        chunk_indices = [all_chunk_indices[i] for i in sampled]
    else:
        chunk_indices = all_chunk_indices

    results = _run_parallel(
        lambda idx: _sample_chunk_stats(
            emb_arr, scales_arr, idx[0], idx[1],
            chunk_h, chunk_w, emb_shape,
            max_per_chunk=max_per_chunk,
        ),
        chunk_indices, workers, console,
        label=f"Sampling for PCA ({len(chunk_indices)}/{total_chunks} chunks)",
    )

    samples = [r for _, r in results if r is not None]
    if not samples:
        components = np.eye(n_components, N_BANDS, dtype=np.float32)
        return {
            "components": components,
            "mean": np.zeros(N_BANDS, dtype=np.float32),
            "p_low": np.zeros(n_components, dtype=np.float32),
            "p_high": np.ones(n_components, dtype=np.float32),
            "explained_variance_ratio": np.zeros(n_components, dtype=np.float32),
        }

    all_data = np.concatenate(samples, axis=0)

    # Remove rows with NaN or inf (can arise from inf scales values)
    finite_mask = np.isfinite(all_data).all(axis=1)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        all_data = all_data[finite_mask]
        if console is not None:
            console.print(f"  PCA: dropped {n_bad} non-finite samples")
        if all_data.shape[0] == 0:
            components = np.eye(n_components, N_BANDS, dtype=np.float32)
            return {
                "components": components,
                "mean": np.zeros(N_BANDS, dtype=np.float32),
                "p_low": np.zeros(n_components, dtype=np.float32),
                "p_high": np.ones(n_components, dtype=np.float32),
                "explained_variance_ratio": np.zeros(n_components, dtype=np.float32),
            }

    if all_data.shape[0] > max_total_samples:
        idx = rng.choice(all_data.shape[0], max_total_samples, replace=False)
        all_data = all_data[idx]

    if console is not None:
        console.print(
            f"  PCA: {all_data.shape[0]} samples from {len(chunk_indices)} chunks, "
            f"computing randomized SVD..."
        )

    mean = all_data.mean(axis=0)
    centred = all_data - mean

    U, S, Vt = _randomized_svd(centred, n_components, rng=rng)
    components = Vt.astype(np.float32)

    total_var = np.sum(centred ** 2)
    explained = (S ** 2) / total_var if total_var > 0 else np.zeros(n_components)

    # Project samples and equalise each component's spread so all three
    # channels use the full [0, 255] range independently.  Without this,
    # PC1 dominates and the image is biased toward one colour.
    projected = centred @ components.T  # (N, 3)

    # Per-component standardisation: shift to median, scale by IQR
    for i in range(n_components):
        pc = projected[:, i]
        median = np.median(pc)
        q25, q75 = np.percentile(pc, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            projected[:, i] = (pc - median) / iqr

    # Now all components have comparable spread; compute stretch percentiles
    p_low = np.percentile(projected, 1, axis=0).astype(np.float32)
    p_high = np.percentile(projected, 99, axis=0).astype(np.float32)

    for i in range(n_components):
        if p_high[i] <= p_low[i]:
            p_high[i] = p_low[i] + 1.0

    # Store the per-component normalisation params so compute_pca_chunk
    # can reproduce the same transform
    pc_medians = np.array([
        np.median(centred @ components[i]) for i in range(n_components)
    ], dtype=np.float32)
    pc_iqrs = np.array([
        max(np.percentile(centred @ components[i], 75) -
            np.percentile(centred @ components[i], 25), 1e-6)
        for i in range(n_components)
    ], dtype=np.float32)

    return {
        "components": components,
        "mean": mean.astype(np.float32),
        "p_low": p_low,
        "p_high": p_high,
        "pc_medians": pc_medians,
        "pc_iqrs": pc_iqrs,
        "explained_variance_ratio": explained.astype(np.float32),
    }


def compute_pca_chunk(
    embedding_int8: np.ndarray,
    scales: np.ndarray,
    pca_basis: dict,
) -> np.ndarray:
    """Compute an RGBA uint8 PCA preview from embedding + scales."""
    h, w = scales.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = np.isfinite(scales) & (scales != 0)
    if not np.any(valid):
        return rgba

    components = pca_basis["components"]
    mean = pca_basis["mean"]
    p_low = pca_basis["p_low"]
    p_high = pca_basis["p_high"]
    pc_medians = pca_basis["pc_medians"]
    pc_iqrs = pca_basis["pc_iqrs"]

    scales_safe = np.where(valid, scales, 0.0)
    float_emb = embedding_int8.astype(np.float32) * scales_safe[:, :, np.newaxis]

    flat = float_emb.reshape(-1, N_BANDS)
    valid_flat = valid.ravel()

    # Project and apply per-component normalisation (same as basis computation)
    projected = (flat - mean) @ components.T
    for i in range(3):
        projected[:, i] = (projected[:, i] - pc_medians[i]) / pc_iqrs[i]

    # Stretch to [0, 1] using the percentiles computed on normalised data
    for i in range(3):
        projected[:, i] = (projected[:, i] - p_low[i]) / (p_high[i] - p_low[i])

    projected = np.clip(projected, 0, 1) * 255
    rgb_flat = projected.astype(np.uint8)

    rgba_flat = rgba.reshape(-1, 4)
    rgba_flat[:, :3] = rgb_flat[:, :3]
    rgba_flat[~valid_flat, :3] = 0
    rgba_flat[:, 3] = np.where(valid_flat, 255, 0).astype(np.uint8)

    return rgba


# =============================================================================
# Generic preview pass (used for both RGB and PCA)
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
        array_name: Name of the output array (e.g. "rgb" or "pca_rgb").
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
# Standalone preview commands (--rgb-only, --pca-only)
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


def add_pca_to_existing_store(
    store_path: Path,
    workers: Optional[int] = None,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add PCA RGB preview array to an existing Zarr store."""
    import zarr

    store = zarr.open_group(str(store_path), mode="r+")

    try:
        _ = store["pca_rgb"]
    except KeyError:
        emb_shape = store["embeddings"].shape
        store.create_array(
            "pca_rgb",
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
        console.print(f"  Pass 1: Computing PCA basis ({workers} threads)...")

    pca_basis = compute_pca_basis(store, workers=workers, console=console)
    if console is not None:
        evr = pca_basis["explained_variance_ratio"]
        console.print(
            f"  PCA explained variance: "
            f"[{evr[0]:.1%}, {evr[1]:.1%}, {evr[2]:.1%}] "
            f"(total {evr.sum():.1%})"
        )
        console.print("  Pass 2: Writing PCA preview...")

    written = write_preview_pass(
        store, "pca_rgb",
        lambda emb, sc: compute_pca_chunk(emb, sc, pca_basis),
        workers=workers, console=console, label="Writing PCA preview",
    )

    store.attrs.update({
        "has_pca_preview": True,
        "pca_explained_variance": pca_basis["explained_variance_ratio"].tolist(),
        "pca_components": pca_basis["components"].tolist(),
        "pca_mean": pca_basis["mean"].tolist(),
        "pca_medians": pca_basis["pc_medians"].tolist(),
        "pca_iqrs": pca_basis["pc_iqrs"].tolist(),
        "pca_stretch": {
            "min": pca_basis["p_low"].tolist(),
            "max": pca_basis["p_high"].tolist(),
        },
    })

    if console is not None:
        console.print(f"  [green]PCA preview: {written} chunks written[/green]")


# =============================================================================
# Preview pyramids (multi-resolution overviews)
# =============================================================================

PYRAMID_LEVELS = 8  # level 0 = full-res, levels 1..7 = 2x coarser each


def build_preview_pyramid(
    store: "zarr.Group",
    preview_name: str,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Build a multi-resolution pyramid from an existing preview array.

    Reads ``store[preview_name]`` (full-res RGBA uint8) and creates
    iteratively coarsened copies at ``{preview_name}_pyramid/{level}``
    for levels 1 through PYRAMID_LEVELS-1.

    Each level halves both spatial dimensions using mean downsampling.
    Per-level attrs record the pixel size, transform, and shape so
    readers can pick the appropriate resolution.

    Args:
        store: Zarr group opened in ``r+`` mode.
        preview_name: Name of the source array (``"rgb"`` or ``"pca_rgb"``).
        console: Optional Rich Console for progress display.

    Returns:
        Number of pyramid levels written (0 if the source array is missing).
    """
    import xarray as xr

    # --- Validate source array exists ---
    try:
        src_arr = store[preview_name]
    except KeyError:
        logger.warning(
            "build_preview_pyramid: source array %r not found in store",
            preview_name,
        )
        return 0

    # --- Read full-res data and store transform ---
    full_data = np.asarray(src_arr[:])
    attrs = dict(store.attrs)
    transform = list(attrs["transform"])
    base_pixel_size = transform[0]
    origin_e = transform[2]
    origin_n = transform[5]

    # --- Delete existing pyramid group if present ---
    pyramid_name = f"{preview_name}_pyramid"
    if pyramid_name in store:
        del store[pyramid_name]

    pyramid_group = store.create_group(pyramid_name)

    # --- Set up progress bar if console available ---
    progress_ctx = None
    progress_obj = None
    progress_task = None
    num_levels = PYRAMID_LEVELS - 1  # levels 1..7

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )
        progress_obj = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        )
        progress_ctx = progress_obj.__enter__()
        progress_task = progress_obj.add_task(
            f"Building {preview_name} pyramid", total=num_levels,
        )

    # --- Iteratively coarsen ---
    current = xr.DataArray(
        full_data,
        dims=["northing", "easting", "rgba"],
    )
    levels_written = 0

    try:
        for level in range(1, PYRAMID_LEVELS):
            h, w = current.shape[0], current.shape[1]

            # Need at least 2 pixels in each spatial dim to coarsen
            if h < 2 or w < 2:
                logger.info(
                    "build_preview_pyramid: stopping at level %d "
                    "(array too small: %dx%d)",
                    level, h, w,
                )
                break

            current = (
                current
                .coarsen(northing=2, easting=2, boundary="trim")
                .mean()
                .astype(np.uint8)
            )

            ch, cw = current.shape[0], current.shape[1]
            level_pixel_size = base_pixel_size * (2 ** level)

            # Write the coarsened array
            arr = pyramid_group.create_array(
                str(level),
                shape=(ch, cw, 4),
                chunks=(min(1024, ch), min(1024, cw), 4),
                dtype=np.uint8,
                fill_value=np.uint8(0),
                compressors=None,
                dimension_names=["northing", "easting", "rgba"],
            )
            arr[:] = current.values

            # Per-level metadata
            level_transform = [
                level_pixel_size, 0.0, origin_e,
                0.0, -level_pixel_size, origin_n,
            ]
            arr.attrs.update({
                "level": level,
                "pixel_size_m": level_pixel_size,
                "transform": level_transform,
                "shape": [ch, cw, 4],
            })

            levels_written += 1

            if progress_obj is not None:
                progress_obj.advance(progress_task)
    finally:
        if progress_ctx is not None:
            progress_obj.__exit__(None, None, None)

    # Store summary attrs on the pyramid group
    pyramid_group.attrs.update({
        "source": preview_name,
        "num_levels": levels_written,
        "base_pixel_size_m": base_pixel_size,
    })

    if console is not None:
        console.print(
            f"  [green]{preview_name} pyramid: "
            f"{levels_written} levels written[/green]"
        )

    return levels_written


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
