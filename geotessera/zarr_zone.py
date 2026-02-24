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

# Number of embedding bands in Tessera
N_BANDS = 128

# Bands used for the RGB preview array (indices into the 128-band embedding)
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


def _sample_chunk_stats(
    emb_arr,
    scales_arr,
    ci: int,
    cj: int,
    chunk_h: int,
    chunk_w: int,
    emb_shape: tuple,
    max_per_chunk: int = 10_000,
) -> Optional[np.ndarray]:
    """Read one chunk and return a subsample of dequantised RGB values.

    Runs in a thread — zarr I/O releases the GIL.

    Returns:
        (N, 3) float32 array of dequantised samples, or None if chunk empty.
    """
    r0 = ci * chunk_h
    r1 = min(r0 + chunk_h, emb_shape[0])
    c0 = cj * chunk_w
    c1 = min(c0 + chunk_w, emb_shape[1])

    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
    valid = ~np.isnan(scales_chunk) & (scales_chunk != 0)
    if not np.any(valid):
        return None

    # Only read the 3 RGB bands (not all 128)
    emb_chunk = np.asarray(
        emb_arr[r0:r1, c0:c1, RGB_PREVIEW_BANDS[0] : RGB_PREVIEW_BANDS[-1] + 1]
    )

    # Dequantise: int8 * scale for each of 3 bands
    scales_valid = scales_chunk[valid]
    rgb_valid = emb_chunk[valid].astype(np.float32) * scales_valid[:, np.newaxis]

    # Subsample if too many valid pixels
    if rgb_valid.shape[0] > max_per_chunk:
        rng = np.random.default_rng(ci * 10007 + cj)
        idx = rng.choice(rgb_valid.shape[0], max_per_chunk, replace=False)
        rgb_valid = rgb_valid[idx]

    return rgb_valid


def compute_stretch_from_store(
    store: "zarr.Group",
    p_low: float = 2,
    p_high: float = 98,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
) -> dict:
    """Compute percentile stretch from an existing store using parallel reads.

    Reads chunks in parallel via threads (zarr I/O releases the GIL),
    collects subsampled dequantised values, then computes global percentiles.

    Args:
        store: Zarr group with embeddings and scales arrays
        p_low: Low percentile (default 2)
        p_high: High percentile (default 98)
        workers: Number of threads (default 8)
        console: Optional Rich Console for progress

    Returns:
        Dict with 'min' and 'max' lists of 3 floats each.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)
    total_chunks = n_rows * n_cols

    samples = []

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Computing stretch ({workers} threads)", total=total_chunks,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _sample_chunk_stats,
                        emb_arr, scales_arr, ci, cj,
                        chunk_h, chunk_w, emb_shape,
                    ): (ci, cj)
                    for ci in range(n_rows)
                    for cj in range(n_cols)
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        samples.append(result)
                    progress.advance(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _sample_chunk_stats,
                    emb_arr, scales_arr, ci, cj,
                    chunk_h, chunk_w, emb_shape,
                ): (ci, cj)
                for ci in range(n_rows)
                for cj in range(n_cols)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    samples.append(result)

    if not samples:
        return {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}

    all_rgb = np.concatenate(samples, axis=0)  # (N, 3)
    stretch_min = [float(np.percentile(all_rgb[:, i], p_low)) for i in range(3)]
    stretch_max = [float(np.percentile(all_rgb[:, i], p_high)) for i in range(3)]

    # Ensure non-zero range
    for i in range(3):
        if stretch_max[i] <= stretch_min[i]:
            stretch_max[i] = stretch_min[i] + 1.0

    return {"min": stretch_min, "max": stretch_max}


def compute_rgb_chunk(
    embedding_int8: np.ndarray,
    scales: np.ndarray,
    band_indices: tuple,
    stretch_min: List[float],
    stretch_max: List[float],
) -> np.ndarray:
    """Compute an RGBA uint8 preview from embedding + scales.

    Args:
        embedding_int8: int8 array (H, W, 128) or (H, W, N_BANDS)
        scales: float32 array (H, W)
        band_indices: Tuple of 3 band indices for R, G, B
        stretch_min: Per-band minimum for normalisation (len 3)
        stretch_max: Per-band maximum for normalisation (len 3)

    Returns:
        uint8 RGBA array of shape (H, W, 4)
    """
    h, w = scales.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = ~np.isnan(scales) & (scales != 0)

    scales_safe = np.where(valid, scales, 0.0)

    for i, band_idx in enumerate(band_indices):
        raw = embedding_int8[:, :, band_idx].astype(np.float32)
        dequant = raw * scales_safe
        lo, hi = stretch_min[i], stretch_max[i]
        normalised = (dequant - lo) / (hi - lo)
        rgba[:, :, i] = np.clip(normalised * 255, 0, 255).astype(np.uint8)

    # Zero out RGB channels for invalid pixels, set alpha
    inv = ~valid
    rgba[inv, 0] = 0
    rgba[inv, 1] = 0
    rgba[inv, 2] = 0
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


# =============================================================================
# UTM helpers
# =============================================================================


def epsg_to_utm_zone(epsg: int) -> int:
    """Extract UTM zone number from an EPSG code.

    Args:
        epsg: EPSG code (326xx for north, 327xx for south)

    Returns:
        UTM zone number (1-60)

    Raises:
        ValueError: If EPSG code is not a UTM zone
    """
    if 32601 <= epsg <= 32660:
        return epsg - 32600
    if 32701 <= epsg <= 32760:
        return epsg - 32700
    raise ValueError(f"EPSG {epsg} is not a UTM zone (expected 326xx or 327xx)")


def epsg_is_south(epsg: int) -> bool:
    """Check if an EPSG code is a southern hemisphere UTM zone.

    Args:
        epsg: EPSG code

    Returns:
        True if southern hemisphere (327xx)
    """
    return 32701 <= epsg <= 32760


def zone_canonical_epsg(zone: int) -> int:
    """Get the canonical (northern hemisphere) EPSG code for a UTM zone.

    Args:
        zone: UTM zone number (1-60)

    Returns:
        EPSG code (326xx)
    """
    return 32600 + zone


def northing_to_canonical(northing: float, epsg: int) -> float:
    """Convert a northing value to canonical (northern hemisphere) coordinates.

    Southern hemisphere tiles use a false northing of 10,000,000m.
    We subtract this to get negative northings for a continuous axis.

    Args:
        northing: Northing in the tile's native CRS
        epsg: EPSG code of the tile

    Returns:
        Canonical northing (may be negative for southern hemisphere)
    """
    if epsg_is_south(epsg):
        return northing - 10_000_000.0
    return northing


# =============================================================================
# Grid computation
# =============================================================================


def _snap_to_grid(value: float, pixel_size: float, snap_floor: bool) -> float:
    """Snap a coordinate to the pixel grid.

    Args:
        value: Coordinate value in metres
        pixel_size: Pixel size in metres
        snap_floor: If True, snap down (for origin easting/min).
                    If False, snap up (for max extent).

    Returns:
        Snapped coordinate
    """
    if snap_floor:
        return math.floor(value / pixel_size) * pixel_size
    else:
        return math.ceil(value / pixel_size) * pixel_size


def compute_zone_grid(tile_infos: List[TileInfo], year: int) -> ZoneGrid:
    """Compute the zone-wide grid that encompasses all tiles.

    Args:
        tile_infos: List of TileInfo for tiles in this zone
        year: Year of embeddings

    Returns:
        ZoneGrid describing the zone store dimensions

    Raises:
        ValueError: If tile_infos is empty or tiles span multiple zones
    """
    if not tile_infos:
        raise ValueError("No tiles provided")

    # All tiles should be in the same zone
    zone = epsg_to_utm_zone(tile_infos[0].epsg)
    pixel_size = 10.0

    # Compute bounding box in canonical coordinates
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

        # Tile origin (top-left corner) in native CRS
        tile_easting = ti.transform.c  # xoff
        tile_northing = ti.transform.f  # yoff (top edge)

        # Convert to canonical northing
        tile_northing_canon = northing_to_canonical(tile_northing, ti.epsg)

        # Tile extent
        tile_right = tile_easting + ti.width * pixel_size
        tile_bottom = tile_northing_canon - ti.height * pixel_size

        min_easting = min(min_easting, tile_easting)
        max_easting = max(max_easting, tile_right)
        min_northing = min(min_northing, tile_bottom)
        max_northing = max(max_northing, tile_northing_canon)

    # Snap to 10m grid
    origin_easting = _snap_to_grid(min_easting, pixel_size, snap_floor=True)
    origin_northing = _snap_to_grid(max_northing, pixel_size, snap_floor=False)
    extent_right = _snap_to_grid(max_easting, pixel_size, snap_floor=False)
    extent_bottom = _snap_to_grid(min_northing, pixel_size, snap_floor=True)

    width_px = round((extent_right - origin_easting) / pixel_size)
    height_px = round((origin_northing - extent_bottom) / pixel_size)

    return ZoneGrid(
        zone=zone,
        year=year,
        canonical_epsg=zone_canonical_epsg(zone),
        origin_easting=origin_easting,
        origin_northing=origin_northing,
        width_px=width_px,
        height_px=height_px,
        pixel_size=pixel_size,
        tiles=tile_infos,
    )


def tile_pixel_offset(
    tile_info: TileInfo, zone_grid: ZoneGrid
) -> Tuple[int, int]:
    """Compute the pixel offset of a tile within the zone grid.

    Args:
        tile_info: Tile metadata
        zone_grid: Zone grid specification

    Returns:
        (row_start, col_start) pixel offsets
    """
    tile_easting = tile_info.transform.c
    tile_northing = northing_to_canonical(tile_info.transform.f, tile_info.epsg)

    col_start = round((tile_easting - zone_grid.origin_easting) / zone_grid.pixel_size)
    row_start = round(
        (zone_grid.origin_northing - tile_northing) / zone_grid.pixel_size
    )

    return row_start, col_start


# =============================================================================
# Store creation & writing
# =============================================================================


def _store_name(zone: int, year: int) -> str:
    """Generate Zarr store directory name.

    Args:
        zone: UTM zone number
        year: Year of embeddings

    Returns:
        Store name like "utm30_2024.zarr"
    """
    return f"utm{zone:02d}_{year}.zarr"


def create_zone_store(
    zone_grid: ZoneGrid,
    output_dir: Path,
    geotessera_version: str = "unknown",
    dataset_version: str = "v1",
    include_rgb: bool = False,
    include_pca: bool = False,
) -> "zarr.Group":
    """Create a new Zarr store for a UTM zone.

    Args:
        zone_grid: Zone grid specification
        output_dir: Directory to create the store in
        geotessera_version: Version of geotessera
        dataset_version: Tessera dataset version
        include_rgb: If True, create an additional RGB preview array
        include_pca: If True, create a PCA RGB preview array

    Returns:
        zarr.Group root of the created store
    """
    import zarr


    store_path = output_dir / _store_name(zone_grid.zone, zone_grid.year)

    # Remove existing store if present
    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)

    store = zarr.open_group(store_path, mode="w", zarr_format=3)

    # Create embeddings array: int8 (northing, easting, band)
    store.create_array(
        "embeddings",
        shape=(zone_grid.height_px, zone_grid.width_px, N_BANDS),
        chunks=(1024, 1024, N_BANDS),
        dtype=np.int8,
        fill_value=np.int8(0),
        compressors=None,
        dimension_names=["northing", "easting", "band"],
    )

    # Create scales array: float32 (northing, easting)
    store.create_array(
        "scales",
        shape=(zone_grid.height_px, zone_grid.width_px),
        chunks=(1024, 1024),
        dtype=np.float32,
        fill_value=np.float32("nan"),
        compressors=None,
        dimension_names=["northing", "easting"],
    )

    # Create RGB preview array if requested
    if include_rgb:
        store.create_array(
            "rgb",
            shape=(zone_grid.height_px, zone_grid.width_px, 4),
            chunks=(1024, 1024, 4),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=None,
            dimension_names=["northing", "easting", "rgba"],
        )

    # Create PCA RGB preview array if requested
    if include_pca:
        store.create_array(
            "pca_rgb",
            shape=(zone_grid.height_px, zone_grid.width_px, 4),
            chunks=(1024, 1024, 4),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=None,
            dimension_names=["northing", "easting", "rgba"],
        )

    # Create coordinate arrays
    easting_coords = (
        zone_grid.origin_easting
        + (np.arange(zone_grid.width_px) + 0.5) * zone_grid.pixel_size
    )
    northing_coords = (
        zone_grid.origin_northing
        - (np.arange(zone_grid.height_px) + 0.5) * zone_grid.pixel_size
    )
    band_coords = np.arange(N_BANDS, dtype=np.int32)

    store.create_array(
        "easting",
        shape=(zone_grid.width_px,),
        dtype=np.float64,
        fill_value=0.0,
        compressors=None,
        dimension_names=["easting"],
    )
    store["easting"][:] = easting_coords

    store.create_array(
        "northing",
        shape=(zone_grid.height_px,),
        dtype=np.float64,
        fill_value=0.0,
        compressors=None,
        dimension_names=["northing"],
    )
    store["northing"][:] = northing_coords

    store.create_array(
        "band",
        shape=(N_BANDS,),
        dtype=np.int32,
        fill_value=0,
        compressors=None,
        dimension_names=["band"],
    )
    store["band"][:] = band_coords

    # CRS WKT
    try:
        from pyproj import CRS

        crs = CRS.from_epsg(zone_grid.canonical_epsg)
        crs_wkt = crs.to_wkt()
    except ImportError:
        crs_wkt = ""

    # Store-level attributes
    store.attrs.update(
        {
            "utm_zone": zone_grid.zone,
            "year": zone_grid.year,
            "crs_epsg": zone_grid.canonical_epsg,
            "crs_wkt": crs_wkt,
            "transform": [
                zone_grid.pixel_size,
                0.0,
                zone_grid.origin_easting,
                0.0,
                -zone_grid.pixel_size,
                zone_grid.origin_northing,
            ],
            "pixel_size_m": zone_grid.pixel_size,
            "geotessera_version": geotessera_version,
            "tessera_dataset_version": dataset_version,
            "n_tiles": len(zone_grid.tiles),
        }
    )

    return store


def write_tile_to_store(
    store: "zarr.Group",
    embedding_int8: np.ndarray,
    scales: np.ndarray,
    row_start: int,
    col_start: int,
) -> None:
    """Write a single tile's data into the zone store.

    Args:
        store: Zarr group (root of zone store)
        embedding_int8: int8 array of shape (H, W, 128)
        scales: float32 array of shape (H, W) with NaN for no-data
        row_start: Starting row in the zone grid
        col_start: Starting column in the zone grid
    """
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
    """Apply landmask to scales array, setting water pixels to NaN.

    Args:
        scales: float32 scales array of shape (H, W) or (H, W, 128)
        landmask_path: Path to the landmask TIFF file

    Returns:
        Modified scales array with NaN where landmask indicates water.
        If input is 3D (H, W, 128), it is reduced to 2D (H, W) by taking
        max across bands.
    """
    import rasterio

    # Reduce 3D scales to 2D if needed
    if scales.ndim == 3:
        scales = np.nanmax(scales, axis=2)

    # Ensure float32 for NaN support
    scales = scales.astype(np.float32, copy=True)

    with rasterio.open(landmask_path) as src:
        landmask = src.read(1)

    # In landmask TIFFs, 0 = water, non-zero = land
    water_mask = landmask == 0

    # Handle shape mismatches (shouldn't happen but be safe)
    if water_mask.shape != scales.shape:
        logger.warning(
            f"Landmask shape {water_mask.shape} != scales shape {scales.shape}, "
            f"skipping landmask for {landmask_path}"
        )
        return scales

    scales[water_mask] = np.float32("nan")
    return scales


# =============================================================================
# Tile info gathering
# =============================================================================


def gather_tile_infos(
    registry: "Registry",
    year: int,
    zones: Optional[List[int]] = None,
    console: Optional["rich.console.Console"] = None,
    workers: Optional[int] = None,
) -> Dict[int, List[TileInfo]]:
    """Gather tile metadata from local files and group by UTM zone.

    Iterates all available tiles for the given year in the registry,
    checks that the corresponding files exist locally, reads CRS info
    from each landmask (in parallel), and groups tiles by UTM zone.

    Args:
        registry: Registry instance for tile/landmask access
        year: Year of embeddings
        zones: Optional list of zone numbers to include. If None, all zones.
        console: Optional Rich Console for progress display
        workers: Number of parallel workers for scanning landmasks.
                 Defaults to min(cpu_count, 8). Set to 1 to disable.

    Returns:
        Dict mapping zone number to list of TileInfo
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .registry import (
        EMBEDDINGS_DIR_NAME,
        LANDMASKS_DIR_NAME,
        tile_to_embedding_paths,
        tile_to_landmask_filename,
    )

    if workers is None:
        workers = min(multiprocessing.cpu_count(), 8)

    # Get tiles for this year directly from MultiIndex (avoids iterating all years)
    gdf = registry._registry_gdf
    try:
        year_slice = gdf.loc[year]  # selects all (lon_i, lat_i) for this year
        tiles = [
            (year, lon_i / 100.0, lat_i / 100.0)
            for lon_i, lat_i in year_slice.index.unique()
        ]
    except KeyError:
        tiles = []

    if console is not None:
        console.print(f"  Found {len(tiles):,} tiles for year {year}")

    zone_set = set(zones) if zones is not None else None

    # Pre-filter by UTM zone (zone is deterministic from longitude)
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

    # Pre-compute file paths and filter to tiles that exist on disk
    scan_args = []
    base_emb = str(registry._embeddings_dir / EMBEDDINGS_DIR_NAME)
    base_lm = str(registry._embeddings_dir / LANDMASKS_DIR_NAME)
    n_missing = 0

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Checking files", total=len(tiles))
            for tile_year, tile_lon, tile_lat in tiles:
                emb_rel, scales_rel = tile_to_embedding_paths(
                    tile_lon, tile_lat, tile_year
                )
                emb_path = os.path.join(base_emb, emb_rel)
                scales_path = os.path.join(base_emb, scales_rel)
                landmask_path = os.path.join(
                    base_lm, tile_to_landmask_filename(tile_lon, tile_lat)
                )
                if (
                    os.path.exists(emb_path)
                    and os.path.exists(scales_path)
                    and os.path.exists(landmask_path)
                ):
                    scan_args.append(
                        (tile_year, tile_lon, tile_lat, emb_path, scales_path, landmask_path)
                    )
                else:
                    n_missing += 1
                progress.advance(task)
        console.print(
            f"  {len(scan_args):,} tiles with files on disk"
            + (f" ({n_missing:,} missing)" if n_missing else "")
        )
    else:
        for tile_year, tile_lon, tile_lat in tiles:
            emb_rel, scales_rel = tile_to_embedding_paths(
                tile_lon, tile_lat, tile_year
            )
            emb_path = os.path.join(base_emb, emb_rel)
            scales_path = os.path.join(base_emb, scales_rel)
            landmask_path = os.path.join(
                base_lm, tile_to_landmask_filename(tile_lon, tile_lat)
            )
            if (
                os.path.exists(emb_path)
                and os.path.exists(scales_path)
                and os.path.exists(landmask_path)
            ):
                scan_args.append(
                    (tile_year, tile_lon, tile_lat, emb_path, scales_path, landmask_path)
                )
            else:
                n_missing += 1

    zones_dict: Dict[int, List[TileInfo]] = {}
    skipped = 0
    all_warnings: List[str] = []

    def _handle_result(result):
        nonlocal skipped
        if result is None:
            skipped += 1
            return
        ti, warnings = _tile_info_from_scan_result(result)
        all_warnings.extend(warnings)
        zone = epsg_to_utm_zone(ti.epsg)
        if zone_set is not None and zone not in zone_set:
            skipped += 1
            return
        zones_dict.setdefault(zone, []).append(ti)
        return ti, zone

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Scanning tiles", total=len(scan_args),
                status=f"{len(scan_args)} tiles, {workers} workers",
            )

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_scan_one_tile, args): args
                    for args in scan_args
                }
                for future in as_completed(futures):
                    ret = _handle_result(future.result())
                    progress.advance(task)
                    if ret is not None:
                        ti, zone = ret
                        progress.update(
                            task,
                            status=f"({ti.lon:.2f}, {ti.lat:.2f}) zone {zone}",
                        )

            progress.update(task, status="Done")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_scan_one_tile, args): args
                for args in scan_args
            }
            for future in as_completed(futures):
                _handle_result(future.result())

    if console is not None:
        total_matched = sum(len(t) for t in zones_dict.values())
        zone_summary = ", ".join(
            f"zone {z}: {len(t)}" for z, t in sorted(zones_dict.items())
        )
        console.print(f"  {total_matched} tiles in {len(zones_dict)} zone(s): {zone_summary}")
        total_skipped = skipped + n_missing
        if total_skipped:
            console.print(f"  [dim]{total_skipped} tiles skipped (missing files or non-UTM)[/dim]")
        if all_warnings:
            console.print(
                f"  [yellow]{len(all_warnings)} grid mismatch warning(s) "
                f"(computed vs landmask):[/yellow]"
            )
            for w in all_warnings:
                console.print(f"    [yellow]{w}[/yellow]")
    else:
        for w in all_warnings:
            logger.warning(w)

    return zones_dict


def compute_tile_grid(lon: float, lat: float, pixel_size: float = 10.0):
    """Compute expected UTM EPSG, transform, and pixel dimensions for a tile.

    Derives the UTM zone from longitude, projects the tile's WGS84 corners
    to UTM, and returns the expected grid parameters.  The origin is the raw
    projected upper-left corner (no snap to pixel grid) and dimensions use
    ``round()`` — matching how the landmask TIFFs were generated.

    Args:
        lon: Tile centre longitude (on 0.05-degree grid)
        lat: Tile centre latitude (on 0.05-degree grid)
        pixel_size: Pixel size in metres (default 10)

    Returns:
        Tuple of (epsg, transform_tuple, height, width) where
        transform_tuple is (a, b, c, d, e, f) for an Affine transform.
    """
    import math
    from pyproj import Transformer

    # Compute UTM zone from longitude
    zone = int(math.floor((lon + 180) / 6)) + 1
    zone = max(1, min(60, zone))
    is_south = lat < 0
    epsg = 32700 + zone if is_south else 32600 + zone

    # Tile corners in WGS84 (0.1-degree tile centred on lon, lat)
    west = lon - 0.05
    east = lon + 0.05
    south = lat - 0.05
    north = lat + 0.05

    # Project corners to UTM
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    ul_e, ul_n = transformer.transform(west, north)  # upper-left
    ur_e, ur_n = transformer.transform(east, north)  # upper-right
    ll_e, ll_n = transformer.transform(west, south)  # lower-left
    lr_e, lr_n = transformer.transform(east, south)  # lower-right

    # Origin is the raw projected upper-left corner (west-most easting,
    # north-most northing) — no snap to pixel grid.  This matches how
    # the landmask TIFFs were generated by rasterio/GDAL.
    origin_e = min(ul_e, ll_e)
    origin_n = max(ul_n, ur_n)

    # Extent from opposite corners
    max_e = max(ur_e, lr_e)
    min_n = min(ll_n, lr_n)

    # Dimensions use round() to match the landmask generation
    width = round((max_e - origin_e) / pixel_size)
    height = round((origin_n - min_n) / pixel_size)

    transform_tuple = (pixel_size, 0.0, origin_e, 0.0, -pixel_size, origin_n)
    return epsg, transform_tuple, height, width


def _lon_to_utm_zone(lon: float) -> int:
    """Compute the UTM zone number for a longitude.

    Args:
        lon: Longitude in decimal degrees

    Returns:
        UTM zone number (1-60)
    """
    zone = int(math.floor((lon + 180) / 6)) + 1
    return max(1, min(60, zone))


def _scan_one_tile(args):
    """Scan a single tile's landmask for CRS info. Runs in a worker process.

    Also computes the expected grid from the tile's lon/lat and warns if
    the landmask disagrees, and checks whether the tile straddles a UTM
    zone boundary.

    Args:
        args: Tuple of (year, lon, lat, emb_path, scales_path, landmask_path)

    Returns:
        Tuple of (year, lon, lat, epsg, transform_tuple, height, width,
                  landmask_path, emb_path, scales_path, warnings_list)
        or None if invalid.
        The affine transform is serialised as a 6-element tuple so it can
        cross the process boundary without pickling rasterio objects.
    """
    import rasterio

    tile_year, tile_lon, tile_lat, emb_path, scales_path, landmask_path = args

    try:
        with rasterio.open(landmask_path) as src:
            epsg = src.crs.to_epsg()
            t = src.transform
            height = src.height
            width = src.width
    except Exception:
        return None

    if epsg is None:
        return None

    try:
        epsg_to_utm_zone(epsg)
    except ValueError:
        return None

    if abs(t.a - 10.0) > 0.01 or abs(t.e - (-10.0)) > 0.01:
        return None

    # Compare with computed grid
    warnings = []

    # Check if tile straddles a UTM zone boundary
    west_lon = tile_lon - 0.05
    east_lon = tile_lon + 0.05
    west_zone = _lon_to_utm_zone(west_lon)
    east_zone = _lon_to_utm_zone(east_lon)
    landmask_zone = epsg_to_utm_zone(epsg)

    if west_zone != east_zone:
        warnings.append(
            f"({tile_lon:.2f}, {tile_lat:.2f}): "
            f"straddles zone boundary: west edge zone {west_zone}, "
            f"east edge zone {east_zone}, landmask zone {landmask_zone}"
        )

    try:
        comp_epsg, comp_tf, comp_h, comp_w = compute_tile_grid(tile_lon, tile_lat)

        if comp_epsg != epsg:
            warnings.append(
                f"({tile_lon:.2f}, {tile_lat:.2f}): "
                f"EPSG mismatch: computed {comp_epsg}, landmask {epsg}"
            )
        if comp_h != height or comp_w != width:
            warnings.append(
                f"({tile_lon:.2f}, {tile_lat:.2f}): "
                f"size mismatch: computed {comp_w}x{comp_h}, "
                f"landmask {width}x{height}"
            )
        # Check transform origin (easting, northing)
        if abs(comp_tf[2] - t.c) > 0.5 or abs(comp_tf[5] - t.f) > 0.5:
            warnings.append(
                f"({tile_lon:.2f}, {tile_lat:.2f}): "
                f"origin mismatch: computed ({comp_tf[2]:.1f}, {comp_tf[5]:.1f}), "
                f"landmask ({t.c:.1f}, {t.f:.1f})"
            )
    except Exception as e:
        warnings.append(
            f"({tile_lon:.2f}, {tile_lat:.2f}): "
            f"grid computation failed: {e}"
        )

    # Serialise the affine transform as a plain tuple
    return (
        tile_year, tile_lon, tile_lat, epsg,
        (t.a, t.b, t.c, t.d, t.e, t.f),
        height, width, landmask_path, emb_path, scales_path,
        warnings,
    )


def _tile_info_from_scan_result(result):
    """Reconstruct a TileInfo and warnings from the serialised scan result.

    Returns:
        Tuple of (TileInfo, list_of_warning_strings)
    """
    from rasterio.transform import Affine

    (
        tile_year, tile_lon, tile_lat, epsg,
        transform_tuple, height, width,
        landmask_path, emb_path, scales_path,
        warnings,
    ) = result

    ti = TileInfo(
        lon=tile_lon,
        lat=tile_lat,
        year=tile_year,
        epsg=epsg,
        transform=Affine(*transform_tuple),
        height=height,
        width=width,
        landmask_path=landmask_path,
        embedding_path=emb_path,
        scales_path=scales_path,
    )
    return ti, warnings


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
    one Zarr store per zone. Use ``zones`` to restrict to specific zones
    (e.g. ``[30]`` to test a single zone).

    Args:
        registry: Registry instance pointing at local tile data
        output_dir: Directory to write Zarr stores to
        year: Year of embeddings
        zones: Optional list of zone numbers to include (default: all)
        dry_run: If True, only compute zone breakdown without writing
        geotessera_version: Version string for metadata
        dataset_version: Tessera dataset version
        console: Optional Rich Console for progress display
        rgb: If True, generate RGB preview arrays (default: True)
        pca: If True, generate PCA RGB preview arrays (default: True)
        workers: Number of threads for parallel I/O (default: cpu_count, max 16)

    Returns:
        List of paths to created Zarr stores
    """
    if workers is None:
        workers = _default_workers()

    output_dir = Path(output_dir)

    # Gather and group tiles
    zones_dict = gather_tile_infos(
        registry,
        year,
        zones=zones,
        console=console,
        workers=workers,
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

    sorted_zones = sorted(zones_dict.items())
    total_tiles = sum(len(t) for t in zones_dict.values())

    for zone_idx, (zone_num, tile_infos) in enumerate(sorted_zones):
        # Compute grid
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

        # Create store
        store = create_zone_store(
            zone_grid,
            output_dir,
            geotessera_version=geotessera_version,
            dataset_version=dataset_version,
            include_rgb=rgb,
            include_pca=pca,
        )

        # Two-phase write: (1) read all tiles in parallel (I/O bound,
        # uses all workers), (2) group by chunk and write each chunk once
        # to avoid read-modify-write amplification on the zarr store.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # --- Phase 1: parallel tile reads ---
        tile_data: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]] = {}
        read_errors = 0

        if console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TextColumn("[dim]{task.fields[status]}", justify="left"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Reading tiles ({workers} threads)",
                    total=len(tile_infos),
                    status="starting...",
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_read_single_tile, ti, zone_grid): ti
                        for ti in tile_infos
                    }
                    for future in as_completed(futures):
                        ti = futures[future]
                        try:
                            tile_data[id(ti)] = future.result()
                        except Exception as e:
                            logger.warning(
                                f"Failed to read tile ({ti.lon}, {ti.lat}): {e}"
                            )
                            read_errors += 1
                        progress.update(
                            task, status=f"({ti.lon:.2f}, {ti.lat:.2f})",
                        )
                        progress.advance(task)
                if read_errors:
                    progress.update(task, status=f"done ({read_errors} errors)")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_read_single_tile, ti, zone_grid): ti
                    for ti in tile_infos
                }
                for future in as_completed(futures):
                    ti = futures[future]
                    try:
                        tile_data[id(ti)] = future.result()
                    except Exception as e:
                        logger.warning(
                            f"Failed to read tile ({ti.lon}, {ti.lat}): {e}"
                        )
                        read_errors += 1

        # --- Phase 2: chunk-batched writes (no read-modify-write) ---
        chunk_groups = _group_tiles_by_chunk(tile_infos, zone_grid)
        n_chunks = len(chunk_groups)
        # Each chunk buffer is ~128 MB (embeddings) + ~4 MB (scales);
        # cap writers to limit peak memory.
        chunk_workers = min(workers, max(1, min(n_chunks, 8)))

        errors = 0
        tiles_written = 0
        if console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TextColumn("[dim]{task.fields[status]}", justify="left"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Writing chunks ({chunk_workers} threads, {n_chunks} chunks)",
                    total=n_chunks,
                    status="starting...",
                )

                with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
                    futures = {
                        pool.submit(
                            _write_chunk_batch,
                            store, ck, tile_data, ctiles, zone_grid,
                        ): ck
                        for ck, ctiles in chunk_groups.items()
                    }
                    for future in as_completed(futures):
                        ck = futures[future]
                        try:
                            n = future.result()
                            tiles_written += n
                        except Exception as e:
                            logger.warning(
                                f"Failed to write chunk {ck}: {e}"
                            )
                            errors += 1
                        progress.update(
                            task, status=f"chunk {ck} ({tiles_written} tiles)",
                        )
                        progress.advance(task)

                status = "done"
                if errors:
                    status = f"done ({errors} errors)"
                progress.update(task, status=status)
        else:
            with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
                futures = {
                    pool.submit(
                        _write_chunk_batch,
                        store, ck, tile_data, ctiles, zone_grid,
                    ): ck
                    for ck, ctiles in chunk_groups.items()
                }
                for future in as_completed(futures):
                    ck = futures[future]
                    try:
                        n = future.result()
                        tiles_written += n
                    except Exception as e:
                        logger.warning(
                            f"Failed to write chunk {ck}: {e}"
                        )
                        errors += 1

        # Free tile data to release memory before preview passes
        del tile_data

        # RGB preview pass: compute stretch from the store, then write preview
        if rgb:
            stretch = compute_stretch_from_store(store, workers=workers, console=console)
            if console is not None:
                console.print(
                    f"  RGB stretch: min={[f'{v:.2f}' for v in stretch['min']]}, "
                    f"max={[f'{v:.2f}' for v in stretch['max']]}"
                )
            written = write_rgb_pass(store, stretch, workers=workers, console=console)
            store.attrs.update({
                "has_rgb_preview": True,
                "rgb_bands": list(RGB_PREVIEW_BANDS),
                "rgb_stretch": stretch,
            })
            if console is not None:
                console.print(f"  [green]RGB preview: {written} chunks written[/green]")

        # PCA preview pass: compute PCA basis from all bands, then write PCA preview
        if pca:
            pca_basis = compute_pca_basis(store, workers=workers, console=console)
            if console is not None:
                evr = pca_basis["explained_variance_ratio"]
                console.print(
                    f"  PCA explained variance: "
                    f"[{evr[0]:.1%}, {evr[1]:.1%}, {evr[2]:.1%}] "
                    f"(total {evr.sum():.1%})"
                )
            pca_written = write_pca_pass(store, pca_basis, workers=workers, console=console)
            store.attrs.update({
                "has_pca_preview": True,
                "pca_explained_variance": pca_basis["explained_variance_ratio"].tolist(),
                "pca_components": pca_basis["components"].tolist(),
                "pca_mean": pca_basis["mean"].tolist(),
                "pca_stretch": {
                    "min": pca_basis["p_low"].tolist(),
                    "max": pca_basis["p_high"].tolist(),
                },
            })
            if console is not None:
                console.print(f"  [green]PCA preview: {pca_written} chunks written[/green]")

        store_path = output_dir / store_name
        created_stores.append(store_path)

    return created_stores


def _read_single_tile(
    tile_info: TileInfo,
    zone_grid: ZoneGrid,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Read a single tile from disk and return its data and pixel offset.

    Returns:
        (embedding_int8, scales, row_start, col_start)
    """
    # Read embedding (int8)
    embedding = np.load(tile_info.embedding_path)
    if embedding.ndim != 3 or embedding.shape[2] != N_BANDS:
        raise ValueError(
            f"Unexpected embedding shape {embedding.shape} for "
            f"({tile_info.lon}, {tile_info.lat})"
        )

    # Read scales (float32)
    scales = np.load(tile_info.scales_path)

    # Apply landmask (sets water pixels to NaN, reduces 3D scales to 2D)
    scales = apply_landmask_to_scales(scales, tile_info.landmask_path)

    # Zero out embeddings where scales are NaN (water/no-data)
    nan_mask = np.isnan(scales)
    embedding[nan_mask] = 0

    # Compute pixel offset
    row_start, col_start = tile_pixel_offset(tile_info, zone_grid)

    # Verify bounds
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


# Chunk size used for grouping (must match create_zone_store)
_CHUNK_PX = 1024


def _group_tiles_by_chunk(
    tile_infos: List[TileInfo],
    zone_grid: ZoneGrid,
) -> Dict[Tuple[int, int], List[TileInfo]]:
    """Group tiles by which chunk(s) they overlap.

    A tile that straddles a chunk boundary appears in multiple groups.

    Returns:
        Dict mapping (chunk_row, chunk_col) to list of TileInfo.
    """
    from collections import defaultdict

    chunk_tiles: Dict[Tuple[int, int], List[TileInfo]] = defaultdict(list)

    for ti in tile_infos:
        row_start, col_start = tile_pixel_offset(ti, zone_grid)
        h, w = ti.height, ti.width

        cr_min = row_start // _CHUNK_PX
        cr_max = (row_start + h - 1) // _CHUNK_PX
        cc_min = col_start // _CHUNK_PX
        cc_max = (col_start + w - 1) // _CHUNK_PX

        for cr in range(cr_min, cr_max + 1):
            for cc in range(cc_min, cc_max + 1):
                chunk_tiles[(cr, cc)].append(ti)

    return dict(chunk_tiles)


def _write_chunk_batch(
    store: "zarr.Group",
    chunk_key: Tuple[int, int],
    tile_data_map: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]],
    tile_infos: List[TileInfo],
    zone_grid: ZoneGrid,
) -> int:
    """Assemble pre-read tile data for one chunk and write once.

    Builds a full chunk buffer in memory and writes it in a single
    slice assignment, avoiding read-modify-write on the underlying
    chunk files.

    Args:
        store: Zarr group (root of zone store)
        chunk_key: (chunk_row, chunk_col)
        tile_data_map: id(tile_info) -> (emb, scales, row, col) from read phase
        tile_infos: Tiles that overlap this chunk
        zone_grid: Zone grid specification

    Returns:
        Number of tiles placed into this chunk.
    """
    cr, cc = chunk_key
    r0 = cr * _CHUNK_PX
    c0 = cc * _CHUNK_PX
    r1 = min(r0 + _CHUNK_PX, zone_grid.height_px)
    c1 = min(c0 + _CHUNK_PX, zone_grid.width_px)
    ch = r1 - r0
    cw = c1 - c0

    emb_buf = np.zeros((ch, cw, N_BANDS), dtype=np.int8)
    scales_buf = np.full((ch, cw), np.float32("nan"), dtype=np.float32)

    written = 0
    for ti in tile_infos:
        data = tile_data_map.get(id(ti))
        if data is None:
            continue

        embedding, scales, row_start, col_start = data
        h, w = embedding.shape[:2]

        # Clip tile region to this chunk
        tr0 = max(row_start, r0)
        tc0 = max(col_start, c0)
        tr1 = min(row_start + h, r1)
        tc1 = min(col_start + w, c1)

        if tr0 >= tr1 or tc0 >= tc1:
            continue

        # Offsets into tile arrays
        tile_r0 = tr0 - row_start
        tile_c0 = tc0 - col_start
        tile_r1 = tr1 - row_start
        tile_c1 = tc1 - col_start

        # Offsets into chunk buffer
        buf_r0 = tr0 - r0
        buf_c0 = tc0 - c0
        buf_r1 = tr1 - r0
        buf_c1 = tc1 - c0

        emb_buf[buf_r0:buf_r1, buf_c0:buf_c1, :] = embedding[
            tile_r0:tile_r1, tile_c0:tile_c1, :
        ]
        scales_buf[buf_r0:buf_r1, buf_c0:buf_c1] = scales[
            tile_r0:tile_r1, tile_c0:tile_c1
        ]
        written += 1

    # Single whole-chunk write — no read-modify-write
    store["embeddings"][r0:r1, c0:c1, :] = emb_buf
    store["scales"][r0:r1, c0:c1] = scales_buf

    return written


# =============================================================================
# RGB preview pass
# =============================================================================


def _process_rgb_chunk(
    emb_arr,
    scales_arr,
    rgb_arr,
    ci: int,
    cj: int,
    chunk_h: int,
    chunk_w: int,
    emb_shape: tuple,
    stretch_min: List[float],
    stretch_max: List[float],
) -> bool:
    """Read one chunk, compute RGB, write to store. Runs in a thread.

    Returns True if chunk had data and was written, False if skipped.
    """
    r0 = ci * chunk_h
    r1 = min(r0 + chunk_h, emb_shape[0])
    c0 = cj * chunk_w
    c1 = min(c0 + chunk_w, emb_shape[1])

    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
    if np.all(np.isnan(scales_chunk) | (scales_chunk == 0)):
        return False

    emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
    rgba = compute_rgb_chunk(
        emb_chunk, scales_chunk,
        RGB_PREVIEW_BANDS, stretch_min, stretch_max,
    )
    rgb_arr[r0:r1, c0:c1, :] = rgba
    return True


def write_rgb_pass(
    store: "zarr.Group",
    stretch: dict,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Write RGB preview data into an existing store's rgb array.

    Reads embeddings and scales, computes RGBA, and writes to the rgb
    array — all in parallel using threads (zarr I/O releases the GIL).

    Args:
        store: Zarr group with embeddings, scales, and rgb arrays
        stretch: Dict with 'min' and 'max' lists (3 floats each)
        workers: Number of threads (default 8)
        console: Optional Rich Console for progress

    Returns:
        Number of chunks written
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    rgb_arr = store["rgb"]

    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)
    total_chunks = n_rows * n_cols

    stretch_min = stretch["min"]
    stretch_max = stretch["max"]
    written = 0

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Writing RGB preview ({workers} threads)",
                total=total_chunks,
            )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_rgb_chunk,
                        emb_arr, scales_arr, rgb_arr,
                        ci, cj, chunk_h, chunk_w, emb_shape,
                        stretch_min, stretch_max,
                    ): (ci, cj)
                    for ci in range(n_rows)
                    for cj in range(n_cols)
                }
                for future in as_completed(futures):
                    if future.result():
                        written += 1
                    progress.advance(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_rgb_chunk,
                    emb_arr, scales_arr, rgb_arr,
                    ci, cj, chunk_h, chunk_w, emb_shape,
                    stretch_min, stretch_max,
                ): (ci, cj)
                for ci in range(n_rows)
                for cj in range(n_cols)
            }
            for future in as_completed(futures):
                if future.result():
                    written += 1

    return written


def add_rgb_to_existing_store(
    store_path: Path,
    workers: Optional[int] = None,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add RGB preview array to an existing Zarr store.

    Two-pass process:
    1. Parallel chunk reads to compute percentile stretch
    2. Parallel RGB chunk writes

    Args:
        store_path: Path to existing .zarr directory
        workers: Number of threads (default: cpu_count, max 16)
        console: Optional Rich Console for progress
    """
    import zarr


    store = zarr.open_group(str(store_path), mode="r+")

    # Create rgb array if missing
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
        console.print(f"  Pass 2: Writing RGB preview...")

    # Pass 2: write RGB
    written = write_rgb_pass(store, stretch, workers=workers, console=console)

    # Update store attrs
    store.attrs.update({
        "has_rgb_preview": True,
        "rgb_bands": list(RGB_PREVIEW_BANDS),
        "rgb_stretch": stretch,
    })

    if console is not None:
        console.print(f"  [green]RGB preview: {written} chunks written[/green]")


# =============================================================================
# PCA preview pass
# =============================================================================


def _sample_chunk_pca_stats(
    emb_arr,
    scales_arr,
    ci: int,
    cj: int,
    chunk_h: int,
    chunk_w: int,
    emb_shape: tuple,
    max_per_chunk: int = 5000,
) -> Optional[np.ndarray]:
    """Read one chunk and return subsampled dequantised values for ALL bands.

    Like ``_sample_chunk_stats`` but reads all 128 bands for PCA.

    Returns:
        (N, 128) float32 array of dequantised samples, or None if chunk empty.
    """
    r0 = ci * chunk_h
    r1 = min(r0 + chunk_h, emb_shape[0])
    c0 = cj * chunk_w
    c1 = min(c0 + chunk_w, emb_shape[1])

    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
    valid = ~np.isnan(scales_chunk) & (scales_chunk != 0)
    if not np.any(valid):
        return None

    emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])

    scales_valid = scales_chunk[valid]
    all_valid = emb_chunk[valid].astype(np.float32) * scales_valid[:, np.newaxis]

    if all_valid.shape[0] > max_per_chunk:
        rng = np.random.default_rng(ci * 10007 + cj)
        idx = rng.choice(all_valid.shape[0], max_per_chunk, replace=False)
        all_valid = all_valid[idx]

    return all_valid


def _randomized_svd(
    data: np.ndarray,
    n_components: int,
    n_oversamples: int = 10,
    n_power_iter: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD for low-rank approximation.

    Much faster than full SVD when n_components << min(n_samples, n_features).
    Uses the Halko-Martinsson-Tropp algorithm.

    Args:
        data: (n_samples, n_features) array (should be centred)
        n_components: Number of components to extract
        n_oversamples: Extra dimensions for accuracy (default 10)
        n_power_iter: Power iterations for better accuracy (default 2)
        rng: Random generator

    Returns:
        (U, S, Vt) truncated to n_components
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples, n_features = data.shape
    k = n_components + n_oversamples

    # Random projection: Omega is (n_features, k)
    omega = rng.standard_normal((n_features, k)).astype(np.float32)

    # Form Y = data @ Omega  →  (n_samples, k)
    Y = data @ omega

    # Power iterations for better approximation of top singular space
    for _ in range(n_power_iter):
        Y = data @ (data.T @ Y)

    # QR factorisation of Y
    Q, _ = np.linalg.qr(Y)  # Q is (n_samples, k)

    # Project data into low-rank space: B = Q.T @ data  →  (k, n_features)
    B = Q.T @ data

    # SVD of the small matrix B
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Recover U in original space
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
    """Compute PCA basis from an existing store using parallel reads.

    Two phases:
    1. Parallel subsampled reads from a random subset of chunks
    2. Randomized SVD on centred data (fast for low n_components)

    Args:
        store: Zarr group with embeddings and scales arrays
        n_components: Number of PCA components (default 3 for RGB)
        max_per_chunk: Max pixels to sample per chunk
        max_total_samples: Cap total samples (default 200K — ample for 128-dim PCA)
        chunk_sample_fraction: Fraction of chunks to read (default 0.25)
        workers: Number of threads
        console: Optional Rich Console for progress

    Returns:
        Dict with 'components' (3, 128), 'mean' (128,),
        'p_low' (3,), 'p_high' (3,), 'explained_variance_ratio' (3,)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)
    total_chunks = n_rows * n_cols

    # Select a random subset of chunks to read — PCA on 128 dims
    # converges well with ~50-100K samples, no need to read everything
    all_chunk_indices = [(ci, cj) for ci in range(n_rows) for cj in range(n_cols)]
    rng = np.random.default_rng(42)

    n_to_sample = max(4, int(total_chunks * chunk_sample_fraction))
    if n_to_sample < total_chunks:
        sampled_indices = rng.choice(
            len(all_chunk_indices), n_to_sample, replace=False,
        )
        chunk_indices = [all_chunk_indices[i] for i in sampled_indices]
    else:
        chunk_indices = all_chunk_indices

    n_sampled_chunks = len(chunk_indices)
    samples = []

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Sampling for PCA ({n_sampled_chunks}/{total_chunks} chunks, {workers} threads)",
                total=n_sampled_chunks,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _sample_chunk_pca_stats,
                        emb_arr, scales_arr, ci, cj,
                        chunk_h, chunk_w, emb_shape,
                        max_per_chunk,
                    ): (ci, cj)
                    for ci, cj in chunk_indices
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        samples.append(result)
                    progress.advance(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _sample_chunk_pca_stats,
                    emb_arr, scales_arr, ci, cj,
                    chunk_h, chunk_w, emb_shape,
                    max_per_chunk,
                ): (ci, cj)
                for ci, cj in chunk_indices
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    samples.append(result)

    if not samples:
        # Return identity-like basis as fallback
        components = np.eye(n_components, N_BANDS, dtype=np.float32)
        return {
            "components": components,
            "mean": np.zeros(N_BANDS, dtype=np.float32),
            "p_low": np.zeros(n_components, dtype=np.float32),
            "p_high": np.ones(n_components, dtype=np.float32),
            "explained_variance_ratio": np.zeros(n_components, dtype=np.float32),
        }

    all_data = np.concatenate(samples, axis=0)  # (N, 128)

    # Cap total samples to keep SVD fast
    if all_data.shape[0] > max_total_samples:
        idx = rng.choice(all_data.shape[0], max_total_samples, replace=False)
        all_data = all_data[idx]

    n_samples = all_data.shape[0]
    if console is not None:
        console.print(
            f"  PCA: {n_samples} samples from {n_sampled_chunks} chunks, "
            f"computing randomized SVD..."
        )

    # Centre the data
    mean = all_data.mean(axis=0)  # (128,)
    centred = all_data - mean  # (N, 128)

    # Randomized SVD — O(N * k^2) instead of O(N * d^2) for full SVD
    # Much faster when n_components (3) << n_features (128)
    U, S, Vt = _randomized_svd(centred, n_components, rng=rng)
    components = Vt.astype(np.float32)  # (3, 128)

    # Explained variance ratio (approximate — uses only the top k singular values)
    # For exact ratio we'd need all singular values, but the approximation
    # is good enough: use total variance from centred data directly
    total_var = np.sum(centred ** 2)
    explained = (S ** 2) / total_var if total_var > 0 else np.zeros(n_components)
    explained = explained.astype(np.float32)

    # Project all samples to get stretch percentiles
    projected = centred @ components.T  # (N, 3)
    p_low = np.percentile(projected, 2, axis=0).astype(np.float32)  # (3,)
    p_high = np.percentile(projected, 98, axis=0).astype(np.float32)  # (3,)

    # Ensure non-zero range
    for i in range(n_components):
        if p_high[i] <= p_low[i]:
            p_high[i] = p_low[i] + 1.0

    return {
        "components": components,
        "mean": mean.astype(np.float32),
        "p_low": p_low,
        "p_high": p_high,
        "explained_variance_ratio": explained,
    }


def compute_pca_chunk(
    embedding_int8: np.ndarray,
    scales: np.ndarray,
    pca_basis: dict,
) -> np.ndarray:
    """Compute an RGBA uint8 PCA preview from embedding + scales.

    Args:
        embedding_int8: int8 array (H, W, 128)
        scales: float32 array (H, W)
        pca_basis: Dict with 'components', 'mean', 'p_low', 'p_high'

    Returns:
        uint8 RGBA array of shape (H, W, 4)
    """
    h, w = scales.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = ~np.isnan(scales) & (scales != 0)
    n_valid = np.count_nonzero(valid)
    if n_valid == 0:
        return rgba

    components = pca_basis["components"]  # (3, 128)
    mean = pca_basis["mean"]              # (128,)
    p_low = pca_basis["p_low"]            # (3,)
    p_high = pca_basis["p_high"]          # (3,)

    # Dequantise valid pixels
    scales_safe = np.where(valid, scales, 0.0)
    float_emb = embedding_int8.astype(np.float32) * scales_safe[:, :, np.newaxis]

    # Reshape to (H*W, 128) for matrix multiply
    flat = float_emb.reshape(-1, N_BANDS)
    valid_flat = valid.ravel()

    # Project: (H*W, 128) @ (128, 3) -> (H*W, 3)
    projected = (flat - mean) @ components.T

    # Stretch to [0, 1]
    for i in range(3):
        projected[:, i] = (projected[:, i] - p_low[i]) / (p_high[i] - p_low[i])

    # Clip and scale to uint8
    projected = np.clip(projected, 0, 1) * 255
    rgb_flat = projected.astype(np.uint8)  # (H*W, 3)

    # Write into RGBA
    rgba_flat = rgba.reshape(-1, 4)
    rgba_flat[:, 0] = rgb_flat[:, 0]
    rgba_flat[:, 1] = rgb_flat[:, 1]
    rgba_flat[:, 2] = rgb_flat[:, 2]

    # Zero out invalid, set alpha
    inv_flat = ~valid_flat
    rgba_flat[inv_flat, 0] = 0
    rgba_flat[inv_flat, 1] = 0
    rgba_flat[inv_flat, 2] = 0
    rgba_flat[:, 3] = np.where(valid_flat, 255, 0).astype(np.uint8)

    return rgba


def _process_pca_chunk(
    emb_arr,
    scales_arr,
    pca_rgb_arr,
    ci: int,
    cj: int,
    chunk_h: int,
    chunk_w: int,
    emb_shape: tuple,
    pca_basis: dict,
) -> bool:
    """Read one chunk, compute PCA RGB, write to store. Runs in a thread.

    Returns True if chunk had data and was written, False if skipped.
    """
    r0 = ci * chunk_h
    r1 = min(r0 + chunk_h, emb_shape[0])
    c0 = cj * chunk_w
    c1 = min(c0 + chunk_w, emb_shape[1])

    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
    if np.all(np.isnan(scales_chunk) | (scales_chunk == 0)):
        return False

    emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
    rgba = compute_pca_chunk(emb_chunk, scales_chunk, pca_basis)
    pca_rgb_arr[r0:r1, c0:c1, :] = rgba
    return True


def write_pca_pass(
    store: "zarr.Group",
    pca_basis: dict,
    workers: int = 8,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Write PCA preview data into an existing store's pca_rgb array.

    Args:
        store: Zarr group with embeddings, scales, and pca_rgb arrays
        pca_basis: Dict from compute_pca_basis
        workers: Number of threads (default 8)
        console: Optional Rich Console for progress

    Returns:
        Number of chunks written
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    pca_rgb_arr = store["pca_rgb"]

    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)
    total_chunks = n_rows * n_cols

    written = 0

    if console is not None:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TextColumn,
            MofNCompleteColumn, TimeElapsedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Writing PCA preview ({workers} threads)",
                total=total_chunks,
            )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_pca_chunk,
                        emb_arr, scales_arr, pca_rgb_arr,
                        ci, cj, chunk_h, chunk_w, emb_shape,
                        pca_basis,
                    ): (ci, cj)
                    for ci in range(n_rows)
                    for cj in range(n_cols)
                }
                for future in as_completed(futures):
                    if future.result():
                        written += 1
                    progress.advance(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_pca_chunk,
                    emb_arr, scales_arr, pca_rgb_arr,
                    ci, cj, chunk_h, chunk_w, emb_shape,
                    pca_basis,
                ): (ci, cj)
                for ci in range(n_rows)
                for cj in range(n_cols)
            }
            for future in as_completed(futures):
                if future.result():
                    written += 1

    return written


def add_pca_to_existing_store(
    store_path: Path,
    workers: Optional[int] = None,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add PCA RGB preview array to an existing Zarr store.

    Two-pass process:
    1. Parallel chunk reads to compute PCA basis (subsample + SVD)
    2. Parallel PCA chunk writes

    Args:
        store_path: Path to existing .zarr directory
        workers: Number of threads (default: cpu_count, max 16)
        console: Optional Rich Console for progress
    """
    import zarr


    store = zarr.open_group(str(store_path), mode="r+")

    # Create pca_rgb array if missing
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
        console.print(f"  Pass 2: Writing PCA preview...")

    written = write_pca_pass(store, pca_basis, workers=workers, console=console)

    # Update store attrs
    store.attrs.update({
        "has_pca_preview": True,
        "pca_explained_variance": pca_basis["explained_variance_ratio"].tolist(),
        "pca_components": pca_basis["components"].tolist(),
        "pca_mean": pca_basis["mean"].tolist(),
        "pca_stretch": {
            "min": pca_basis["p_low"].tolist(),
            "max": pca_basis["p_high"].tolist(),
        },
    })

    if console is not None:
        console.print(f"  [green]PCA preview: {written} chunks written[/green]")


# =============================================================================
# Reading support
# =============================================================================


def open_zone_store(path) -> "xarray.Dataset":
    """Open a zone Zarr store as an xarray Dataset.

    Args:
        path: Path to the .zarr directory

    Returns:
        xarray Dataset with embeddings, scales, and coordinate arrays
    """
    import xarray as xr

    ds = xr.open_zarr(str(path))
    return ds


def read_region_from_zone(
    path,
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read a spatial subset from a zone store.

    Args:
        path: Path to the .zarr directory
        bbox: Bounding box in UTM coordinates (min_easting, min_northing,
              max_easting, max_northing)

    Returns:
        Tuple of (embeddings_int8, scales_float32, attrs)
        where embeddings is (H, W, 128) int8 and scales is (H, W) float32
    """
    import zarr

    store = zarr.open_group(str(path), mode="r")
    attrs = dict(store.attrs)

    transform = attrs["transform"]
    pixel_size = transform[0]
    origin_easting = transform[2]
    origin_northing = transform[5]

    min_e, min_n, max_e, max_n = bbox

    # Convert to pixel coordinates
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
