"""Zone-wide Zarr format for consolidated Tessera embeddings.

This module provides tools for building and reading Zarr v3 stores that
consolidate all tiles within a UTM zone into a single store per year.
This enables efficient spatial subsetting and cloud-native access.

Store layout:
    utm{zone:02d}_{year}.zarr/
        embeddings    # int8    (northing, easting, band)  chunks=(1024, 1024, 128)
        scales        # float32 (northing, easting)        chunks=(1024, 1024)

NaN in scales indicates no-data (water or no coverage).
"""

import logging
import math
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


@dataclass
class BandStats:
    """Reservoir-sampled accumulator for per-band min/max of RGB preview bands.

    Collects dequantised values (int8 * scale) for bands 0, 1, 2 using
    reservoir sampling, then computes percentile stretch for consistent
    visualisation across an entire zone store.
    """

    max_samples: int = 2_000_000
    _samples: List = field(default_factory=lambda: [[], [], []])
    _count: int = 0

    def accumulate_tile(
        self, embedding_int8: np.ndarray, scales: np.ndarray
    ) -> None:
        """Add dequantised band 0/1/2 values from one tile.

        Args:
            embedding_int8: int8 array (H, W, 128)
            scales: float32 array (H, W) with NaN for no-data
        """
        valid = ~np.isnan(scales) & (scales != 0)
        if not np.any(valid):
            return

        for i, band_idx in enumerate(RGB_PREVIEW_BANDS):
            raw = embedding_int8[:, :, band_idx][valid].astype(np.float32)
            vals = raw * scales[valid]
            # Reservoir sampling: keep up to max_samples per band
            room = self.max_samples - len(self._samples[i])
            if room >= len(vals):
                self._samples[i].append(vals)
            elif room > 0:
                self._samples[i].append(vals[:room])
            else:
                # Random replacement for overflow
                rng = np.random.default_rng(self._count)
                indices = rng.integers(0, self._count + len(vals), size=min(1000, len(vals)))
                keep = indices[indices < self.max_samples]
                if len(keep) > 0:
                    combined = np.concatenate(self._samples[i])
                    combined[keep[:len(keep)]] = vals[:len(keep)]
                    self._samples[i] = [combined]

        self._count += int(np.sum(valid))

    def compute_stretch(
        self, p_low: float = 2, p_high: float = 98
    ) -> dict:
        """Compute percentile-based stretch for each RGB band.

        Args:
            p_low: Low percentile (default 2)
            p_high: High percentile (default 98)

        Returns:
            Dict with 'min' and 'max' lists of 3 floats each.
        """
        mins = []
        maxs = []
        for i in range(3):
            if self._samples[i]:
                all_vals = np.concatenate(self._samples[i])
                lo = float(np.percentile(all_vals, p_low))
                hi = float(np.percentile(all_vals, p_high))
                if hi <= lo:
                    hi = lo + 1.0
                mins.append(lo)
                maxs.append(hi)
            else:
                mins.append(0.0)
                maxs.append(1.0)
        return {"min": mins, "max": maxs}


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

    for i, band_idx in enumerate(band_indices):
        raw = embedding_int8[:, :, band_idx].astype(np.float32)
        dequant = raw * scales
        lo, hi = stretch_min[i], stretch_max[i]
        normalised = (dequant - lo) / (hi - lo)
        channel = np.clip(normalised * 255, 0, 255).astype(np.uint8)
        channel[~valid] = 0
        rgba[:, :, i] = channel

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
) -> "zarr.Group":
    """Create a new Zarr store for a UTM zone.

    Args:
        zone_grid: Zone grid specification
        output_dir: Directory to create the store in
        geotessera_version: Version of geotessera
        dataset_version: Tessera dataset version
        include_rgb: If True, create an additional RGB preview array

    Returns:
        zarr.Group root of the created store
    """
    import zarr
    from zarr.codecs import ZstdCodec

    zstd = ZstdCodec(level=3)

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
        compressors=zstd,
        dimension_names=["northing", "easting", "band"],
    )

    # Create scales array: float32 (northing, easting)
    store.create_array(
        "scales",
        shape=(zone_grid.height_px, zone_grid.width_px),
        chunks=(1024, 1024),
        dtype=np.float32,
        fill_value=np.float32("nan"),
        compressors=zstd,
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
            compressors=zstd,
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
        compressors=zstd,
        dimension_names=["easting"],
    )
    store["easting"][:] = easting_coords

    store.create_array(
        "northing",
        shape=(zone_grid.height_px,),
        dtype=np.float64,
        fill_value=0.0,
        compressors=zstd,
        dimension_names=["northing"],
    )
    store["northing"][:] = northing_coords

    store.create_array(
        "band",
        shape=(N_BANDS,),
        dtype=np.int32,
        fill_value=0,
        compressors=zstd,
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

    # Get all tiles for this year
    tiles = [
        (y, lon, lat)
        for y, lon, lat in registry.get_available_embeddings()
        if y == year
    ]

    # Pre-compute file paths and filter to tiles that exist on disk
    scan_args = []
    base_emb = str(registry._embeddings_dir / EMBEDDINGS_DIR_NAME)
    base_lm = str(registry._embeddings_dir / LANDMASKS_DIR_NAME)
    zone_set = set(zones) if zones is not None else None

    for tile_year, tile_lon, tile_lat in tiles:
        emb_rel, scales_rel = tile_to_embedding_paths(tile_lon, tile_lat, tile_year)
        emb_path = str(Path(base_emb) / emb_rel)
        scales_path = str(Path(base_emb) / scales_rel)
        landmask_path = str(
            Path(base_lm) / tile_to_landmask_filename(tile_lon, tile_lat)
        )

        # Quick existence check before submitting to workers
        if (
            Path(emb_path).exists()
            and Path(scales_path).exists()
            and Path(landmask_path).exists()
        ):
            scan_args.append(
                (tile_year, tile_lon, tile_lat, emb_path, scales_path, landmask_path)
            )

    zones_dict: Dict[int, List[TileInfo]] = {}
    skipped = 0
    all_warnings: List[str] = []
    n_missing = len(tiles) - len(scan_args)

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

    Returns:
        List of paths to created Zarr stores
    """
    output_dir = Path(output_dir)

    # Gather and group tiles
    zones_dict = gather_tile_infos(
        registry,
        year,
        zones=zones,
        console=console,
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
        )

        # Create band stats accumulator if RGB is requested
        band_stats = BandStats() if rgb else None

        # Write tiles with progress bar
        errors = 0
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
                    f"Writing zone {zone_num}",
                    total=len(tile_infos),
                    status="starting...",
                )

                for ti in tile_infos:
                    progress.update(
                        task,
                        status=f"({ti.lon:.2f}, {ti.lat:.2f})",
                    )
                    try:
                        _write_single_tile(store, ti, zone_grid, band_stats=band_stats)
                    except Exception as e:
                        logger.warning(
                            f"Failed to write tile ({ti.lon}, {ti.lat}): {e}"
                        )
                        errors += 1
                    progress.advance(task)

                status = "done"
                if errors:
                    status = f"done ({errors} errors)"
                progress.update(task, status=status)
        else:
            for ti in tile_infos:
                try:
                    _write_single_tile(store, ti, zone_grid, band_stats=band_stats)
                except Exception as e:
                    logger.warning(
                        f"Failed to write tile ({ti.lon}, {ti.lat}): {e}"
                    )
                    errors += 1

        # RGB preview pass (two-pass: stats already accumulated during tile writing)
        if rgb and band_stats is not None:
            stretch = band_stats.compute_stretch()
            if console is not None:
                console.print(
                    f"  RGB stretch: min={[f'{v:.2f}' for v in stretch['min']]}, "
                    f"max={[f'{v:.2f}' for v in stretch['max']]}"
                )
            written = write_rgb_pass(store, stretch, console=console)
            store.attrs.update({
                "has_rgb_preview": True,
                "rgb_bands": list(RGB_PREVIEW_BANDS),
                "rgb_stretch": stretch,
            })
            if console is not None:
                console.print(f"  [green]RGB preview: {written} chunks written[/green]")

        store_path = output_dir / store_name
        created_stores.append(store_path)

    return created_stores


def _write_single_tile(
    store: "zarr.Group",
    tile_info: TileInfo,
    zone_grid: ZoneGrid,
    band_stats: Optional[BandStats] = None,
) -> None:
    """Read a single tile's data and write it into the zone store.

    Args:
        store: Zarr group (root of zone store)
        tile_info: Tile metadata
        zone_grid: Zone grid specification
        band_stats: Optional BandStats accumulator for RGB preview
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

    write_tile_to_store(store, embedding, scales, row_start, col_start)

    if band_stats is not None:
        band_stats.accumulate_tile(embedding, scales)


# =============================================================================
# RGB preview pass
# =============================================================================


def write_rgb_pass(
    store: "zarr.Group",
    stretch: dict,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Write RGB preview data into an existing store's rgb array.

    Reads embeddings and scales chunk-by-chunk and writes computed RGBA
    into the rgb array.

    Args:
        store: Zarr group with embeddings, scales, and rgb arrays
        stretch: Dict with 'min' and 'max' lists (3 floats each)
        console: Optional Rich Console for progress

    Returns:
        Number of chunks written
    """
    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    rgb_arr = store["rgb"]

    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)

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
                "Writing RGB preview",
                total=n_rows * n_cols,
            )

            for ci in range(n_rows):
                for cj in range(n_cols):
                    r0 = ci * chunk_h
                    r1 = min(r0 + chunk_h, emb_shape[0])
                    c0 = cj * chunk_w
                    c1 = min(c0 + chunk_w, emb_shape[1])

                    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
                    if np.all(np.isnan(scales_chunk) | (scales_chunk == 0)):
                        progress.advance(task)
                        continue

                    emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
                    rgba = compute_rgb_chunk(
                        emb_chunk, scales_chunk,
                        RGB_PREVIEW_BANDS, stretch_min, stretch_max,
                    )
                    rgb_arr[r0:r1, c0:c1, :] = rgba
                    written += 1
                    progress.advance(task)
    else:
        for ci in range(n_rows):
            for cj in range(n_cols):
                r0 = ci * chunk_h
                r1 = min(r0 + chunk_h, emb_shape[0])
                c0 = cj * chunk_w
                c1 = min(c0 + chunk_w, emb_shape[1])

                scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
                if np.all(np.isnan(scales_chunk) | (scales_chunk == 0)):
                    continue

                emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
                rgba = compute_rgb_chunk(
                    emb_chunk, scales_chunk,
                    RGB_PREVIEW_BANDS, stretch_min, stretch_max,
                )
                rgb_arr[r0:r1, c0:c1, :] = rgba
                written += 1

    return written


def add_rgb_to_existing_store(
    store_path: Path,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add RGB preview array to an existing Zarr store.

    Two-pass process:
    1. Iterate all chunks, accumulate BandStats from existing embeddings+scales
    2. Write RGB preview data

    Args:
        store_path: Path to existing .zarr directory
        console: Optional Rich Console for progress
    """
    import zarr
    from zarr.codecs import ZstdCodec

    store = zarr.open_group(str(store_path), mode="r+")

    # Create rgb array if missing
    try:
        _ = store["rgb"]
    except KeyError:
        zstd = ZstdCodec(level=3)
        emb_shape = store["embeddings"].shape
        store.create_array(
            "rgb",
            shape=(emb_shape[0], emb_shape[1], 4),
            chunks=(1024, 1024, 4),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=zstd,
            dimension_names=["northing", "easting", "rgba"],
        )

    if console is not None:
        console.print(f"  Pass 1: Computing band statistics...")

    # Pass 1: accumulate stats
    band_stats = BandStats()
    emb_arr = store["embeddings"]
    scales_arr = store["scales"]
    emb_shape = emb_arr.shape
    chunk_h, chunk_w = emb_arr.chunks[:2]
    n_rows = math.ceil(emb_shape[0] / chunk_h)
    n_cols = math.ceil(emb_shape[1] / chunk_w)

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
                "Accumulating stats",
                total=n_rows * n_cols,
            )
            for ci in range(n_rows):
                for cj in range(n_cols):
                    r0 = ci * chunk_h
                    r1 = min(r0 + chunk_h, emb_shape[0])
                    c0 = cj * chunk_w
                    c1 = min(c0 + chunk_w, emb_shape[1])

                    scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
                    if np.all(np.isnan(scales_chunk) | (scales_chunk == 0)):
                        progress.advance(task)
                        continue

                    emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
                    band_stats.accumulate_tile(emb_chunk, scales_chunk)
                    progress.advance(task)
    else:
        for ci in range(n_rows):
            for cj in range(n_cols):
                r0 = ci * chunk_h
                r1 = min(r0 + chunk_h, emb_shape[0])
                c0 = cj * chunk_w
                c1 = min(c0 + chunk_w, emb_shape[1])

                scales_chunk = np.asarray(scales_arr[r0:r1, c0:c1])
                if np.all(np.isnan(scales_chunk) | (scales_chunk == 0)):
                    continue

                emb_chunk = np.asarray(emb_arr[r0:r1, c0:c1, :])
                band_stats.accumulate_tile(emb_chunk, scales_chunk)

    stretch = band_stats.compute_stretch()
    if console is not None:
        console.print(f"  Stretch: min={stretch['min']}, max={stretch['max']}")
        console.print(f"  Pass 2: Writing RGB preview...")

    # Pass 2: write RGB
    written = write_rgb_pass(store, stretch, console=console)

    # Update store attrs
    store.attrs.update({
        "has_rgb_preview": True,
        "rgb_bands": list(RGB_PREVIEW_BANDS),
        "rgb_stretch": stretch,
    })

    if console is not None:
        console.print(f"  [green]RGB preview: {written} chunks written[/green]")


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
