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
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Number of embedding bands in Tessera
N_BANDS = 128


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
) -> "zarr.Group":
    """Create a new Zarr store for a UTM zone.

    Args:
        zone_grid: Zone grid specification
        output_dir: Directory to create the store in
        geotessera_version: Version of geotessera
        dataset_version: Tessera dataset version

    Returns:
        zarr.Group root of the created store
    """
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

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
        codecs=[BytesCodec(), ZstdCodec(level=3)],
    )

    # Create scales array: float32 (northing, easting)
    store.create_array(
        "scales",
        shape=(zone_grid.height_px, zone_grid.width_px),
        chunks=(1024, 1024),
        dtype=np.float32,
        fill_value=np.float32("nan"),
        codecs=[BytesCodec(), ZstdCodec(level=3)],
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
        codecs=[BytesCodec(), ZstdCodec(level=3)],
    )
    store["easting"][:] = easting_coords

    store.create_array(
        "northing",
        shape=(zone_grid.height_px,),
        dtype=np.float64,
        fill_value=0.0,
        codecs=[BytesCodec(), ZstdCodec(level=3)],
    )
    store["northing"][:] = northing_coords

    store.create_array(
        "band",
        shape=(N_BANDS,),
        dtype=np.int32,
        fill_value=0,
        codecs=[BytesCodec(), ZstdCodec(level=3)],
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
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[int, List[TileInfo]]:
    """Gather tile metadata from local files and group by UTM zone.

    Iterates all available tiles for the given year in the registry,
    checks that the corresponding files exist locally, reads CRS info
    from each landmask, and groups tiles by UTM zone.

    Args:
        registry: Registry instance for tile/landmask access
        year: Year of embeddings
        zones: Optional list of zone numbers to include. If None, all zones.
        progress_callback: Optional callback for status updates

    Returns:
        Dict mapping zone number to list of TileInfo
    """
    import rasterio
    from .registry import (
        EMBEDDINGS_DIR_NAME,
        LANDMASKS_DIR_NAME,
        tile_to_embedding_paths,
        tile_to_landmask_filename,
    )

    # Get all tiles for this year
    tiles = [
        (y, lon, lat)
        for y, lon, lat in registry.get_available_embeddings()
        if y == year
    ]

    if progress_callback:
        progress_callback(f"Found {len(tiles)} tiles for year {year}")

    zones_dict: Dict[int, List[TileInfo]] = {}

    for tile_year, tile_lon, tile_lat in tiles:
        # Build file paths
        emb_rel, scales_rel = tile_to_embedding_paths(tile_lon, tile_lat, tile_year)
        emb_path = str(registry._embeddings_dir / EMBEDDINGS_DIR_NAME / emb_rel)
        scales_path = str(registry._embeddings_dir / EMBEDDINGS_DIR_NAME / scales_rel)
        landmask_filename = tile_to_landmask_filename(tile_lon, tile_lat)
        landmask_path = str(registry._embeddings_dir / LANDMASKS_DIR_NAME / landmask_filename)

        # Check files exist
        if not Path(emb_path).exists():
            logger.warning(f"Embedding not found: {emb_path}")
            continue
        if not Path(scales_path).exists():
            logger.warning(f"Scales not found: {scales_path}")
            continue
        if not Path(landmask_path).exists():
            logger.warning(f"Landmask not found: {landmask_path}")
            continue

        # Read landmask to get CRS, transform, dimensions
        try:
            with rasterio.open(landmask_path) as src:
                epsg = src.crs.to_epsg()
                transform = src.transform
                height = src.height
                width = src.width
        except Exception as e:
            logger.warning(
                f"Failed to read landmask for ({tile_lon}, {tile_lat}): {e}"
            )
            continue

        if epsg is None:
            logger.warning(
                f"No EPSG code for landmask ({tile_lon}, {tile_lat}), skipping"
            )
            continue

        # Extract zone
        try:
            zone = epsg_to_utm_zone(epsg)
        except ValueError:
            logger.warning(
                f"Non-UTM EPSG {epsg} for tile ({tile_lon}, {tile_lat}), skipping"
            )
            continue

        # Filter by requested zones
        if zones is not None and zone not in zones:
            continue

        # Verify grid alignment
        if abs(transform.a - 10.0) > 0.01 or abs(transform.e - (-10.0)) > 0.01:
            logger.warning(
                f"Tile ({tile_lon}, {tile_lat}) has non-standard pixel size "
                f"({transform.a}, {transform.e}), skipping"
            )
            continue

        ti = TileInfo(
            lon=tile_lon,
            lat=tile_lat,
            year=tile_year,
            epsg=epsg,
            transform=transform,
            height=height,
            width=width,
            landmask_path=landmask_path,
            embedding_path=emb_path,
            scales_path=scales_path,
        )

        zones_dict.setdefault(zone, []).append(ti)

    if progress_callback:
        zone_summary = ", ".join(
            f"zone {z}: {len(t)} tiles" for z, t in sorted(zones_dict.items())
        )
        progress_callback(f"Grouped into zones: {zone_summary}")

    return zones_dict


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
    progress_callback: Optional[Callable[[str], None]] = None,
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
        progress_callback: Optional callback for status messages

    Returns:
        List of paths to created Zarr stores
    """
    output_dir = Path(output_dir)

    # Gather and group tiles
    zones_dict = gather_tile_infos(
        registry,
        year,
        zones=zones,
        progress_callback=progress_callback,
    )

    if not zones_dict:
        if progress_callback:
            progress_callback("No tiles found matching criteria")
        return []

    if dry_run:
        for zone_num, tile_infos in sorted(zones_dict.items()):
            if progress_callback:
                progress_callback(
                    f"Zone {zone_num} (EPSG:{zone_canonical_epsg(zone_num)}): "
                    f"{len(tile_infos)} tiles"
                )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created_stores: List[Path] = []

    for zone_num, tile_infos in sorted(zones_dict.items()):
        if progress_callback:
            progress_callback(
                f"Building zone {zone_num} ({len(tile_infos)} tiles)..."
            )

        # Compute grid
        zone_grid = compute_zone_grid(tile_infos, year)

        if progress_callback:
            progress_callback(
                f"  Grid: {zone_grid.width_px}x{zone_grid.height_px} pixels "
                f"({zone_grid.width_px * zone_grid.pixel_size / 1000:.1f} x "
                f"{zone_grid.height_px * zone_grid.pixel_size / 1000:.1f} km)"
            )

        # Create store
        store = create_zone_store(
            zone_grid,
            output_dir,
            geotessera_version=geotessera_version,
            dataset_version=dataset_version,
        )

        # Write tiles
        for i, ti in enumerate(tile_infos):
            if progress_callback:
                progress_callback(
                    f"  Writing tile {i + 1}/{len(tile_infos)}: "
                    f"({ti.lon:.2f}, {ti.lat:.2f})"
                )

            try:
                _write_single_tile(store, ti, zone_grid)
            except Exception as e:
                logger.warning(
                    f"Failed to write tile ({ti.lon}, {ti.lat}): {e}"
                )
                continue

        store_path = output_dir / _store_name(zone_grid.zone, zone_grid.year)
        created_stores.append(store_path)

        if progress_callback:
            progress_callback(
                f"  Wrote {store_path.name}"
            )

    return created_stores


def _write_single_tile(
    store: "zarr.Group",
    tile_info: TileInfo,
    zone_grid: ZoneGrid,
) -> None:
    """Read a single tile's data and write it into the zone store.

    Args:
        store: Zarr group (root of zone store)
        tile_info: Tile metadata
        zone_grid: Zone grid specification
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
