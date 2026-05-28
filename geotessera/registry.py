"""Registry management for Tessera data files.

This module handles all registry-related operations including loading and querying
the Parquet registry, and direct HTTP downloads with local caching.

Also includes utilities for block-based registry management, organizing global grid
data into 5x5 degree blocks for efficient data access.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Iterator, Callable
import os
import math
import re
import logging
import numpy as np
import hashlib
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import time

from botocore.httpchecksum import CrtCrc64NvmeChecksum

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required for registry operations")

try:
    import geopandas as gpd
except ImportError:
    raise ImportError(
        "geopandas is required for spatial operations. Install with: pip install geopandas"
    )

# Constants for block-based registry management
BLOCK_SIZE = 5  # 5x5 degree blocks

# Default dataset variant. The bare ``global_0.1_degree_representation`` dir on
# S3 corresponds to this variant; named variants get a ``.<name>`` suffix.
DEFAULT_VARIANT = "vultr"


def _parse_dataset_version(spec: str) -> Tuple[str, str]:
    """Parse a flexible dataset-version spec.

    Returns ``(s3_path_component, normalized_version)``. Accepts inputs like
    ``"v1"``, ``"1"``, ``"1.0"``, ``"v1.0"`` (all → ``("v1", "1.0")``) and
    ``"v1.1"``, ``"1.1"`` (→ ``("v1.1", "1.1")``). The legacy S3 layout uses
    ``v1/`` for the 1.0 series — `.0` minors are stripped from the path.
    """
    s = spec.strip()
    if s.startswith("v"):
        s = s[1:]
    parts = s.split(".")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else "0"
    norm = f"{major}.{minor}"
    path = f"v{major}" if minor == "0" else f"v{major}.{minor}"
    return path, norm


def _variant_subdir(variant: str) -> str:
    """Map a variant name to its embeddings-dir name on S3."""
    if variant == DEFAULT_VARIANT:
        return EMBEDDINGS_DIR_NAME
    return f"{EMBEDDINGS_DIR_NAME}.{variant}"


def _version_path_from_norm(norm: str) -> str:
    """Inverse of ``_parse_dataset_version`` for the path component.

    ``"1.0"`` → ``"v1"`` (legacy S3 layout uses ``v1/`` for the 1.0 series);
    ``"1.1"`` → ``"v1.1"``; ``"2.0"`` → ``"v2"``; etc.
    """
    major, _, minor = norm.partition(".")
    if not minor or minor == "0":
        return f"v{major}"
    return f"v{major}.{minor}"


# Well-known dataset versions on the public bucket. Used by the client when a
# multi-version operation (e.g. ``coverage --by-source --dataset-version=all``)
# needs to enumerate manifests without a separate listing call. Extend this as
# new versions are published.
KNOWN_VERSIONS = ("v1", "v1.1")

# Sidecar filename written into output directories alongside downloaded tiles
# (NPY format) to record which dataset version/variant produced the files.
TESSERA_METADATA_FILENAME = "tessera_metadata.json"


def write_tessera_metadata(
    output_dir: Union[str, Path],
    dataset_version: str,
    dataset_variant: str,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Write a ``tessera_metadata.json`` sidecar describing a download.

    The local layout always uses ``global_0.1_degree_representation/`` for NPY
    tiles regardless of variant; this sidecar records the provenance so
    downstream tools can recover which dataset was downloaded.
    """
    import json
    from datetime import datetime, timezone

    version_path, version_norm = _parse_dataset_version(dataset_version)
    payload: Dict[str, object] = {
        "dataset_version": version_norm,
        "dataset_version_path": version_path,
        "dataset_variant": dataset_variant,
        "embeddings_subdir": EMBEDDINGS_DIR_NAME,
        "s3_embeddings_subdir": _variant_subdir(dataset_variant),
        "source_url_prefix": (
            f"{TESSERA_BASE_URL}/{version_path}/{_variant_subdir(dataset_variant)}/"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    out = Path(output_dir) / TESSERA_METADATA_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


# ==============================================================================
# COORDINATE SYSTEM HIERARCHY
# ==============================================================================
# This module uses a three-level coordinate hierarchy:
#
# 1. BLOCKS (5×5 degrees): Registry files are organized into blocks for efficient
#    loading. Each block contains up to 2,500 tiles (50×50 grid).
#
# 2. TILES (0.1×0.1 degrees): Individual data files containing embeddings or
#    landmasks. Tiles are centered at 0.05-degree offsets (e.g., 0.05, 0.15, 0.25).
#
# 3. WORLD: Arbitrary decimal degree coordinates provided by users.
#
# Function naming convention:
# - block_* : Operations on 5-degree registry blocks
# - tile_*  : Operations on 0.1-degree data tiles
# - *_from_world : Convert from arbitrary coordinates to block/tile coords
# ==============================================================================


# Block-level functions (5-degree registry organization)
def block_from_world(lon: float, lat: float) -> Tuple[int, int]:
    """Convert world coordinates to containing registry block coordinates.

    Registry blocks are 5×5 degree squares used to organize registry files.
    Each block can contain up to 2,500 tiles.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        tuple: (block_lon, block_lat) lower-left corner of the containing block

    Raises:
        ValueError: If coordinates are out of bounds or not finite

    Examples:
        >>> block_from_world(3.2, 52.7)
        (0, 50)
        >>> block_from_world(-7.8, -23.4)
        (-10, -25)
    """
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} out of bounds [-180, 180]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of bounds [-90, 90]")
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("Coordinates must be finite numbers")

    block_lon = math.floor(lon / BLOCK_SIZE) * BLOCK_SIZE
    block_lat = math.floor(lat / BLOCK_SIZE) * BLOCK_SIZE
    return int(block_lon), int(block_lat)


def block_to_embeddings_registry_filename(
    year: str, block_lon: int, block_lat: int
) -> str:
    """Generate registry filename for an embeddings block.

    Args:
        year: Year string (e.g., "2024")
        block_lon: Block longitude (lower-left corner)
        block_lat: Block latitude (lower-left corner)

    Returns:
        str: Registry filename like "embeddings_2024_lon-55_lat-25.txt"
    """
    # Format longitude and latitude to avoid negative zero
    lon_str = f"lon{block_lon}" if block_lon != 0 else "lon0"
    lat_str = f"lat{block_lat}" if block_lat != 0 else "lat0"
    return f"embeddings_{year}_{lon_str}_{lat_str}.txt"


def block_to_landmasks_registry_filename(block_lon: int, block_lat: int) -> str:
    """Generate registry filename for a landmasks block.

    Args:
        block_lon: Block longitude (lower-left corner)
        block_lat: Block latitude (lower-left corner)

    Returns:
        str: Registry filename like "landmasks_lon-55_lat-25.txt"
    """
    # Format longitude and latitude to avoid negative zero
    lon_str = f"lon{block_lon}" if block_lon != 0 else "lon0"
    lat_str = f"lat{block_lat}" if block_lat != 0 else "lat0"
    return f"landmasks_{lon_str}_{lat_str}.txt"


def blocks_in_bounds(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float
) -> list:
    """Get all registry blocks that intersect with given bounds.

    Args:
        min_lon: Minimum longitude
        max_lon: Maximum longitude
        min_lat: Minimum latitude
        max_lat: Maximum latitude

    Returns:
        list: List of (block_lon, block_lat) tuples

    Raises:
        ValueError: If coordinates are out of bounds, not finite, or min > max
    """
    # Validate longitude bounds
    if not (-180 <= min_lon <= 180):
        raise ValueError(f"Minimum longitude {min_lon} out of bounds [-180, 180]")
    if not (-180 <= max_lon <= 180):
        raise ValueError(f"Maximum longitude {max_lon} out of bounds [-180, 180]")
    if min_lon > max_lon:
        raise ValueError(f"Minimum longitude {min_lon} greater than maximum {max_lon}")

    # Validate latitude bounds
    if not (-90 <= min_lat <= 90):
        raise ValueError(f"Minimum latitude {min_lat} out of bounds [-90, 90]")
    if not (-90 <= max_lat <= 90):
        raise ValueError(f"Maximum latitude {max_lat} out of bounds [-90, 90]")
    if min_lat > max_lat:
        raise ValueError(f"Minimum latitude {min_lat} greater than maximum {max_lat}")

    # Validate all coordinates are finite
    if not all(math.isfinite(x) for x in [min_lon, max_lon, min_lat, max_lat]):
        raise ValueError("All coordinates must be finite numbers")

    blocks = []

    # Get block coordinates for corners
    min_block_lon = math.floor(min_lon / BLOCK_SIZE) * BLOCK_SIZE
    max_block_lon = math.floor(max_lon / BLOCK_SIZE) * BLOCK_SIZE
    min_block_lat = math.floor(min_lat / BLOCK_SIZE) * BLOCK_SIZE
    max_block_lat = math.floor(max_lat / BLOCK_SIZE) * BLOCK_SIZE

    # Iterate through all blocks in range
    lon = min_block_lon
    while lon <= max_block_lon:
        lat = min_block_lat
        while lat <= max_block_lat:
            blocks.append((int(lon), int(lat)))
            lat += BLOCK_SIZE
        lon += BLOCK_SIZE

    return blocks


# Tile-level functions (0.1-degree data tiles)
def tile_from_world(lon: float, lat: float) -> Tuple[float, float]:
    """Convert world coordinates to containing tile center coordinates.

    Tiles are 0.1×0.1 degree squares centered at 0.05-degree offsets
    (e.g., -0.05, 0.05, 0.15, 0.25, etc.).

    Args:
        lon: World longitude in decimal degrees
        lat: World latitude in decimal degrees

    Returns:
        Tuple of (tile_lon, tile_lat) representing the tile center

    Raises:
        ValueError: If coordinates are out of bounds or not finite

    Examples:
        >>> tile_from_world(0.17, 52.23)
        (0.15, 52.25)
        >>> tile_from_world(-0.12, -0.03)
        (-0.15, -0.05)
    """
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} out of bounds [-180, 180]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of bounds [-90, 90]")
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("Coordinates must be finite numbers")

    tile_lon = np.floor(lon * 10) / 10 + 0.05
    tile_lat = np.floor(lat * 10) / 10 + 0.05
    return round(float(tile_lon), 2), round(float(tile_lat), 2)


def parse_grid_name(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract tile coordinates from a grid filename.

    Args:
        filename: Grid filename like "grid_-50.55_-20.65"

    Returns:
        tuple: (lon, lat) as floats, or (None, None) if parsing fails
    """
    match = re.match(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def coord_to_grid_int(coord: float) -> np.int32:
    """Convert a float coordinate to an integer grid index.

    Multiplies by 100 and rounds to get an integer index.
    This is the inverse of grid_int_to_coord().

    Args:
        coord: Longitude or latitude coordinate (e.g., -0.15, 51.55)

    Returns:
        Integer grid index (e.g., -15, 5155) as numpy.int32
    """
    return np.int32(np.round(coord * 100))


def tile_to_grid_name(lon: float, lat: float) -> str:
    """Generate grid name for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        str: Grid name like "grid_-50.55_-20.65"
    """
    return f"grid_{lon:.2f}_{lat:.2f}"


def tile_to_embedding_paths(lon: float, lat: float, year: int) -> Tuple[Path, Path]:
    """Generate embedding and scales file paths for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude
        year: Year of embeddings

    Returns:
        Tuple of (embedding_path, scales_path) as Path objects
    """
    grid_name = tile_to_grid_name(lon, lat)
    embedding_path = Path(str(year)) / grid_name / f"{grid_name}.npy"
    scales_path = Path(str(year)) / grid_name / f"{grid_name}_scales.npy"
    return embedding_path, scales_path


def tile_to_geotiff_path(lon: float, lat: float, year: int) -> Path:
    """Generate GeoTIFF file path for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude
        year: Year of embeddings

    Returns:
        Path: Relative path like "{year}/grid_{lon}_{lat}/grid_{lon}_{lat}_{year}.tiff"
    """
    grid_name = tile_to_grid_name(lon, lat)
    return Path(str(year)) / grid_name / f"{grid_name}_{year}.tiff"


def tile_to_landmask_filename(lon: float, lat: float) -> str:
    """Generate landmask filename for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        Landmask filename like "grid_0.15_52.25.tiff"
    """
    return f"{tile_to_grid_name(lon, lat)}.tiff"


def tile_to_bounds(lon: float, lat: float) -> Tuple[float, float, float, float]:
    """Get geographic bounds for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        Tuple of (west, south, east, north) bounds
    """
    return (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)


def tile_to_box(lon: float, lat: float):
    """Create a Shapely box geometry for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        Shapely box geometry representing the tile bounds
    """
    from shapely.geometry import box

    west, south, east, north = tile_to_bounds(lon, lat)
    return box(west, south, east, north)


# Base URL for Tessera data downloads
TESSERA_BASE_URL = "https://s3.us-west-2.amazonaws.com/tessera-embeddings"

# Directory structure constants (mirrors remote structure)
EMBEDDINGS_DIR_NAME = "global_0.1_degree_representation"  # NPY embeddings and scales
LANDMASKS_DIR_NAME = "global_0.1_degree_tiff_all"  # Landmask TIFFs

# Note: Default manifest URLs are constructed with version in Registry.__init__
# Format: {TESSERA_BASE_URL}/{version}/manifest.parquet


def download_file_to_temp(
    url: str,
    expected_hash: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cache_path: Optional[Path] = None,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    """Download a file from URL with retry logic, optional caching and If-Modified-Since support.

    Args:
        url: URL to download from
        expected_hash: Optional SHA256 hash to verify
        progress_callback: Optional callback(bytes_downloaded, total_bytes, status)
        cache_path: Optional path for caching. If provided, uses If-Modified-Since to avoid redownloading unchanged files.
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Timeout in seconds for each request (default: 60)

    Returns:
        Path to downloaded file (temporary if cache_path=None, otherwise cache_path)
        Caller is responsible for cleanup of temporary files (cache_path=None case)

    Raises:
        URLError: If download fails after all retries
        HTTPError: If server returns error (except 304 Not Modified when using cache, and transient 5xx errors)
        ValueError: If hash verification fails
    """
    import tempfile
    from email.utils import formatdate, parsedate_to_datetime
    from urllib.error import URLError

    # Helper function to execute request with retry logic
    def execute_request_with_retry(request, max_retries, timeout):
        """Execute HTTP request with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(max_retries):
            try:
                return urlopen(request, timeout=timeout)
            except HTTPError as e:
                # Don't retry client errors (4xx) except 429 (rate limit)
                if 400 <= e.code < 500 and e.code != 429:
                    raise
                # Retry on server errors (5xx) and rate limiting (429)
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_time = 2**attempt
                    logging.getLogger(__name__).debug(
                        f"HTTP {e.code} error, retrying in {backoff_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff_time)
            except URLError as e:
                # Retry on network errors
                last_exception = e
                if attempt < max_retries - 1:
                    backoff_time = 2**attempt
                    logging.getLogger(__name__).debug(
                        f"Network error, retrying in {backoff_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff_time)

        # All retries exhausted, raise the last exception
        raise last_exception

    # Handle conditional requests for cached files. We prefer ETag-based
    # validation (If-None-Match) over time-based (If-Modified-Since) because
    # the ETag changes deterministically whenever the S3 object is replaced,
    # whereas Last-Modified can match if the local file's mtime was set from
    # the prior Last-Modified header. The x-amz-checksum-mode opt-in asks S3
    # to return the per-object x-amz-checksum-crc64nvme response header so
    # the downloader can verify content integrity end-to-end.
    headers = {
        "User-Agent": "geotessera",
        "x-amz-checksum-mode": "ENABLED",
    }

    # The ETag sidecar lives next to the cached file. Tiny (~50 bytes) so
    # we don't bother with xattrs or a central manifest.
    etag_sidecar = None
    if cache_path:
        etag_sidecar = cache_path.with_suffix(cache_path.suffix + ".etag")

    if cache_path and cache_path.exists():
        # Prefer If-None-Match (ETag) over If-Modified-Since when we have a
        # stored ETag — it's clock- and mtime-agnostic.
        if etag_sidecar and etag_sidecar.exists():
            try:
                stored_etag = etag_sidecar.read_text().strip()
                if stored_etag:
                    headers["If-None-Match"] = stored_etag
            except OSError:
                pass

        # Belt-and-braces: also send If-Modified-Since so the server can use
        # whichever validator it has indexed.
        cache_mtime = cache_path.stat().st_mtime
        if_modified_since = formatdate(cache_mtime, usegmt=True)
        headers["If-Modified-Since"] = if_modified_since

        # Make conditional request
        request = Request(url, headers=headers)

        try:
            response = execute_request_with_retry(request, max_retries, timeout)
            # 200 OK means file was modified, proceed with download
        except HTTPError as e:
            if e.code == 304:
                # 304 Not Modified - use cached version
                if progress_callback:
                    progress_callback(0, 0, "Cache is current")
                return str(cache_path)
            else:
                # Other HTTP errors should be raised
                raise
    else:
        # No cache or cache_path not provided - regular download
        request = Request(url, headers=headers)
        response = execute_request_with_retry(request, max_retries, timeout)

    # Determine output path
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            delete=False,
            prefix=f".{cache_path.name}_tmp_",
            suffix=cache_path.suffix,
        )
    else:
        temp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".npy")

    temp_path = Path(temp_file.name)

    try:
        with response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            start_time = time.time()
            last_update_time = start_time

            # Compute CRC64NVMe incrementally for server-side checksum verification.
            # S3 returns x-amz-checksum-crc64nvme as a FULL_OBJECT base64-encoded CRC
            # on every GET/HEAD when the object was uploaded with that algorithm.
            expected_crc64nvme_b64 = response.headers.get("x-amz-checksum-crc64nvme")
            crc64nvme = CrtCrc64NvmeChecksum() if expected_crc64nvme_b64 else None

            # Format file size for display
            def format_bytes(bytes_val):
                """Format bytes as human-readable string."""
                for unit in ["B", "KB", "MB", "GB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.1f}{unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.1f}TB"

            if progress_callback:
                size_str = (
                    format_bytes(total_size) if total_size > 0 else "unknown size"
                )
                progress_callback(0, total_size, f"Starting ({size_str})")

            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                temp_file.write(chunk)
                if crc64nvme is not None:
                    crc64nvme.update(chunk)
                downloaded += len(chunk)

                if progress_callback and total_size > 0:
                    current_time = time.time()
                    # Update progress with speed info every ~100ms or on significant progress
                    if (
                        current_time - last_update_time > 0.1
                        or downloaded == total_size
                    ):
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            speed = downloaded / elapsed
                            speed_str = format_bytes(speed) + "/s"
                            downloaded_str = format_bytes(downloaded)
                            total_str = format_bytes(total_size)
                            status = f"{downloaded_str}/{total_str} @ {speed_str}"
                        else:
                            downloaded_str = format_bytes(downloaded)
                            total_str = format_bytes(total_size)
                            status = f"{downloaded_str}/{total_str}"

                        progress_callback(downloaded, total_size, status)
                        last_update_time = current_time

        temp_file.close()

        # Verify S3 server-side CRC64NVMe checksum if the server advertised one.
        if crc64nvme is not None:
            if progress_callback:
                progress_callback(downloaded, downloaded, "Verifying CRC64NVMe...")
            import base64

            actual_crc64nvme_b64 = base64.b64encode(crc64nvme.digest()).decode()
            if actual_crc64nvme_b64 != expected_crc64nvme_b64:
                temp_path.unlink()
                raise ValueError(
                    f"CRC64NVMe mismatch: expected {expected_crc64nvme_b64}, "
                    f"got {actual_crc64nvme_b64}"
                )

        # Verify hash if provided
        if expected_hash:
            if progress_callback:
                progress_callback(downloaded, downloaded, "Verifying hash...")
            actual_hash = calculate_file_hash(temp_path)
            if actual_hash != expected_hash:
                temp_path.unlink()
                raise ValueError(
                    f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
                )

        # Set file mtime from Last-Modified header if available
        last_modified_str = response.headers.get("Last-Modified")
        if last_modified_str:
            try:
                last_modified_dt = parsedate_to_datetime(last_modified_str)
                last_modified_timestamp = last_modified_dt.timestamp()
                os.utime(temp_path, (last_modified_timestamp, last_modified_timestamp))
            except (ValueError, TypeError) as e:
                # Parsing errors - invalid date format
                logging.getLogger(__name__).debug(
                    f"Could not parse Last-Modified header: {e}"
                )
            except OSError as e:
                # Filesystem errors - permissions, disk full, etc.
                logging.getLogger(__name__).warning(f"Could not set file mtime: {e}")

        # If caching, atomically move to cache location and record the new
        # ETag alongside it so the next request can use If-None-Match.
        if cache_path:
            temp_path.rename(cache_path)
            final_path = cache_path
            if etag_sidecar is not None:
                new_etag = response.headers.get("ETag")
                try:
                    if new_etag:
                        etag_sidecar.write_text(new_etag)
                    elif etag_sidecar.exists():
                        # Server didn't return an ETag this time — drop the
                        # stale sidecar rather than reusing a wrong validator.
                        etag_sidecar.unlink()
                except OSError as e:
                    logging.getLogger(__name__).debug(
                        f"Could not persist ETag sidecar {etag_sidecar}: {e}"
                    )
        else:
            final_path = temp_path

        if progress_callback:
            total_str = format_bytes(downloaded)
            progress_callback(downloaded, downloaded, f"Complete ({total_str})")

        return str(final_path)

    except Exception:
        temp_file.close()
        if temp_path.exists():
            temp_path.unlink()
        raise


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class Registry:
    """Registry management for Tessera data files using Parquet.

    Handles all registry-related operations including:
    - Loading and querying Parquet registry
    - Direct HTTP downloads to embeddings_dir for persistent tile storage
    - Parsing available embeddings and landmasks

    Note: The Parquet registry itself (~few MB) is cached separately from tile data.
    Data tiles are downloaded to embeddings_dir and persist for reuse across sessions.
    """

    def __init__(
        self,
        version: str,
        variant: str = DEFAULT_VARIANT,
        cache_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        registry_url: Optional[str] = None,
        registry_path: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
        landmasks_registry_url: Optional[str] = None,
        landmasks_registry_path: Optional[Union[str, Path]] = None,
        verify_hashes: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Registry manager with optimized Parquet registries.

        Args:
            version: Dataset version. Accepts ``"v1"``/``"1.0"`` or ``"v1.1"``/``"1.1"``.
            variant: Dataset variant to filter the manifest by. The default
                ``"vultr"`` corresponds to the bare
                ``global_0.1_degree_representation`` directory on S3; other
                values (e.g. ``"cambridge"``) map to the ``.<variant>`` suffix.
            cache_dir: Optional directory for caching Parquet registries only (not data files)
            embeddings_dir: Directory for storing embedding tiles (defaults to current directory).
                Expected structure: global_0.1_degree_representation[.<variant>]/{year}/
                grid_{lon}_{lat}.npy and _scales.npy, global_0.1_degree_tiff_all/landmask_{lon}_{lat}.tif.
                Tiles are downloaded here and persist for reuse across sessions.
            registry_url: URL to download embeddings manifest parquet from (default: remote)
            registry_path: Local path to existing embeddings manifest parquet file
            registry_dir: Directory containing manifest.parquet and landmasks.parquet files (alternative to individual paths)
            landmasks_registry_url: URL to download landmasks Parquet registry from (default: remote)
            landmasks_registry_path: Local path to existing landmasks Parquet registry file
            verify_hashes: If True (default), verify SHA256 hashes of downloaded files.
                Set to False to skip hash verification. Can also be disabled via
                GEOTESSERA_SKIP_HASH=1 environment variable.
            logger: Optional logger instance. If not provided, creates a new one
        """
        # Resolve version into S3 path component and normalised numeric form.
        self._version_path, self._version_norm = _parse_dataset_version(version)
        self._variant = variant
        self._embeddings_subdir = _variant_subdir(variant)
        # Preserve the original kwarg for callers that still read .version.
        self.version = self._version_path
        # Public read-only view of the variant-aware subdir name used both in
        # local mirrors and S3 URLs.
        self.embeddings_subdir = self._embeddings_subdir
        self.variant = self._variant
        # Populated by _load_registry() with the local path to the manifest
        # parquet (cache file or user-supplied). Useful for tools that need
        # the raw unfiltered manifest (e.g. multi-source coverage rendering).
        self.manifest_path: Optional[Path] = None
        self.logger = logger or logging.getLogger(__name__)

        # Check environment variable for hash verification override
        env_skip_hash = os.environ.get("GEOTESSERA_SKIP_HASH", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self.verify_hashes = verify_hashes and not env_skip_hash

        if env_skip_hash:
            self.logger.warning(
                "Hash verification disabled via GEOTESSERA_SKIP_HASH environment variable"
            )
        elif not verify_hashes:
            self.logger.warning(
                "Hash verification disabled via verify_hashes parameter"
            )

        # Set up cache directory for Parquet registries only
        if cache_dir:
            self._registry_cache_dir = Path(cache_dir)
        else:
            # Use platform-appropriate cache directory
            if os.name == "nt":
                base = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser()
            else:
                base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            self._registry_cache_dir = base / "geotessera"

        self._registry_cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up embeddings directory for tile storage (defaults to cwd)
        if embeddings_dir:
            self._embeddings_dir = Path(embeddings_dir)
        else:
            self._embeddings_dir = Path.cwd()
            # Use debug level since this is only relevant when actually fetching tiles
            self.logger.debug(
                f"embeddings_dir not specified, using current directory: {self._embeddings_dir}"
            )

        # Handle registry_dir convenience parameter
        if registry_dir:
            registry_dir_path = Path(registry_dir)
            if not registry_path:
                candidate = registry_dir_path / "manifest.parquet"
                if candidate.exists():
                    registry_path = candidate
            if not landmasks_registry_path:
                candidate = registry_dir_path / "landmasks.parquet"
                if candidate.exists():
                    landmasks_registry_path = candidate

        # Embeddings manifest (GeoDataFrame with spatial index). The wire format
        # mirrors the file-scan inventory schema; geometry is derived from
        # lon/lat at load time if the parquet was written as a plain DataFrame.
        # One manifest per version on S3; the consumer fetches the manifest
        # matching its dataset_version and filters by variant on load.
        self._registry_gdf: Optional[gpd.GeoDataFrame] = None
        self._registry_url = (
            registry_url or f"{TESSERA_BASE_URL}/{self._version_path}/manifest.parquet"
        )
        self._registry_path = Path(registry_path) if registry_path else None

        # Landmasks Parquet registry (still per-version on S3).
        self._landmasks_df: Optional[pd.DataFrame] = None
        self._landmasks_registry_url = (
            landmasks_registry_url
            or f"{TESSERA_BASE_URL}/{self._version_path}/landmasks.parquet"
        )
        self._landmasks_registry_path = (
            Path(landmasks_registry_path) if landmasks_registry_path else None
        )

        # Load registries
        self._load_registry()
        self._load_landmasks_registry()

    def _load_registry(self):
        """Load manifest as GeoDataFrame with If-Modified-Since refresh.

        The manifest is a plain parquet using the file-scan inventory schema
        (year, lon, lat, grid_size, scales_size, ...). A Point geometry column
        is materialised from lon/lat at load time so spatial queries via
        GeoPandas's R-tree still work.
        """
        registry_path = None

        if self._registry_path and self._registry_path.exists():
            # Load from local file (no updates check for explicit paths)
            self.logger.info(f"Loading manifest from local file: {self._registry_path}")
            registry_path = self._registry_path
        else:
            # Use cached version with If-Modified-Since to check for updates.
            # Cache layout mirrors the S3 prefix so multiple versions coexist.
            registry_cache_path = (
                self._registry_cache_dir / self._version_path / "manifest.parquet"
            )
            registry_cache_path.parent.mkdir(parents=True, exist_ok=True)

            if registry_cache_path.exists():
                self.logger.info(f"Using cached manifest: {registry_cache_path}")
                # The downloader returns str(cache_path) on both 304 (cached)
                # and 200 (fresh download with atomic rename), so we can't
                # tell which happened from the return value. The ETag sidecar
                # IS rewritten only on a real refresh, so a content change
                # there is the canonical "was refreshed" signal.
                etag_sidecar = registry_cache_path.with_suffix(
                    registry_cache_path.suffix + ".etag"
                )
                pre_etag = (
                    etag_sidecar.read_text() if etag_sidecar.exists() else None
                )
                try:
                    self.logger.info("Checking for manifest updates...")
                    result_path = download_file_to_temp(
                        self._registry_url, cache_path=registry_cache_path
                    )
                    registry_path = Path(result_path)
                    post_etag = (
                        etag_sidecar.read_text() if etag_sidecar.exists() else None
                    )
                    if pre_etag is not None and pre_etag == post_etag:
                        self.logger.info(
                            "Verified with server - manifest is current (no download needed)"
                        )
                    else:
                        self.logger.info("Downloaded updated manifest from server")
                except Exception as e:
                    self.logger.warning(f"Could not check for updates: {e}")
                    self.logger.info("Using existing cached manifest")
                    registry_path = registry_cache_path
            else:
                # Download the manifest to cache for the first time
                self.logger.info(f"Downloading manifest from {self._registry_url}")
                try:
                    result_path = download_file_to_temp(
                        self._registry_url, cache_path=registry_cache_path
                    )
                    registry_path = Path(result_path)
                    self.logger.info("Manifest downloaded successfully")
                except Exception as e:
                    raise RuntimeError(f"Failed to download manifest: {e}") from e

        # Load as plain parquet first; promote to GeoDataFrame if needed.
        try:
            df = pd.read_parquet(registry_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load manifest parquet: {e}") from e

        # Record the on-disk manifest path for tools that need the unfiltered
        # parquet (multi-source coverage rendering, manifest introspection, …).
        self.manifest_path = Path(registry_path)

        self.logger.info(f"Loaded manifest with {len(df):,} tiles")

        # Validate required columns (file-scan inventory schema).
        required_columns = {"lat", "lon", "year", "grid_size"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Manifest is missing required columns: {missing}")

        # Filter to the requested (version, variant) if those columns are present.
        # Missing columns mean a single-(version, variant) manifest; trust it as-is.
        before = len(df)
        if "version" in df.columns:
            df = df[df["version"].astype(str) == self._version_norm]
        if "variant" in df.columns:
            df = df[df["variant"].astype(str) == self._variant]
        if len(df) != before:
            self.logger.info(
                f"Filtered manifest to version={self._version_norm}, "
                f"variant={self._variant}: {len(df):,} tiles (from {before:,})"
            )
        if df.empty:
            raise ValueError(
                f"Manifest has no rows for version={self._version_norm}, "
                f"variant={self._variant}. Check the dataset_version and "
                f"dataset_variant arguments."
            )

        # Defensive dedupe: bucket-side misfiling (e.g. a tile filed under the
        # wrong grid directory) can cause the scanner to emit two rows for the
        # same (year, lon, lat). Duplicates here would make the MultiIndex
        # non-unique and turn .loc[] lookups into multi-row Series, which
        # downstream int(row["grid_size"]) cannot handle.
        before_dedupe = len(df)
        df = df.drop_duplicates(subset=["year", "lon", "lat"], keep="first")
        if len(df) != before_dedupe:
            self.logger.warning(
                f"Dropped {before_dedupe - len(df):,} duplicate (year, lon, lat) "
                "rows from manifest"
            )

        # Materialise geometry from lon/lat for spatial queries (.cx).
        if "geometry" in df.columns:
            self._registry_gdf = gpd.GeoDataFrame(
                df, geometry="geometry", crs="EPSG:4326"
            )
        else:
            geometry = gpd.points_from_xy(df["lon"], df["lat"])
            self._registry_gdf = gpd.GeoDataFrame(
                df, geometry=geometry, crs="EPSG:4326"
            )

        # Ensure lon_i/lat_i columns exist (backwards compat with old parquet files)
        if "lon_i" not in self._registry_gdf.columns:
            self._registry_gdf["lon_i"] = (
                (self._registry_gdf["lon"] * 100).round().astype(np.int32)
            )
        if "lat_i" not in self._registry_gdf.columns:
            self._registry_gdf["lat_i"] = (
                (self._registry_gdf["lat"] * 100).round().astype(np.int32)
            )

        # Set MultiIndex for O(1) lookups via .loc[(year, lon_i, lat_i)]
        self._registry_gdf["year"] = self._registry_gdf["year"].astype(int)
        self._registry_gdf = self._registry_gdf.set_index(["year", "lon_i", "lat_i"])

    def _load_landmasks_registry(self):
        """Load landmasks Parquet registry from local path or download from remote with If-Modified-Since refresh."""
        if self._landmasks_registry_path and self._landmasks_registry_path.exists():
            # Load from local file (no updates check for explicit paths)
            self.logger.info(
                f"Loading landmasks registry from local file: {self._landmasks_registry_path}"
            )
            self._landmasks_df = pd.read_parquet(self._landmasks_registry_path)
        else:
            # Use cached version with If-Modified-Since to check for updates.
            # Cache layout mirrors S3 so multiple versions coexist.
            landmasks_cache_path = (
                self._registry_cache_dir / self._version_path / "landmasks.parquet"
            )
            landmasks_cache_path.parent.mkdir(parents=True, exist_ok=True)

            if landmasks_cache_path.exists():
                self.logger.info(
                    f"Using cached landmasks registry: {landmasks_cache_path}"
                )
                # Compare the ETag sidecar pre/post: only a 200 (real
                # refresh) rewrites it; a 304 leaves it untouched.
                etag_sidecar = landmasks_cache_path.with_suffix(
                    landmasks_cache_path.suffix + ".etag"
                )
                pre_etag = (
                    etag_sidecar.read_text() if etag_sidecar.exists() else None
                )
                try:
                    self.logger.info("Checking for landmasks registry updates...")
                    result_path = download_file_to_temp(
                        self._landmasks_registry_url, cache_path=landmasks_cache_path
                    )
                    landmasks_path = Path(result_path)
                    self._landmasks_df = pd.read_parquet(landmasks_path)
                    post_etag = (
                        etag_sidecar.read_text() if etag_sidecar.exists() else None
                    )
                    if pre_etag is not None and pre_etag == post_etag:
                        self.logger.info(
                            "Verified with server - landmasks registry is current (no download needed)"
                        )
                    else:
                        self.logger.info(
                            "Downloaded updated landmasks registry from server"
                        )
                except Exception as e:
                    # Landmasks are optional, if update check fails use cached version
                    self.logger.warning(f"Could not check for landmasks updates: {e}")
                    self.logger.info("Using existing cached landmasks registry")
                    try:
                        self._landmasks_df = pd.read_parquet(landmasks_cache_path)
                    except (OSError, ValueError, ImportError) as read_error:
                        # OSError: File not readable, ValueError: Invalid parquet, ImportError: Missing deps
                        self.logger.warning(
                            f"Could not read cached landmasks: {read_error}"
                        )
                        self._landmasks_df = None
                        return
            else:
                # Download the landmasks registry to cache for the first time
                self.logger.info(
                    f"Downloading landmasks registry from {self._landmasks_registry_url}"
                )
                try:
                    # Download landmasks registry with caching
                    result_path = download_file_to_temp(
                        self._landmasks_registry_url, cache_path=landmasks_cache_path
                    )
                    landmasks_path = Path(result_path)
                    self._landmasks_df = pd.read_parquet(landmasks_path)
                    self.logger.info("Landmasks registry downloaded successfully")
                except Exception as e:
                    # Landmasks are optional, so just warn instead of failing
                    self.logger.warning(f"Failed to download landmasks registry: {e}")
                    self._landmasks_df = None
                    return

        # Validate landmasks registry structure. Hash columns are no longer
        # required — integrity is verified via the S3 CRC64NVMe header at
        # download time.
        if self._landmasks_df is not None:
            required_columns = {"lat", "lon", "file_size"}
            if not required_columns.issubset(self._landmasks_df.columns):
                missing = required_columns - set(self._landmasks_df.columns)
                self.logger.warning(
                    f"Landmasks registry is missing required columns: {missing}"
                )
                self._landmasks_df = None
            else:
                # Ensure lon_i/lat_i columns exist (backwards compat with old parquet files)
                if "lon_i" not in self._landmasks_df.columns:
                    self._landmasks_df["lon_i"] = (
                        (self._landmasks_df["lon"] * 100).round().astype(np.int32)
                    )
                if "lat_i" not in self._landmasks_df.columns:
                    self._landmasks_df["lat_i"] = (
                        (self._landmasks_df["lat"] * 100).round().astype(np.int32)
                    )

                # Set index for O(1) lookups via .loc[(lon_i, lat_i)]
                self._landmasks_df = self._landmasks_df.set_index(["lon_i", "lat_i"])

    def _lookup_tile(self, year: int, lon: float, lat: float) -> pd.Series:
        """Look up a tile row by year and coordinates.

        Args:
            year: Year of the tile
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            pd.Series with the tile's registry data

        Raises:
            ValueError: If tile not found in registry
        """
        lon_i = int(coord_to_grid_int(lon))
        lat_i = int(coord_to_grid_int(lat))
        try:
            return self._registry_gdf.loc[(int(year), lon_i, lat_i)]
        except KeyError:
            raise ValueError(
                f"Tile not found in registry: year={year}, lon={lon:.2f}, lat={lat:.2f}"
            )

    def _lookup_landmask(self, lon: float, lat: float) -> pd.Series:
        """Look up a landmask row by coordinates.

        Args:
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            pd.Series with the landmask's registry data

        Raises:
            ValueError: If landmask not found in registry
        """
        lon_i = int(coord_to_grid_int(lon))
        lat_i = int(coord_to_grid_int(lat))
        try:
            return self._landmasks_df.loc[(lon_i, lat_i)]
        except KeyError:
            raise ValueError(
                f"Landmask not found in registry: lon={lon:.2f}, lat={lat:.2f}"
            )

    def iter_tiles_in_region(
        self, bounds: Tuple[float, float, float, float], year: int
    ) -> Iterator[Tuple[int, float, float]]:
        """Lazy iterator over tiles in a region using GeoPandas spatial indexing.

        GeoPandas automatically uses R-tree spatial indexing for fast queries.
        This method:
        - Starts yielding immediately (low latency)
        - Uses constant memory regardless of region size
        - Allows early termination without processing all tiles
        - Leverages GeoPandas built-in R-tree for optimal performance

        Note: Registry stores tiles as Point geometries at their centers, but tiles
        are 0.1° x 0.1° boxes. The query is expanded by 0.05° (half tile width) in
        all directions to catch tiles whose boxes intersect the bounds but whose
        centers fall outside.

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load

        Yields:
            Tuples of (year, tile_lon, tile_lat) for each tile in the region

        Example:
            >>> registry = Registry('v1')
            >>> bounds = (-0.2, 51.4, 0.1, 51.6)  # London
            >>> for year, lon, lat in registry.iter_tiles_in_region(bounds, 2024):
            ...     embedding = fetch_embedding(lon, lat, year)
            ...     process(embedding)  # Start processing immediately
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Expand bounds by half a tile width (0.05°) to catch tiles whose boxes
        # intersect the query region. Registry stores tiles as Point geometries at
        # their centers, but tiles are 0.1° x 0.1° boxes, so we need to expand the
        # query to include tiles whose centers are up to 0.05° outside the bounds.
        expansion = 0.05
        tiles = self._registry_gdf.cx[
            min_lon - expansion : max_lon + expansion,
            min_lat - expansion : max_lat + expansion,
        ]

        # Filter by year using index level
        tiles = tiles[tiles.index.get_level_values("year") == year]

        # Yield unique (year, lon_i, lat_i) tuples from the index
        seen = set()
        for idx in tiles.index:
            if idx not in seen:
                seen.add(idx)
                year_val, lon_i, lat_i = idx
                yield (year_val, lon_i / 100.0, lat_i / 100.0)

    def load_blocks_for_region(
        self, bounds: Tuple[float, float, float, float], year: int
    ) -> List[Tuple[int, float, float]]:
        """Load tiles for a region (list-returning version for backward compatibility).

        For memory-efficient streaming, use iter_tiles_in_region() instead.

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load

        Returns:
            List of (year, tile_lon, tile_lat) tuples for tiles in the region
        """
        # Use iterator and materialize to list (vectorized, 10-100x faster than iterrows)
        tiles_list = list(self.iter_tiles_in_region(bounds, year))

        if tiles_list:
            self.logger.info(f"Found {len(tiles_list)} tiles for region in year {year}")

        return tiles_list

    def get_available_years(self) -> List[int]:
        """List all years with available Tessera embeddings.

        Returns:
            List of years with available data, sorted in ascending order.
        """
        return sorted(
            self._registry_gdf.index.get_level_values("year").unique().tolist()
        )

    def get_tile_counts_by_year(self) -> Dict[int, int]:
        """Get count of tiles per year using efficient pandas operations.

        Returns:
            Dictionary mapping year to tile count
        """
        # Count unique (lon_i, lat_i) index pairs per year level
        idx = self._registry_gdf.index
        counts = (
            pd.DataFrame(
                {
                    "year": idx.get_level_values("year"),
                    "lon_i": idx.get_level_values("lon_i"),
                    "lat_i": idx.get_level_values("lat_i"),
                }
            )
            .drop_duplicates()
            .groupby("year")
            .size()
            .to_dict()
        )
        return {int(year): int(count) for year, count in counts.items()}

    def get_available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of all available embeddings with vectorized conversion.

        Returns:
            List of (year, lon, lat) tuples for all available embedding tiles
        """
        # Use unique index tuples directly - already (year, lon_i, lat_i)
        unique_idx = self._registry_gdf.index.unique()
        return [
            (int(year), lon_i / 100.0, lat_i / 100.0)
            for year, lon_i, lat_i in unique_idx
        ]

    def fetch(
        self,
        path: Optional[str] = None,
        progressbar: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        refresh: bool = False,
        year: Optional[int] = None,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        is_scales: bool = False,
    ) -> str:
        """Fetch a file using local embeddings_dir or direct HTTP download.

        Args:
            path: Optional path to the file (relative to base URL or embeddings_dir).
                  If not provided, will be calculated from year/lon/lat.
            progressbar: Whether to show download progress
            progress_callback: Optional callback for progress updates
            refresh: If True, force re-download even if local file exists
            year: Year of the tile (required if path not provided)
            lon: Longitude of the tile (required if path not provided)
            lat: Latitude of the tile (required if path not provided)
            is_scales: If True, fetch scales file instead of embedding file

        Returns:
            Path to the file in embeddings_dir
        """
        # Calculate path from coordinates if not provided
        if path is None:
            if year is None or lon is None or lat is None:
                raise ValueError(
                    "Must provide either 'path' or all of (year, lon, lat)"
                )
            embedding_path, scales_path = tile_to_embedding_paths(lon, lat, year)
            path = scales_path if is_scales else embedding_path

        # Local layout always uses the bare ``global_0.1_degree_representation``
        # subdir regardless of variant; variant/version provenance lives in the
        # ``tessera_metadata.json`` sidecar written by the CLI download flow.
        local_path = self._embeddings_dir / EMBEDDINGS_DIR_NAME / path

        # Check if file exists locally and not refreshing
        if local_path.exists() and not refresh:
            # Use existing local file
            return str(local_path)

        # Download to embeddings_dir. Integrity is verified end-to-end against
        # the S3 x-amz-checksum-crc64nvme response header inside the downloader.
        # Use as_posix() to ensure forward slashes in URL even on Windows
        path_str = path.as_posix() if isinstance(path, Path) else path
        url = (
            f"{TESSERA_BASE_URL}/{self._version_path}/"
            f"{self._embeddings_subdir}/{path_str}"
        )
        downloaded_path = download_file_to_temp(
            url,
            progress_callback=progress_callback,
            cache_path=local_path,
        )

        # Return path to saved file
        return downloaded_path

    def fetch_landmask(
        self,
        filename: Optional[str] = None,
        progressbar: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        refresh: bool = False,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
    ) -> str:
        """Fetch a landmask file using local embeddings_dir or direct HTTP download.

        Args:
            filename: Optional name of the landmask file. If not provided, will be
                      calculated from lon/lat.
            progressbar: Whether to show download progress
            progress_callback: Optional callback for progress updates
            refresh: If True, force re-download even if local file exists
            lon: Longitude of the tile (required if filename not provided)
            lat: Latitude of the tile (required if filename not provided)

        Returns:
            Path to the file in embeddings_dir
        """
        # Calculate filename from coordinates if not provided
        if filename is None:
            if lon is None or lat is None:
                raise ValueError("Must provide either 'filename' or both (lon, lat)")
            filename = tile_to_landmask_filename(lon, lat)

        # Determine local file path
        local_path = self._embeddings_dir / LANDMASKS_DIR_NAME / filename

        # Check if file exists locally and not refreshing
        if local_path.exists() and not refresh:
            # Use existing local file
            return str(local_path)

        # Download to embeddings_dir. Integrity is verified end-to-end against
        # the S3 x-amz-checksum-crc64nvme response header inside the downloader.
        url = f"{TESSERA_BASE_URL}/{self._version_path}/{LANDMASKS_DIR_NAME}/{filename}"
        downloaded_path = download_file_to_temp(
            url,
            progress_callback=progress_callback,
            cache_path=local_path,
        )

        # Return path to saved file
        return downloaded_path

    @property
    def available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of available embeddings."""
        return self.get_available_embeddings()

    def get_landmask_count(self) -> int:
        """Get count of unique landmask tiles using efficient pandas operations.

        Returns:
            Count of unique landmask tiles
        """
        if self._landmasks_df is not None:
            # Count unique (lon_i, lat_i) index pairs in landmasks registry
            return len(self._landmasks_df.index.unique())

        # Fallback: count unique tiles in embeddings registry
        idx = self._registry_gdf.index.droplevel("year")
        return len(idx.unique())

    @property
    def available_landmasks(self) -> List[Tuple[float, float]]:
        """Get list of available landmasks with vectorized conversion.

        Falls back to embedding tiles if landmasks registry is not available.

        Note: For performance, use get_landmask_count() if you only need the count.
        """
        # Use landmasks registry if available
        if self._landmasks_df is not None:
            unique_idx = self._landmasks_df.index.unique()
            return [(lon_i / 100.0, lat_i / 100.0) for lon_i, lat_i in unique_idx]

        # Fallback: assume landmasks are available for all embedding tiles
        unique_idx = self._registry_gdf.index.droplevel("year").unique()
        return [(lon_i / 100.0, lat_i / 100.0) for lon_i, lat_i in unique_idx]

    def get_manifest_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get manifest information (git hash and repo URL).

        For Parquet registries, this information is not stored in the registry.
        Returns empty values for API compatibility.

        Returns:
            Tuple of (git_hash, repo_url) - both None for Parquet registries
        """
        return None, None

    def get_tile_file_size(self, year: int, lon: float, lat: float) -> int:
        """Get the file size of an embedding tile from the registry.

        Args:
            year: Year of the tile
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            File size in bytes

        Raises:
            ValueError: If tile not found in registry or file_size column missing
        """
        if "grid_size" not in self._registry_gdf.columns:
            raise ValueError(
                "Manifest is missing 'grid_size' column. "
                "Please update your manifest to include file size metadata."
            )

        row = self._lookup_tile(year, lon, lat)
        return int(row["grid_size"])

    def get_scales_file_size(self, year: int, lon: float, lat: float) -> int:
        """Get the file size of a scales file from the registry.

        Args:
            year: Year of the tile
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            File size in bytes

        Raises:
            ValueError: If tile not found in registry or scales_size column missing
        """
        if "scales_size" not in self._registry_gdf.columns:
            raise ValueError(
                "Registry is missing 'scales_size' column. "
                "Please update your registry to include scales file size metadata."
            )

        row = self._lookup_tile(year, lon, lat)
        return int(row["scales_size"])

    def get_landmask_file_size(self, lon: float, lat: float) -> int:
        """Get the file size of a landmask tile from the registry.

        Args:
            lon: Longitude of the tile center
            lat: Latitude of the tile center

        Returns:
            File size in bytes

        Raises:
            ValueError: If landmask not found in registry or file_size column missing
        """
        if self._landmasks_df is None:
            raise ValueError(
                "Landmasks registry is not loaded. "
                "Please ensure landmasks.parquet is available."
            )

        if "file_size" not in self._landmasks_df.columns:
            raise ValueError(
                "Landmasks registry is missing 'file_size' column. "
                "Please update your landmasks registry to include file size metadata."
            )

        row = self._lookup_landmask(lon, lat)
        return int(row["file_size"])

    def calculate_download_requirements(
        self,
        tiles: List[Tuple[int, float, float]],
        output_dir: Path,
        format_type: str,
        check_existing: bool = True,
    ) -> Tuple[int, int, Dict[str, int]]:
        """Calculate download requirements for a set of tiles.

        Args:
            tiles: List of (year, lon, lat) tuples
            output_dir: Output directory where files would be downloaded
            format_type: Either 'npy' or 'tiff'
            check_existing: If True, skip files that already exist (for resume).
                           If False, calculate as if downloading all files (for dry-run estimates).

        Returns:
            Tuple of (total_bytes, total_files, file_sizes_dict)
            - total_bytes: Total download size in bytes
            - total_files: Number of files to download
            - file_sizes_dict: Dictionary mapping file keys to sizes (for NPY format tracking)

        Raises:
            ValueError: If registry is missing required columns or tiles not found
        """
        total_bytes = 0
        total_files = 0
        file_sizes = {}  # For NPY format: cache file sizes by key

        if format_type == "npy":
            # For NPY format: embedding + scales + landmask per tile
            for tile_year, tile_lon, tile_lat in tiles:
                # Use tile_to_embedding_paths for correct directory structure
                embedding_rel, scales_rel = tile_to_embedding_paths(
                    tile_lon, tile_lat, tile_year
                )
                embedding_final = output_dir / EMBEDDINGS_DIR_NAME / embedding_rel
                scales_final = output_dir / EMBEDDINGS_DIR_NAME / scales_rel
                landmask_final = (
                    output_dir
                    / LANDMASKS_DIR_NAME
                    / tile_to_landmask_filename(tile_lon, tile_lat)
                )

                # Create cache keys for tracking file sizes
                embedding_key = f"embedding_{tile_year}_{tile_lon}_{tile_lat}"
                scales_key = f"scales_{tile_year}_{tile_lon}_{tile_lat}"
                landmask_key = f"landmask_{tile_lon}_{tile_lat}"

                # Only count files that need downloading
                if not check_existing or not embedding_final.exists():
                    size = self.get_tile_file_size(tile_year, tile_lon, tile_lat)
                    file_sizes[embedding_key] = size
                    total_bytes += size
                    total_files += 1

                if not check_existing or not scales_final.exists():
                    # Get actual scales file size from registry
                    size = self.get_scales_file_size(tile_year, tile_lon, tile_lat)
                    file_sizes[scales_key] = size
                    total_bytes += size
                    total_files += 1

                if not check_existing or not landmask_final.exists():
                    size = self.get_landmask_file_size(tile_lon, tile_lat)
                    file_sizes[landmask_key] = size
                    total_bytes += size
                    total_files += 1
        else:
            # For TIFF format: one GeoTIFF per tile
            # TIFF files will be larger than NPY due to dequantization (int8 -> float32)
            # and additional metadata. Estimate as 4x the size of quantized embedding.
            for tile_year, tile_lon, tile_lat in tiles:
                embedding_size = self.get_tile_file_size(tile_year, tile_lon, tile_lat)
                landmask_size = self.get_landmask_file_size(tile_lon, tile_lat)
                # Estimate TIFF size: 4x embedding (float32 vs int8) + landmask overhead
                tiff_size = (embedding_size * 4) + landmask_size
                total_bytes += tiff_size
                total_files += 1

        return total_bytes, total_files, file_sizes
