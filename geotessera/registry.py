"""Registry management for Tessera data files.

This module handles all registry-related operations including loading and querying
the Parquet registry, and direct HTTP downloads with local caching.

Also includes utilities for block-based registry management, organizing global grid
data into 5x5 degree blocks for efficient data access.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import os
import math
import re
import numpy as np
import logging
import hashlib
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required for registry operations")

# Constants for block-based registry management
BLOCK_SIZE = 5  # 5x5 degree blocks

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

    Examples:
        >>> block_from_world(3.2, 52.7)
        (0, 50)
        >>> block_from_world(-7.8, -23.4)
        (-10, -25)
    """
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
    """
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

    Examples:
        >>> tile_from_world(0.17, 52.23)
        (0.15, 52.25)
        >>> tile_from_world(-0.12, -0.03)
        (-0.15, -0.05)
    """
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


def tile_to_grid_name(lon: float, lat: float) -> str:
    """Generate grid name for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude

    Returns:
        str: Grid name like "grid_-50.55_-20.65"
    """
    return f"grid_{lon:.2f}_{lat:.2f}"


def tile_to_embedding_paths(lon: float, lat: float, year: int) -> Tuple[str, str]:
    """Generate embedding and scales file paths for a tile.

    Args:
        lon: Tile center longitude
        lat: Tile center latitude
        year: Year of embeddings

    Returns:
        Tuple of (embedding_path, scales_path)
    """
    grid_name = tile_to_grid_name(lon, lat)
    embedding_path = f"{year}/{grid_name}/{grid_name}.npy"
    scales_path = f"{year}/{grid_name}/{grid_name}_scales.npy"
    return embedding_path, scales_path


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
TESSERA_BASE_URL = "https://dl.geotessera.org"

# Default Parquet registry URLs
DEFAULT_REGISTRY_URL = f"{TESSERA_BASE_URL}/registry.parquet"
DEFAULT_LANDMASKS_REGISTRY_URL = f"{TESSERA_BASE_URL}/landmasks.parquet"


def download_file_to_temp(url: str, expected_hash: Optional[str] = None, progress_callback: Optional[callable] = None) -> str:
    """Download a file from URL to a temporary file with optional hash verification.

    Args:
        url: URL to download from
        expected_hash: Optional SHA256 hash to verify
        progress_callback: Optional callback(bytes_downloaded, total_bytes, status)

    Returns:
        Path to downloaded temporary file (caller is responsible for cleanup)

    Raises:
        URLError: If download fails
        ValueError: If hash verification fails
    """
    import tempfile

    # Create a temporary file that won't be automatically deleted
    temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.npy')
    temp_path = Path(temp_file.name)

    try:
        request = Request(url, headers={'User-Agent': 'geotessera'})

        with urlopen(request) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0

            if progress_callback:
                progress_callback(0, total_size, "Starting download")

            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                temp_file.write(chunk)
                downloaded += len(chunk)

                if progress_callback and total_size > 0:
                    progress_callback(downloaded, total_size, "Downloading")

        temp_file.close()

        # Verify hash if provided
        if expected_hash:
            actual_hash = calculate_file_hash(temp_path)
            if actual_hash != expected_hash:
                temp_path.unlink()
                raise ValueError(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")

        if progress_callback:
            progress_callback(downloaded, downloaded, "Complete")

        return str(temp_path)

    except Exception as e:
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
    - Direct HTTP downloads to temporary files (no persistent caching of data tiles)
    - Parsing available embeddings and landmasks

    Note: Only the Parquet registry itself is cached (~few MB). Data tiles are
    downloaded to temporary files and immediately cleaned up after use, resulting
    in zero persistent storage overhead for embedding data.
    """

    def __init__(
        self,
        version: str,
        cache_dir: Optional[Union[str, Path]] = None,
        registry_url: Optional[str] = None,
        registry_path: Optional[Union[str, Path]] = None,
        landmasks_registry_url: Optional[str] = None,
        landmasks_registry_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize Registry manager with Parquet registries.

        Args:
            version: Dataset version identifier
            cache_dir: Optional directory for caching Parquet registries only (not data files)
            registry_url: URL to download embeddings Parquet registry from (default: remote)
            registry_path: Local path to existing embeddings Parquet registry file
            landmasks_registry_url: URL to download landmasks Parquet registry from (default: remote)
            landmasks_registry_path: Local path to existing landmasks Parquet registry file
        """
        self.version = version

        # Set up cache directory for Parquet registries only
        if cache_dir:
            self._registry_cache_dir = Path(cache_dir)
        else:
            # Use platform-appropriate cache directory
            if os.name == 'nt':
                base = Path(os.environ.get('LOCALAPPDATA', '~')).expanduser()
            else:
                base = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')).expanduser()
            self._registry_cache_dir = base / 'geotessera'

        self._registry_cache_dir.mkdir(parents=True, exist_ok=True)

        # Embeddings Parquet registry
        self._registry_df: Optional[pd.DataFrame] = None
        self._registry_url = registry_url or DEFAULT_REGISTRY_URL
        self._registry_path = Path(registry_path) if registry_path else None

        # Landmasks Parquet registry
        self._landmasks_df: Optional[pd.DataFrame] = None
        self._landmasks_registry_url = landmasks_registry_url or DEFAULT_LANDMASKS_REGISTRY_URL
        self._landmasks_registry_path = Path(landmasks_registry_path) if landmasks_registry_path else None

        # Initialize registries
        self._load_registry()
        self._load_landmasks_registry()

    def _load_registry(self):
        """Load Parquet registry from local path or download from remote."""
        if self._registry_path and self._registry_path.exists():
            # Load from local file
            print(f"Loading registry from local file: {self._registry_path}")
            self._registry_df = pd.read_parquet(self._registry_path)
        else:
            # Check if we have a cached version of the registry
            registry_cache_path = self._registry_cache_dir / "registry.parquet"

            if registry_cache_path.exists():
                print(f"Using cached registry: {registry_cache_path}")
                self._registry_df = pd.read_parquet(registry_cache_path)
            else:
                # Download the registry to cache (registry is small, so we cache it)
                print(f"Downloading registry from {self._registry_url}")
                try:
                    import tempfile
                    # Download registry using the download function
                    temp_path = download_file_to_temp(self._registry_url)

                    # Move to cache location
                    Path(temp_path).rename(registry_cache_path)

                    self._registry_df = pd.read_parquet(registry_cache_path)
                    print("✓ Registry downloaded and loaded successfully")
                except Exception as e:
                    raise RuntimeError(f"Failed to download registry: {e}") from e

        # Validate registry structure
        required_columns = {'lat', 'lon', 'year', 'hash', 'file_path'}
        if not required_columns.issubset(self._registry_df.columns):
            missing = required_columns - set(self._registry_df.columns)
            raise ValueError(f"Registry is missing required columns: {missing}")

    def _load_landmasks_registry(self):
        """Load landmasks Parquet registry from local path or download from remote."""
        if self._landmasks_registry_path and self._landmasks_registry_path.exists():
            # Load from local file
            print(f"Loading landmasks registry from local file: {self._landmasks_registry_path}")
            self._landmasks_df = pd.read_parquet(self._landmasks_registry_path)
        else:
            # Check if we have a cached version of the landmasks registry
            landmasks_cache_path = self._registry_cache_dir / "landmasks.parquet"

            if landmasks_cache_path.exists():
                print(f"Using cached landmasks registry: {landmasks_cache_path}")
                self._landmasks_df = pd.read_parquet(landmasks_cache_path)
            else:
                # Download the landmasks registry to cache (registry is small, so we cache it)
                print(f"Downloading landmasks registry from {self._landmasks_registry_url}")
                try:
                    import tempfile
                    # Download registry using the download function
                    temp_path = download_file_to_temp(self._landmasks_registry_url)

                    # Move to cache location
                    Path(temp_path).rename(landmasks_cache_path)

                    self._landmasks_df = pd.read_parquet(landmasks_cache_path)
                    print("✓ Landmasks registry downloaded and loaded successfully")
                except Exception as e:
                    # Landmasks are optional, so just warn instead of failing
                    print(f"Warning: Failed to download landmasks registry: {e}")
                    self._landmasks_df = None
                    return

        # Validate landmasks registry structure
        if self._landmasks_df is not None:
            required_columns = {'lat', 'lon', 'hash', 'file_path'}
            if not required_columns.issubset(self._landmasks_df.columns):
                missing = required_columns - set(self._landmasks_df.columns)
                print(f"Warning: Landmasks registry is missing required columns: {missing}")
                self._landmasks_df = None

    def ensure_block_loaded(self, year: int, lon: float, lat: float):
        """No-op for Parquet registry - all data is already loaded.

        This method is kept for API compatibility but does nothing since
        the Parquet registry loads all metadata at initialization.

        Args:
            year: Year to load (e.g., 2024)
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
        """
        pass  # All data is already loaded in Parquet

    def ensure_tile_block_loaded(self, lon: float, lat: float):
        """No-op for Parquet registry - all data is already loaded.

        This method is kept for API compatibility but does nothing since
        the Parquet registry loads all metadata at initialization.

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
        """
        pass  # All data is already loaded in Parquet

    def load_blocks_for_region(
        self, bounds: Tuple[float, float, float, float], year: int
    ) -> List[Tuple[float, float]]:
        """Load tiles for a specific region from the Parquet registry.

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load

        Returns:
            List of (tile_lon, tile_lat) tuples for tiles available in the region
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Query Parquet dataframe for tiles in the region
        mask = (
            (self._registry_df['year'] == year) &
            (self._registry_df['lon'] >= min_lon - 0.05) &
            (self._registry_df['lon'] <= max_lon + 0.05) &
            (self._registry_df['lat'] >= min_lat - 0.05) &
            (self._registry_df['lat'] <= max_lat + 0.05)
        )

        tiles = self._registry_df[mask][['lon', 'lat']].drop_duplicates()
        tiles_list = [(row['lon'], row['lat']) for _, row in tiles.iterrows()]

        print(f"Found {len(tiles_list)} tiles for region in year {year}")
        return tiles_list

    def get_available_years(self) -> List[int]:
        """List all years with available Tessera embeddings.

        Returns:
            List of years with available data, sorted in ascending order.
        """
        return sorted(self._registry_df['year'].unique().tolist())

    def get_available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of all available embeddings as (year, lon, lat) tuples.

        Returns:
            List of (year, lon, lat) tuples for all available embedding tiles
        """
        result = []
        for _, row in self._registry_df[['year', 'lon', 'lat']].drop_duplicates().iterrows():
            result.append((int(row['year']), float(row['lon']), float(row['lat'])))
        return result

    def fetch(
        self,
        path: str,
        progressbar: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Fetch a file using direct HTTP download to a temporary file.

        Args:
            path: Path to the file (relative to base URL)
            progressbar: Whether to show download progress
            progress_callback: Optional callback for progress updates

        Returns:
            Local file path to temporary file (caller is responsible for cleanup)
        """
        # Look up file hash from registry if it's a tracked file
        file_hash = None
        if path in self._registry_df['file_path'].values:
            file_hash = self._registry_df[self._registry_df['file_path'] == path]['hash'].iloc[0]

        # Download the file to a temporary location
        url = f"{TESSERA_BASE_URL}/{self.version}/global_0.1_degree_representation/{path}"

        return download_file_to_temp(url, expected_hash=file_hash, progress_callback=progress_callback)

    def fetch_landmask(
        self,
        filename: str,
        progressbar: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Fetch a landmask file using direct HTTP download to a temporary file.

        Args:
            filename: Name of the landmask file
            progressbar: Whether to show download progress
            progress_callback: Optional callback for progress updates

        Returns:
            Local file path to temporary file (caller is responsible for cleanup)
        """
        # Look up file hash from landmasks registry if available
        file_hash = None
        if self._landmasks_df is not None and filename in self._landmasks_df['file_path'].values:
            file_hash = self._landmasks_df[self._landmasks_df['file_path'] == filename]['hash'].iloc[0]

        # Download the file to a temporary location
        url = f"{TESSERA_BASE_URL}/{self.version}/global_0.1_degree_tiff_all/{filename}"

        return download_file_to_temp(url, expected_hash=file_hash, progress_callback=progress_callback)

    @property
    def available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of available embeddings."""
        return self.get_available_embeddings()

    @property
    def available_landmasks(self) -> List[Tuple[float, float]]:
        """Get list of available landmasks from the landmasks Parquet registry.

        Falls back to embedding tiles if landmasks registry is not available.
        """
        # Use landmasks registry if available
        if self._landmasks_df is not None:
            result = []
            for _, row in self._landmasks_df[['lon', 'lat']].drop_duplicates().iterrows():
                result.append((float(row['lon']), float(row['lat'])))
            return result

        # Fallback: assume landmasks are available for all embedding tiles
        result = []
        for _, row in self._registry_df[['lon', 'lat']].drop_duplicates().iterrows():
            result.append((float(row['lon']), float(row['lat'])))
        return result

    @property
    def loaded_blocks(self) -> set:
        """Get set of loaded embedding blocks (all blocks for Parquet)."""
        # Return all unique (year, block_lon, block_lat) combinations
        blocks = set()
        for _, row in self._registry_df[['year', 'lon', 'lat']].iterrows():
            block_lon, block_lat = block_from_world(row['lon'], row['lat'])
            blocks.add((int(row['year']), block_lon, block_lat))
        return blocks
