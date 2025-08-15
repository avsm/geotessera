"""Registry management for Tessera data files.

This module handles all registry-related operations including loading and managing
registry files (local or remote), block-based lazy loading of registry data,
parsing available embeddings and landmasks, and managing pooch instances for data fetching.

Also includes utilities for block-based registry management, organizing global grid
data into 5x5 degree blocks for efficient data access.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import os
import subprocess
import math
import re
import pooch
import numpy as np


# Constants for block-based registry management
BLOCK_SIZE = 5  # 5x5 degree blocks


# Registry utility functions
def get_block_coordinates(lon: float, lat: float) -> Tuple[int, int]:
    """Calculate the block coordinates for a given longitude and latitude.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        tuple: (block_lon, block_lat) representing the lower-left corner of the block
    """
    block_lon = math.floor(lon / BLOCK_SIZE) * BLOCK_SIZE
    block_lat = math.floor(lat / BLOCK_SIZE) * BLOCK_SIZE
    return int(block_lon), int(block_lat)


def get_embeddings_registry_filename(year: str, block_lon: int, block_lat: int) -> str:
    """Generate the registry filename for a specific embeddings block.

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


def get_landmasks_registry_filename(block_lon: int, block_lat: int) -> str:
    """Generate the registry filename for a specific landmask tiles block.

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


def get_all_blocks_in_range(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float
) -> list:
    """Get all block coordinates that intersect with a given bounding box.

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


# Tile coordinate utility functions
def parse_grid_coordinates(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract longitude and latitude from a grid filename.

    Args:
        filename: Grid filename like "grid_-50.55_-20.65"

    Returns:
        tuple: (lon, lat) as floats, or (None, None) if parsing fails
    """
    match = re.match(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def get_tile_name(lon: float, lat: float) -> str:
    """Generate the tile name for a specific coordinate.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        str: Tile name like "grid_-50.55_-20.65"
    """
    return f"grid_{lon:.2f}_{lat:.2f}"


def world_to_tile_coords(lat: float, lon: float) -> Tuple[float, float]:
    """Convert world coordinates to tile center coordinates.
    
    Tessera tiles are on a 0.1-degree grid with centers at 0.05-degree offsets
    (e.g., -0.05, 0.05, 0.15, 0.25, etc.).
    
    This fixes the bug from issue #10 where coordinates were incorrectly calculated.
    
    Args:
        lat: World latitude in decimal degrees
        lon: World longitude in decimal degrees
    
    Returns:
        Tuple of (tile_lat, tile_lon) representing the tile center coordinates
        
    Examples:
        >>> world_to_tile_coords(52.23, 0.17)
        (52.25, 0.15)
        >>> world_to_tile_coords(-0.03, -0.12)
        (-0.05, -0.15)
    """
    # Fix for issue #10: tiles are offset by 0.05 degrees from the 0.1-degree grid
    tile_lat = np.floor(lat * 10) / 10 + 0.05
    tile_lon = np.floor(lon * 10) / 10 + 0.05
    return tile_lat, tile_lon


def tile_to_embedding_path(lat: float, lon: float, year: int) -> Tuple[str, str]:
    """Generate embedding and scales file paths for a tile.
    
    Args:
        lat: Tile center latitude
        lon: Tile center longitude
        year: Year of embeddings
        
    Returns:
        Tuple of (embedding_path, scales_path)
    """
    grid_name = get_tile_name(lon, lat)  # Note: get_tile_name uses (lon, lat) order
    embedding_path = f"{year}/{grid_name}/{grid_name}.npy"
    scales_path = f"{year}/{grid_name}/{grid_name}_scales.npy"
    return embedding_path, scales_path


def tile_to_landmask_filename(lat: float, lon: float) -> str:
    """Generate landmask filename for a tile.
    
    Args:
        lat: Tile center latitude
        lon: Tile center longitude
        
    Returns:
        Landmask filename like "grid_0.15_52.25.tiff"
    """
    return f"{get_tile_name(lon, lat)}.tiff"  # Note: get_tile_name uses (lon, lat) order


def get_tile_bounds(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """Get geographic bounds for a tile.
    
    Args:
        lat: Tile center latitude
        lon: Tile center longitude
        
    Returns:
        Tuple of (west, south, east, north) bounds
    """
    return (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)


def get_tile_box(lat: float, lon: float):
    """Create a Shapely box geometry for a tile.
    
    Args:
        lat: Tile center latitude
        lon: Tile center longitude
        
    Returns:
        Shapely box geometry representing the tile bounds
    """
    from shapely.geometry import box
    west, south, east, north = get_tile_bounds(lat, lon)
    return box(west, south, east, north)


def get_registry_path_for_tile(
    registry_base_dir: str, year: str, lon: float, lat: float
) -> str:
    """Get the full path to the registry file containing a specific tile.

    Args:
        registry_base_dir: Base directory for registry files (should be the parent directory)
        year: Year string (e.g., "2024")
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        str: Full path to the registry file
    """
    block_lon, block_lat = get_block_coordinates(lon, lat)
    registry_filename = get_embeddings_registry_filename(year, block_lon, block_lat)
    return os.path.join(registry_base_dir, "registry", registry_filename)


def get_registry_path_for_tiles(registry_base_dir: str, lon: float, lat: float) -> str:
    """Get the full path to the tiles registry file containing a specific coordinate.

    Args:
        registry_base_dir: Base directory for registry files (should be the parent directory)
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        str: Full path to the tiles registry file
    """
    block_lon, block_lat = get_block_coordinates(lon, lat)
    registry_filename = get_landmasks_registry_filename(block_lon, block_lat)
    return os.path.join(registry_base_dir, "registry", registry_filename)


# Base URL for Tessera data downloads
TESSERA_BASE_URL = "https://dl-2.tessera.wiki"


class Registry:
    """Registry management for Tessera data files.
    
    Handles all registry-related operations including:
    - Loading and managing registry files (local or remote)
    - Block-based lazy loading of registry data
    - Parsing available embeddings and landmasks
    - Managing pooch instances for data fetching
    """
    
    def __init__(self, version: str, cache_dir: Optional[Union[str, Path]], 
                 registry_dir: Optional[Union[str, Path]], auto_update: bool,
                 manifests_repo_url: str):
        """Initialize Registry manager.
        
        Args:
            version: Dataset version identifier
            cache_dir: Local directory for caching downloaded files
            registry_dir: Local directory containing registry files
            auto_update: Whether to auto-update tessera-manifests repository
            manifests_repo_url: Git repository URL for tessera-manifests
        """
        self.version = version
        self._cache_dir = cache_dir
        self._auto_update = auto_update
        self._manifests_repo_url = manifests_repo_url
        self._registry_dir = self._resolve_registry_dir(registry_dir)
        
        # Pooch instances for fetching data
        self._pooch = None
        self._landmask_pooch = None
        
        # Available data tracking
        self._available_embeddings = []
        self._available_landmasks = []
        self._loaded_blocks = set()  # Track which blocks have been loaded for embeddings
        self._loaded_tile_blocks = set()  # Track which blocks have been loaded for landmasks
        
        # Registry file paths
        self._registry_base_dir = None
        self._registry_file = None
        
        # Initialize pooch instances
        self._initialize_pooch()
    
    def _resolve_registry_dir(self, registry_dir: Optional[Union[str, Path]]) -> Optional[str]:
        """Resolve the registry directory path from multiple sources.
        
        This method normalizes the registry directory path to always point to the
        directory containing the actual registry files (embeddings/, landmasks/).
        
        Priority order:
        1. Explicit registry_dir parameter
        2. TESSERA_REGISTRY_DIR environment variable
        3. Auto-clone tessera-manifests repository to cache dir
        
        Args:
            registry_dir: Directory containing registry files or parent directory
                         with 'registry' subdirectory
        
        Returns:
            Path to directory containing registry files, or None for remote-only mode
        """
        resolved_path = None
        
        # 1. Use explicit parameter if provided
        if registry_dir is not None:
            resolved_path = str(registry_dir)
        # 2. Check environment variable
        elif os.environ.get("TESSERA_REGISTRY_DIR"):
            resolved_path = os.environ.get("TESSERA_REGISTRY_DIR")
        # 3. Auto-clone tessera-manifests repository
        else:
            return self._setup_tessera_manifests()  # This already returns registry subdir
        
        # Normalize the path to point to the actual registry directory
        if resolved_path:
            registry_path = Path(resolved_path)
            
            # If the path contains a 'registry' subdirectory, use that
            if (registry_path / "registry").exists():
                return str(registry_path / "registry")
            # Otherwise assume the path already points to the registry directory
            else:
                return str(registry_path)
        
        return None
    
    def _setup_tessera_manifests(self) -> str:
        """Setup tessera-manifests repository in cache directory.
        
        Clones or updates the tessera-manifests repository from GitHub.
        
        Returns:
            Path to the tessera-manifests directory
        """
        cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        manifests_dir = Path(cache_path) / "tessera-manifests"
        
        if manifests_dir.exists():
            if self._auto_update:
                # Update existing repository
                try:
                    print(f"Updating tessera-manifests repository in {manifests_dir}")
                    subprocess.run(
                        ["git", "fetch", "origin"],
                        cwd=manifests_dir,
                        check=True,
                        capture_output=True,
                    )
                    
                    subprocess.run(
                        ["git", "reset", "--hard", "origin/main"],
                        cwd=manifests_dir,
                        check=True,
                        capture_output=True,
                    )
                    
                    print("✓ tessera-manifests updated to latest version")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Failed to update tessera-manifests: {e}")
        else:
            # Clone repository
            try:
                print(f"Cloning tessera-manifests repository to {manifests_dir}")
                subprocess.run(
                    ["git", "clone", self._manifests_repo_url, str(manifests_dir)],
                    check=True,
                    capture_output=True,
                )
                
                print("✓ tessera-manifests repository cloned successfully")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone tessera-manifests repository: {e}")
        
        # Return the registry subdirectory path
        registry_dir = manifests_dir / "registry"
        return str(registry_dir)
    
    def _initialize_pooch(self):
        """Initialize Pooch data fetchers for embeddings and land masks.
        
        Sets up two Pooch instances:
        1. Main fetcher for numpy embedding files (.npy and _scales.npy)
        2. Land mask fetcher for GeoTIFF files containing binary land/water
           masks and coordinate reference system metadata
        
        Registry files are loaded lazily per year to improve performance.
        """
        cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        
        # Initialize main pooch for numpy embeddings
        self._pooch = pooch.create(
            path=cache_path,
            base_url=f"{TESSERA_BASE_URL}/{self.version}/global_0.1_degree_representation/",
            version=self.version,
            registry=None,
            env="TESSERA_DATA_DIR",
        )
        
        # Registry files will be loaded lazily when needed
        # This is handled by ensure_block_loaded method
        
        # Initialize land mask pooch for landmask GeoTIFF files
        # These TIFFs serve dual purposes:
        # 1. Binary land/water distinction (pixel values 0=water, 1=land)
        # 2. Coordinate reference system metadata for proper georeferencing
        self._landmask_pooch = pooch.create(
            path=cache_path,
            base_url=f"{TESSERA_BASE_URL}/{self.version}/global_0.1_degree_tiff_all/",
            version=self.version,
            registry=None,
            env="TESSERA_DATA_DIR",  # CR:avsm FIXME this should be a separate subdir
        )
        
        # Load registry index for block-based registries
        self._load_registry_index()
        
        # Try to load tiles registry index
        self._load_tiles_registry_index()
    
    def _load_tiles_registry_index(self):
        """Load the registry index for block-based tile registries.
        
        Downloads and caches the registry index file that lists all
        available tile block registry files.
        """
        try:
            # The registry file should already be loaded by _load_registry_index
            # This method exists for compatibility but doesn't need to re-download
            pass
        
        except Exception as e:
            print(f"Warning: Could not load registry: {e}")
            # Continue without landmask support if registry loading fails
    
    def _load_registry_index(self):
        """Load the registry index for block-based registries.
        
        If registry_dir is provided, loads registry files from local directory.
        Otherwise downloads and caches the registry index file from remote.
        """
        if self._registry_dir:
            # Use local registry directory (already normalized to point to registry files)
            registry_path = Path(self._registry_dir)
            self._registry_base_dir = str(registry_path)
            
            # Look for master registry file in the local directory
            master_registry = registry_path / "registry.txt"
            if master_registry.exists():
                self._registry_file = str(master_registry)
            else:
                # No master registry file, we'll scan directories later
                self._registry_file = None
        else:
            # Original behavior: download from remote
            cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
            self._registry_base_dir = cache_path
            
            # Download the master registry containing hashes of registry files
            self._registry_file = pooch.retrieve(
                url=f"{TESSERA_BASE_URL}/{self.version}/registry/registry.txt",
                known_hash=None,
                fname="registry.txt",
                path=cache_path,
                progressbar=True,
            )
    
    def _get_registry_hash(self, registry_filename: str) -> Optional[str]:
        """Get the hash for a specific registry file from the master registry.txt.
        
        Args:
            registry_filename: Name of the registry file to look up
        
        Returns:
            Hash string if found, None otherwise
        """
        try:
            if not self._registry_file or not Path(self._registry_file).exists():
                return None
            
            with open(self._registry_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split(" ", 1)
                        if len(parts) == 2 and parts[0] == registry_filename:
                            return parts[1]
            return None
        except Exception:
            return None
    
    def ensure_block_loaded(self, year: int, lon: float, lat: float):
        """Ensure registry data for a specific block is loaded.
        
        Loads only the registry file containing the specific coordinates needed,
        providing efficient lazy loading of registry data.
        
        Args:
            year: Year to load (e.g., 2024)
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
        """
        block_lon, block_lat = get_block_coordinates(lon, lat)
        block_key = (year, block_lon, block_lat)
        
        if block_key in self._loaded_blocks:
            return
        
        registry_filename = get_embeddings_registry_filename(str(year), block_lon, block_lat)
        
        if self._registry_dir:
            # Load from local directory
            embeddings_dir = Path(self._registry_base_dir) / "embeddings"
            registry_file = embeddings_dir / registry_filename
            
            if not registry_file.exists():
                raise FileNotFoundError(f"Embeddings registry file not found: {registry_file}")
            
            # Load the registry file directly (now in correct pooch format)
            self._pooch.load_registry(str(registry_file))
            self._loaded_blocks.add(block_key)
            self._parse_available_embeddings()
            return
        else:
            # Original behavior: download from remote
            # Get the hash from the master registry.txt file
            registry_hash = self._get_registry_hash(registry_filename)
            
            # Download the specific block registry file
            registry_url = f"{TESSERA_BASE_URL}/{self.version}/registry/{registry_filename}"
            registry_file = pooch.retrieve(
                url=registry_url,
                known_hash=registry_hash,
                fname=registry_filename,
                path=self._registry_base_dir,
                progressbar=False,  # Don't show progress for individual block downloads
            )
        
        # Load the registry into the pooch instance
        self._pooch.load_registry(registry_file)
        self._loaded_blocks.add(block_key)
        
        # Update available embeddings cache
        self._parse_available_embeddings()
    
    def load_all_blocks(self):
        """Load all available block registries to build complete embedding list.
        
        This method is used when a complete listing of all embeddings is needed,
        such as for generating coverage maps. It scans the local registry directory
        or parses the master registry to find all block files and loads them.
        """
        try:
            if self._registry_dir:
                # Scan local embeddings directory for registry files
                embeddings_dir = Path(self._registry_base_dir) / "embeddings"
                if not embeddings_dir.exists():
                    print(f"Warning: Embeddings directory not found: {embeddings_dir}")
                    return
                
                # Find all embeddings registry files
                block_files = []
                for file_path in embeddings_dir.glob("embeddings_*.txt"):
                    if "_lon" in file_path.name and "_lat" in file_path.name:
                        block_files.append(file_path.name)
                
                print(f"Found {len(block_files)} block registry files to load")
                
                # Load each block registry
                for i, block_file in enumerate(block_files):
                    if (i + 1) % 100 == 0:  # Progress indicator every 100 blocks
                        print(f"Loading block registries: {i + 1}/{len(block_files)}")
                    
                    try:
                        registry_file_path = embeddings_dir / block_file
                        
                        # Load the registry file directly (now in correct pooch format)
                        self._pooch.load_registry(str(registry_file_path))
                        
                        # Mark this block as loaded
                        # Parse filename format: embeddings_YYYY_lonXXX_latYYY.txt
                        # Examples: embeddings_2024_lon-15_lat10.txt, embeddings_2024_lon130_lat45.txt
                        parts = block_file.replace(".txt", "").split("_")
                        if len(parts) >= 4:
                            year = int(parts[1])  # parts[0] is "embeddings", parts[1] is year
                            
                            # Extract lon and lat values
                            lon_part = None
                            lat_part = None
                            for j, part in enumerate(parts):
                                if part.startswith("lon"):
                                    lon_part = part[3:]  # Remove 'lon' prefix
                                elif part.startswith("lat"):
                                    lat_part = part[3:]  # Remove 'lat' prefix
                            
                            if lon_part and lat_part:
                                # Convert to block coordinates (assuming these are already block coordinates)
                                block_lon = int(lon_part)
                                block_lat = int(lat_part)
                                self._loaded_blocks.add((year, block_lon, block_lat))
                    
                    except Exception as e:
                        print(f"Warning: Failed to load block registry {block_file}: {e}")
                        continue
            
            else:
                # Parse registry.txt to find all block registry files
                block_files = []
                with open(self._registry_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split(" ", 1)
                            if len(parts) == 2:
                                filename = parts[0]
                                # Look for embeddings registry files (format: embeddings_YYYY_lonXXX_latYYY.txt)
                                if filename.startswith("embeddings_") and "_lon" in filename and "_lat" in filename and filename.endswith(".txt"):
                                    block_files.append(filename)
                
                print(f"Found {len(block_files)} block registry files to load")
                
                # Load each block registry
                for i, block_file in enumerate(block_files):
                    if (i + 1) % 100 == 0:  # Progress indicator every 100 blocks
                        print(f"Loading block registries: {i + 1}/{len(block_files)}")
                    
                    try:
                        # Download the block registry file
                        registry_url = f"{TESSERA_BASE_URL}/{self.version}/registry/{block_file}"
                        registry_hash = self._get_registry_hash(block_file)
                        
                        downloaded_file = pooch.retrieve(
                            url=registry_url,
                            known_hash=registry_hash,
                            fname=block_file,
                            path=self._registry_base_dir,
                            progressbar=False,  # Don't show progress for individual files
                        )
                        
                        # Load the registry into the pooch instance
                        self._pooch.load_registry(downloaded_file)
                        
                        # Mark this block as loaded
                        # Parse filename format: embeddings_YYYY_lonXXX_latYYY.txt
                        # Examples: embeddings_2024_lon-15_lat10.txt, embeddings_2024_lon130_lat45.txt
                        parts = block_file.replace(".txt", "").split("_")
                        if len(parts) >= 4:
                            year = int(parts[1])  # parts[0] is "embeddings", parts[1] is year
                            
                            # Extract lon and lat values
                            lon_part = None
                            lat_part = None
                            for j, part in enumerate(parts):
                                if part.startswith("lon"):
                                    lon_part = part[3:]  # Remove 'lon' prefix
                                elif part.startswith("lat"):
                                    lat_part = part[3:]  # Remove 'lat' prefix
                            
                            if lon_part and lat_part:
                                # Convert to block coordinates (assuming these are already block coordinates)
                                block_lon = int(lon_part)
                                block_lat = int(lat_part)
                                self._loaded_blocks.add((year, block_lon, block_lat))
                    
                    except Exception as e:
                        print(f"Warning: Failed to load block registry {block_file}: {e}")
                        continue
            
            # Update available embeddings cache
            self._parse_available_embeddings()
            print(f"Loaded {len(self._available_embeddings)} total embeddings")
        
        except Exception as e:
            print(f"Error loading all blocks: {e}")
    
    def load_blocks_for_region(self, bounds: Tuple[float, float, float, float], year: int):
        """Load only the registry blocks needed for a specific region.
        
        This is much more efficient than loading all blocks globally when only
        working with a specific geographic region.
        
        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Get all blocks that intersect with the region
        required_blocks = get_all_blocks_in_range(min_lon, max_lon, min_lat, max_lat)
        
        print(
            f"Loading {len(required_blocks)} registry blocks for region bounds: "
            f"({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})"
        )
        
        # Load each required block
        blocks_loaded = 0
        for block_lon, block_lat in required_blocks:
            block_key = (year, block_lon, block_lat)
            
            if block_key not in self._loaded_blocks:
                # Use the center of the block to trigger loading
                center_lon = block_lon + 2.5  # Center of 5-degree block
                center_lat = block_lat + 2.5  # Center of 5-degree block
                
                try:
                    self.ensure_block_loaded(year, center_lon, center_lat)
                    blocks_loaded += 1
                except Exception as e:
                    raise RuntimeError(f"Failed to load registry block ({block_lon}, {block_lat}): {e}") from e
        
        # Calculate how many blocks are actually available
        blocks_available = sum(
            1 for block_lon, block_lat in required_blocks
            if (year, block_lon, block_lat) in self._loaded_blocks
        )
        
        if blocks_loaded > 0:
            print(
                f"Successfully loaded {blocks_loaded} new registry blocks ({blocks_available}/{len(required_blocks)} total available)"
            )
        else:
            print(
                f"Using {blocks_available}/{len(required_blocks)} already loaded registry blocks"
            )
        
        # Update available embeddings cache
        self._parse_available_embeddings()
    
    def ensure_tile_block_loaded(self, lon: float, lat: float):
        """Ensure registry data for a specific tile block is loaded.
        
        Loads only the registry file containing the specific coordinates needed
        for landmask tiles, providing efficient lazy loading.
        
        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
        """
        block_lon, block_lat = get_block_coordinates(lon, lat)
        block_key = (block_lon, block_lat)
        
        if block_key in self._loaded_tile_blocks:
            return
        
        registry_filename = get_landmasks_registry_filename(block_lon, block_lat)
        
        if self._registry_dir:
            # Load from local directory using block-based landmasks
            landmasks_dir = Path(self._registry_base_dir) / "landmasks"
            landmasks_registry_file = landmasks_dir / registry_filename
            
            if not landmasks_registry_file.exists():
                raise FileNotFoundError(f"Landmasks registry file not found: {landmasks_registry_file}")
            
            # Load the block-specific landmasks registry
            self._landmask_pooch.load_registry(str(landmasks_registry_file))
            self._parse_available_landmasks()
            
            # Mark this block as loaded
            self._loaded_tile_blocks.add(block_key)
            return
        else:
            # Original behavior: download from remote
            # Get the hash from the master registry.txt file
            registry_hash = self._get_registry_hash(registry_filename)
            
            # Download the specific tile block registry file
            registry_url = f"{TESSERA_BASE_URL}/{self.version}/registry/{registry_filename}"
            registry_file = pooch.retrieve(
                url=registry_url,
                known_hash=registry_hash,
                fname=registry_filename,
                path=self._registry_base_dir,
                progressbar=False,  # Don't show progress for individual block downloads
            )
        
        # Load the registry into the landmask pooch instance
        self._landmask_pooch.load_registry(registry_file)
        self._loaded_tile_blocks.add(block_key)
        
        # Update available landmasks cache
        self._parse_available_landmasks()
    
    def _parse_available_embeddings(self):
        """Parse registry files to build index of available embedding tiles.
        
        Scans through loaded registry files to extract metadata about available
        tiles. Each tile is identified by year, latitude, and longitude. This
        method is called automatically when registry files are loaded.
        
        The index is stored as a sorted list of (year, lat, lon) tuples for
        efficient searching and iteration.
        """
        embeddings = []
        
        if self._pooch and self._pooch.registry:
            for file_path in self._pooch.registry.keys():
                # Only process .npy files that are not scale files
                if file_path.endswith(".npy") and not file_path.endswith("_scales.npy"):
                    # Parse file path: e.g., "2024/grid_0.15_52.05/grid_0.15_52.05.npy"
                    parts = file_path.split("/")
                    if len(parts) >= 3:
                        year_str = parts[0]
                        grid_name = parts[1]  # e.g., "grid_0.15_52.05"
                        
                        try:
                            year = int(year_str)
                            
                            # Extract coordinates from grid name
                            if grid_name.startswith("grid_"):
                                coords = grid_name[5:].split("_")  # Remove "grid_" prefix
                                if len(coords) == 2:
                                    lon = float(coords[0])
                                    lat = float(coords[1])
                                    embeddings.append((year, lat, lon))
                        
                        except (ValueError, IndexError):
                            continue
        
        # Sort by year, then lat, then lon for consistent ordering
        embeddings.sort(key=lambda x: (x[0], x[1], x[2]))
        self._available_embeddings = embeddings
    
    def _parse_available_landmasks(self):
        """Parse land mask registry to index available GeoTIFF files.
        
        Land mask files serve dual purposes:
        1. Provide binary land/water classification (0=water, 1=land)
        2. Store coordinate reference system metadata for proper georeferencing
        
        This method builds an index of available land mask tiles as (lat, lon)
        tuples for efficient lookup during merge operations.
        """
        landmasks = []
        
        if not self._landmask_pooch or not self._landmask_pooch.registry:
            return
        
        for file_path in self._landmask_pooch.registry.keys():
            # Parse file path: e.g., "grid_0.15_52.05.tiff"
            if file_path.endswith(".tiff"):
                # Extract coordinates from filename
                filename = Path(file_path).name
                if filename.startswith("grid_"):
                    coords = filename[5:-5].split("_")  # Remove "grid_" prefix and ".tiff" suffix
                    if len(coords) == 2:
                        try:
                            lon = float(coords[0])
                            lat = float(coords[1])
                            landmasks.append((lat, lon))
                        except ValueError:
                            continue
        
        # Sort by lat, then lon for consistent ordering
        landmasks.sort(key=lambda x: (x[0], x[1]))
        self._available_landmasks = landmasks
    
    def get_available_years(self) -> List[int]:
        """List all years with available Tessera embeddings.
        
        Returns the years that have been loaded in blocks, or the common
        range of years if no blocks have been loaded yet.
        
        Returns:
            List of years with available data, sorted in ascending order.
        """
        loaded_years = {year for year, _, _ in self._loaded_blocks}
        if loaded_years:
            return sorted(loaded_years)
        else:
            # Return common range if no blocks loaded yet via an index
            # FIXME: determine this from the registry directly
            return list(range(2017, 2025))
    
    def fetch(self, path: str, progressbar: bool = True) -> str:
        """Fetch a file from the main pooch registry.
        
        Args:
            path: Path to the file in the registry
            progressbar: Whether to show download progress
        
        Returns:
            Local file path to the cached file
        """
        return self._pooch.fetch(path, progressbar=progressbar)
    
    def fetch_landmask(self, filename: str, progressbar: bool = True) -> str:
        """Fetch a landmask file from the landmask pooch registry.
        
        Args:
            filename: Name of the landmask file
            progressbar: Whether to show download progress
        
        Returns:
            Local file path to the cached file
        """
        return self._landmask_pooch.fetch(filename, progressbar=progressbar)
    
    @property
    def available_embeddings(self) -> List[Tuple[int, float, float]]:
        """Get list of available embeddings."""
        return self._available_embeddings
    
    @property
    def available_landmasks(self) -> List[Tuple[float, float]]:
        """Get list of available landmasks."""
        return self._available_landmasks
    
    @property
    def loaded_blocks(self) -> set:
        """Get set of loaded embedding blocks."""
        return self._loaded_blocks
