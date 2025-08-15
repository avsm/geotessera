"""Core module for accessing and working with Tessera geospatial embeddings.

This module provides the main GeoTessera class which interfaces with pre-computed
satellite embeddings from the Tessera foundation model. The embeddings compress
a full year of Sentinel-1 and Sentinel-2 observations into 128-dimensional
representation maps at 10m spatial resolution.

The module handles:
- Automatic data fetching and caching from remote servers
- Dequantization of compressed embeddings using scale factors
- Geographic tile discovery and intersection analysis
- Visualization and export of embeddings as GeoTIFF files
- Merging multiple tiles with proper coordinate alignment
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Iterator, Dict
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry

from .registry import (
    Registry,
    world_to_tile_coords, 
    tile_to_embedding_path,
    tile_to_landmask_filename,
    get_tile_bounds,
    get_tile_box
)




class GeoTessera:
    """Interface for accessing Tessera foundation model embeddings.

    GeoTessera provides access to pre-computed embeddings from the Tessera
    foundation model, which processes Sentinel-1 and Sentinel-2 satellite imagery
    to generate dense representation maps. Each embedding compresses a full year
    of temporal-spectral observations into 128 channels at 10m resolution.

    The embeddings are organized in a global 0.1-degree grid system, with each
    tile covering approximately 11km × 11km at the equator. Files are fetched
    on-demand and cached locally for efficient access.

    Attributes:
        version: Dataset version identifier (default: "v1")
        cache_dir: Local directory for caching downloaded files
        registry_dir: Local directory containing registry files (if None, downloads from remote)

    Example:
        >>> gt = GeoTessera()
        >>> # Fetch embeddings for Cambridge, UK
        >>> embedding = gt.fetch_embedding(lat=52.2053, lon=0.1218)
        >>> print(f"Shape: {embedding.shape}")  # (height, width, 128)
        >>> # Visualize as RGB composite
        >>> gt.visualize_embedding(embedding, bands=[10, 20, 30])
    """

    def __init__(
        self,
        version: str = "v1",
        cache_dir: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
        auto_update: bool = False,
        manifests_repo_url: str = "https://github.com/ucam-eo/tessera-manifests.git",
    ):
        """Initialize GeoTessera client for accessing Tessera embeddings.

        Creates a client instance that can fetch and work with pre-computed
        satellite embeddings. Data is automatically cached locally after first
        download to improve performance.

        Args:
            version: Dataset version to use. Currently "v1" is available.
            cache_dir: Directory for caching downloaded files. If None, uses
                      the system's default cache directory (~/.cache/geotessera
                      on Unix-like systems).
            registry_dir: Local directory containing registry files. If provided,
                         registry files will be loaded from this directory instead
                         of being downloaded via pooch. Should point to directory
                         containing "registry" subdirectory with embeddings and
                         landmasks folders. If None, will check TESSERA_REGISTRY_DIR
                         environment variable, and if that's also not set, will
                         auto-clone the tessera-manifests repository.
            auto_update: If True, updates the tessera-manifests repository to
                        the latest version from upstream (main branch). Only
                        applies when using the auto-cloned manifests repository.
            manifests_repo_url: Git repository URL for tessera-manifests. Only used
                               when auto-cloning the manifests repository (when no
                               registry_dir is specified and TESSERA_REGISTRY_DIR is
                               not set). Defaults to the official repository.

        Raises:
            ValueError: If the specified version is not supported.

        Note:
            The client lazily loads registry files for each year as needed,
            improving startup performance when working with specific years.
        """
        self.version = version
        
        # Initialize the Registry subclass to handle all registry operations
        self.registry = Registry(
            version=version,
            cache_dir=cache_dir,
            registry_dir=registry_dir,
            auto_update=auto_update,
            manifests_repo_url=manifests_repo_url
        )


    def get_available_years(self) -> List[int]:
        """List all years with available Tessera embeddings.

        Returns the years that have been loaded in blocks, or the common
        range of years if no blocks have been loaded yet.

        Returns:
            List of years with available data, sorted in ascending order.

        Example:
            >>> gt = GeoTessera()
            >>> years = gt.get_available_years()
            >>> print(years)  # [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        """
        return self.registry.get_available_years()

    def fetch_embedding(
        self, lat: float, lon: float, year: int = 2024, progressbar: bool = True
    ) -> np.ndarray:
        """Fetch and dequantize Tessera embeddings for a geographic location.

        Downloads both the quantized embedding array and its corresponding scale
        factors, then performs dequantization by element-wise multiplication.
        The embeddings represent learned features from a full year of Sentinel-1
        and Sentinel-2 satellite observations.

        Args:
            lat: Latitude in decimal degrees. Will be rounded to nearest 0.1°
                 grid cell (e.g., 52.23 → 52.20).
            lon: Longitude in decimal degrees. Will be rounded to nearest 0.1°
                 grid cell (e.g., 0.17 → 0.15).
            year: Year of embeddings to fetch (2017-2024). Different years may
                  capture different environmental conditions.
            progressbar: Whether to display download progress. Set to False for
                        batch processing to reduce output verbosity.

        Returns:
            Dequantized embedding array of shape (height, width, 128) containing
            128-dimensional feature vectors for each 10m pixel. Typical tile
            dimensions are approximately 1100×1100 pixels.

        Raises:
            ValueError: If the requested tile is not available or year is invalid.
            IOError: If download fails after retries.

        Example:
            >>> gt = GeoTessera()
            >>> # Fetch embeddings for central London
            >>> embedding = gt.fetch_embedding(lat=51.5074, lon=-0.1278)
            >>> print(f"Tile shape: {embedding.shape}")
            >>> print(f"Feature dimensions: {embedding.shape[-1]} channels")

        Note:
            Files are cached after first download. Subsequent requests for the
            same tile will load from cache unless the cache is cleared.
        """
        # Convert world coordinates to tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        
        # Ensure the registry for this coordinate block is loaded
        self.registry.ensure_block_loaded(year, tile_lon, tile_lat)
        
        # Generate file paths
        embedding_path, scales_path = tile_to_embedding_path(tile_lat, tile_lon, year)

        embedding_file = self.registry.fetch(embedding_path, progressbar=progressbar)
        scales_file = self.registry.fetch(scales_path, progressbar=progressbar)

        # Load both files
        embedding = np.load(embedding_file)  # shape: (height, width, channels)
        scales = np.load(scales_file)  # shape: (height, width)

        # Dequantize by multiplying embedding by scales across all channels
        # Broadcasting scales from (height, width) to (height, width, channels)
        dequantized = embedding.astype(np.float32) * scales[:, :, np.newaxis]

        return dequantized

    def _fetch_landmask(self, lat: float, lon: float, progressbar: bool = True) -> str:
        """Download land mask GeoTIFF for coordinate reference information.

        Land mask files contain binary land/water data and crucial CRS metadata
        that defines the optimal projection for each tile. This metadata is used
        during tile merging to ensure proper geographic alignment.

        Args:
            lat: Latitude in decimal degrees (rounded to 0.1° grid).
            lon: Longitude in decimal degrees (rounded to 0.1° grid).
            progressbar: Whether to show download progress.

        Returns:
            Local file path to the cached land mask GeoTIFF.

        Raises:
            RuntimeError: If land mask registry was not loaded successfully.

        Note:
            This is an internal method used primarily during merge operations.
            End users typically don't need to call this directly.
        """
        # Convert world coordinates to tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        
        # Ensure the registry for this coordinate block is loaded
        self.registry.ensure_tile_block_loaded(tile_lon, tile_lat)

        # Generate landmask filename
        landmask_filename = tile_to_landmask_filename(tile_lat, tile_lon)

        return self.registry.fetch_landmask(landmask_filename, progressbar=progressbar)

    def _list_available_landmasks(self) -> Iterator[Tuple[float, float]]:
        """Iterate over available land mask tiles.

        Provides access to the catalog of land mask GeoTIFF files. Each file
        contains binary land/water classification and coordinate system metadata
        for its corresponding embedding tile.

        Returns:
            Iterator yielding (latitude, longitude) tuples for each available
            land mask, sorted by latitude then longitude.

        Note:
            Land masks are auxiliary data used primarily for coordinate alignment
            during tile merging operations.
        """
        return iter(self.registry.available_landmasks)

    def _count_available_landmasks(self) -> int:
        """Count total number of available land mask files.

        Returns:
            Number of land mask GeoTIFF files in the registry.

        Note:
            Land mask availability may be limited compared to embedding tiles.
            Not all embedding tiles have corresponding land masks.
        """
        return len(self.registry.available_landmasks)


    def list_available_embeddings(self) -> Iterator[Tuple[int, float, float]]:
        """Iterate over all available embedding tiles across all years.

        Provides an iterator over the complete catalog of available Tessera
        embeddings. Each tile covers a 0.1° × 0.1° area (approximately
        11km × 11km at the equator) and contains embeddings for one year.

        Returns:
            Iterator yielding (year, latitude, longitude) tuples for each
            available tile. Tiles are sorted by year, then latitude, then
            longitude.

        Example:
            >>> gt = GeoTessera()
            >>> # Count tiles in a specific region
            >>> uk_tiles = [(y, lat, lon) for y, lat, lon in gt.list_available_embeddings()
            ...             if 49 <= lat <= 59 and -8 <= lon <= 2]
            >>> print(f"UK tiles available: {len(uk_tiles)}")

        Note:
            On first call, this method will load registry files for all available
            years, which may take a few seconds.
        """
        # If no blocks have been loaded yet, load all available blocks
        if not self.registry.loaded_blocks:
            self.registry.load_all_blocks()

        return iter(self.registry.available_embeddings)

    def count_available_embeddings(self) -> int:
        """Count total number of available embedding tiles across all years.

        Returns:
            Total number of available embedding tiles in the dataset.

        Example:
            >>> gt = GeoTessera()
            >>> total = gt.count_available_embeddings()
            >>> print(f"Total tiles available: {total:,}")
        """
        return len(self.registry.available_embeddings)

    def get_tiles_for_topojson(
        self, topojson_path: Union[str, Path]
    ) -> List[Tuple[float, float, str]]:
        """Find all embedding tiles that intersect with region geometries.

        Analyzes a region file (GeoJSON, TopoJSON, Shapefile, GeoPackage) containing
        geographic features and identifies which Tessera embedding tiles overlap with
        those features. Uses improved geometry-based intersection without grid rounding
        that could miss edge tiles.

        Args:
            topojson_path: Path to a region file containing one or more geographic
                          features. Supports GeoJSON, TopoJSON, Shapefile (.shp),
                          and GeoPackage (.gpkg) formats.

        Returns:
            List of tuples containing (latitude, longitude, tile_path) for each
            tile that intersects with any geometry in the region file. The
            tile_path can be used with the Pooch fetcher.

        Example:
            >>> gt = GeoTessera()
            >>> # Find tiles covering a region (any supported format)
            >>> tiles = gt.get_tiles_for_topojson("boundary.shp")
            >>> print(f"Need {len(tiles)} tiles to cover the region")

        Note:
            This method now uses precise geometric intersection testing without
            grid rounding that could cause edge clipping issues. It returns tiles
            for all available years - use find_tiles_for_geometry() if you need
            year-specific filtering.
        """
        # Load region using general I/O utility
        gdf = gpd.read_file(topojson_path)

        # Ensure it's in the correct CRS
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Create a unified geometry (union of all features)
        unified_geom = gdf.unary_union

        # Find intersecting tiles across all available years
        overlapping_tiles = []
        for year, lat, lon in self.list_available_embeddings():
            # Create tile bounding box (tile coordinates represent center)
            tile_box = get_tile_box(lat, lon)

            # Check intersection with precise geometry testing
            if unified_geom.intersects(tile_box):
                embedding_path, _ = tile_to_embedding_path(lat, lon, year)
                overlapping_tiles.append((lat, lon, embedding_path))

        return overlapping_tiles

    def visualize_topojson_as_tiff(
        self,
        topojson_path: Union[str, Path],
        output_path: str = "topojson_tiles.tiff",
        bands: List[int] = [0, 1, 2],
        normalize: bool = True,
    ) -> str:
        """Create a GeoTIFF mosaic of embeddings covering a TopoJSON region.

        Generates a georeferenced TIFF image by mosaicking all Tessera tiles that
        intersect with the geometries in a TopoJSON file. The output is a clean
        satellite-style visualization without any overlays or decorations.

        Args:
            topojson_path: Path to TopoJSON file defining the region of interest.
            output_path: Output filename for the GeoTIFF (default: "topojson_tiles.tiff").
            bands: Three embedding channel indices to map to RGB. Default [0,1,2]
                   uses the first three channels. Try different combinations to
                   highlight different features.
            normalize: If True, normalizes each band to 0-1 range for better
                      contrast. If False, uses raw embedding values.

        Returns:
            Path to the created GeoTIFF file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If no tiles overlap with the TopoJSON region.

        Example:
            >>> gt = GeoTessera()
            >>> # Create false-color image of a national park
            >>> gt.visualize_topojson_as_tiff(
            ...     "park_boundary.json",
            ...     "park_tessera.tiff",
            ...     bands=[10, 20, 30]  # Custom band combination
            ... )

        Note:
            The output TIFF includes georeferencing information and can be
            opened in GIS software like QGIS or ArcGIS. Large regions may
            take significant time to process and require substantial memory.
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError(
                "Please install rasterio and pillow for TIFF export: pip install rasterio pillow"
            )

        # Read the TopoJSON file
        gpd.read_file(topojson_path)

        # Get overlapping tiles
        tiles = self.get_tiles_for_topojson(topojson_path)

        if not tiles:
            print("No overlapping tiles found")
            return output_path

        # Calculate bounding box for all tiles
        lon_min = min(lon - 0.05 for _, lon, _ in tiles)
        lat_min = min(lat - 0.05 for lat, _, _ in tiles)
        lon_max = max(lon + 0.05 for _, lon, _ in tiles)
        lat_max = max(lat + 0.05 for lat, _, _ in tiles)

        # Download and process each tile
        tile_data_dict = {}
        print(f"Processing {len(tiles)} tiles for TIFF export...")

        for i, (lat, lon, tile_path) in enumerate(tiles):
            print(f"Processing tile {i + 1}/{len(tiles)}: ({lat:.2f}, {lon:.2f})")

            try:
                # Download and dequantize the tile data
                data = self.fetch_embedding(lat=lat, lon=lon, progressbar=False)

                # Extract bands for visualization
                vis_data = data[:, :, bands].copy()

                # Normalize if requested
                if normalize:
                    for j in range(vis_data.shape[2]):
                        channel = vis_data[:, :, j]
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if max_val > min_val:
                            vis_data[:, :, j] = (channel - min_val) / (
                                max_val - min_val
                            )

                # Ensure we have valid RGB data in [0,1] range
                vis_data = np.clip(vis_data, 0, 1)

                # Store the processed tile data
                tile_data_dict[(lat, lon)] = vis_data

            except Exception as e:
                print(f"WARNING: Failed to download tile ({lat:.2f}, {lon:.2f}): {e}")
                tile_data_dict[(lat, lon)] = None
                # CR:avsm TODO raise error

        # Determine the resolution based on the first valid tile
        tile_height, tile_width = None, None
        for (lat, lon), tile_data in tile_data_dict.items():
            if tile_data is not None:
                tile_height, tile_width = tile_data.shape[:2]
                break

        if tile_height is None:
            raise ValueError("No valid tiles were downloaded")

        # Calculate the size of the output mosaic
        # Each tile covers 0.1 degrees, calculate pixels per degree
        pixels_per_degree_lat = tile_height / 0.1
        pixels_per_degree_lon = tile_width / 0.1

        # Calculate output dimensions
        mosaic_width = int((lon_max - lon_min) * pixels_per_degree_lon)
        mosaic_height = int((lat_max - lat_min) * pixels_per_degree_lat)

        # Create the mosaic array
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)

        # Place each tile in the mosaic
        for (lat, lon), tile_data in tile_data_dict.items():
            if tile_data is not None:
                # Calculate pixel coordinates for this tile
                x_start = int((lon - lon_min) * pixels_per_degree_lon)
                y_start = int(
                    (lat_max - lat - 0.1) * pixels_per_degree_lat
                )  # Flip Y axis

                # Get actual tile dimensions
                tile_h, tile_w = tile_data.shape[:2]

                # Calculate end positions
                y_end = y_start + tile_h
                x_end = x_start + tile_w

                # Clip to mosaic bounds
                y_start_clipped = max(0, y_start)
                x_start_clipped = max(0, x_start)
                y_end_clipped = min(mosaic_height, y_end)
                x_end_clipped = min(mosaic_width, x_end)

                # Calculate tile region to copy
                tile_y_start = y_start_clipped - y_start
                tile_x_start = x_start_clipped - x_start
                tile_y_end = tile_y_start + (y_end_clipped - y_start_clipped)
                tile_x_end = tile_x_start + (x_end_clipped - x_start_clipped)

                # Place tile in mosaic if there's any overlap
                if y_end_clipped > y_start_clipped and x_end_clipped > x_start_clipped:
                    mosaic[
                        y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped
                    ] = tile_data[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

        # Convert to uint8 for TIFF export
        mosaic_uint8 = (mosaic * 255).astype(np.uint8)

        # Create georeferencing transform
        transform = from_bounds(
            lon_min, lat_min, lon_max, lat_max, mosaic_width, mosaic_height
        )

        # Write the GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=mosaic_height,
            width=mosaic_width,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",  # WGS84
            transform=transform,
            compress="lzw",
        ) as dst:
            # Write RGB bands
            for i in range(3):
                dst.write(mosaic_uint8[:, :, i], i + 1)

        print(f"Exported high-resolution TIFF to {output_path}")
        print(f"Dimensions: {mosaic_width}x{mosaic_height} pixels")
        print(
            f"Geographic bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}"
        )

        return output_path

    def export_single_tile_as_tiff(
        self,
        lat: float,
        lon: float,
        output_path: str,
        year: int = 2024,
        bands: List[int] = [0, 1, 2],
        normalize: bool = True,
    ) -> str:
        """Export a single Tessera embedding tile as a georeferenced GeoTIFF.

        Creates a GeoTIFF file from a single embedding tile, selecting three
        channels to visualize as RGB. The output includes proper georeferencing
        metadata for use in GIS applications.

        Args:
            lat: Latitude of tile in decimal degrees (rounded to 0.1° grid).
            lon: Longitude of tile in decimal degrees (rounded to 0.1° grid).
            output_path: Filename for the output GeoTIFF.
            year: Year of embeddings to export (2017-2024).
            bands: Channel indices to map to RGB. Default is [0, 1, 2]. Each index must be
                   between 0-127. Different combinations highlight different
                   features (e.g., vegetation, water, urban areas).
            normalize: If True, stretches values to use full 0-255 range for
                      better visualization. If False, preserves relative values.

        Returns:
            Path to the created GeoTIFF file.

        Raises:
            ImportError: If rasterio is not installed.

        Example:
            >>> gt = GeoTessera()
            >>> # Export a tile over Paris with custom visualization
            >>> gt.export_single_tile_as_tiff(
            ...     lat=48.85, lon=2.35,
            ...     output_path="paris_2024.tiff",
            ...     bands=[25, 50, 75]  # Custom band selection
            ... )

        Note:
            Output files can be large (typically 10-50 MB per tile). The GeoTIFF
            uses LZW compression to reduce file size.
        """
        from .export import embeddings_to_geotiff
        
        # Fetch the embedding data
        embedding = self.fetch_embedding(lat=lat, lon=lon, year=year, progressbar=True)
        
        # Use the existing export function
        result_path = embeddings_to_geotiff(
            embedding=embedding,
            lat=lat,
            lon=lon,
            output_path=output_path,
            gt=self,
            bands=bands,
            normalize=normalize
        )
        
        print(f"Exported tile to {result_path}")
        return result_path

    def _merge_landmasks_for_region(
        self,
        bounds: Tuple[float, float, float, float],
        output_path: str,
        target_crs: str = "EPSG:4326",
    ) -> str:
        """Merge land mask tiles for a geographic region with proper alignment.

        Combines multiple land mask GeoTIFF tiles into a single file, handling
        coordinate system differences between tiles. Each tile may use a different
        optimal projection (e.g., different UTM zones), so this method reprojects
        all tiles to a common coordinate system before merging.

        The land masks provide:
        - Binary classification: 0 = water, 1 = land
        - Coordinate system metadata for accurate georeferencing
        - Projection information to avoid coordinate skew

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
                    in WGS84 decimal degrees.
            output_path: Filename for the merged GeoTIFF output.
            target_crs: Target coordinate reference system. Default "EPSG:4326"
                       (WGS84). Can be any CRS supported by rasterio.

        Returns:
            Path to the created merged land mask file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If no land mask tiles are found for the region.

        Note:
            This is an internal method used by merge_embeddings_for_region().
            Binary masks are automatically converted to visible grayscale
            (0 → 0, 1 → 255) for better visualization.
        """
        from .processing import merge_landmasks_for_region
        return merge_landmasks_for_region(
            gt=self,
            bounds=bounds,
            output_path=output_path,
            target_crs=target_crs
        )

    def merge_embeddings_for_region(
        self,
        bounds: Tuple[float, float, float, float],
        output_path: str,
        target_crs: str = "EPSG:4326",
        bands: Optional[List[int]] = None,
        year: int = 2024,
    ) -> str:
        """Create a seamless mosaic of Tessera embeddings for a geographic region.

        Merges multiple embedding tiles into a single georeferenced GeoTIFF,
        handling coordinate system differences and ensuring perfect alignment.
        This method uses land mask files to obtain optimal projection metadata
        for each tile, preventing coordinate skew when tiles span different
        UTM zones.

        The process:
        1. Identifies all tiles intersecting the bounding box
        2. Downloads embeddings and corresponding land masks
        3. Creates georeferenced temporary files using land mask CRS metadata
        4. Reprojects tiles to common coordinate system if needed
        5. Merges all tiles into seamless mosaic

        Args:
            bounds: Region bounds as (min_lon, min_lat, max_lon, max_lat) in
                    decimal degrees. Example: (-0.2, 51.4, 0.1, 51.6) for London.
            output_path: Filename for the output GeoTIFF mosaic.
            target_crs: Coordinate system for output. Default "EPSG:4326" (WGS84).
                       Use local projections (e.g., UTM) for accurate area measurements.
            bands: List of channel indices to include in output. If None (default),
                   exports all 128 channels. If specified, exports only the selected
                   channels (e.g., [0,1,2] for specific band selection).
                   All indices must be in range 0-127.
            year: Year of embeddings to merge (2017-2024).

        Returns:
            Path to the created mosaic GeoTIFF file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If no tiles found for region or invalid parameters.
            RuntimeError: If land masks are not available for alignment.

        Examples:
            >>> gt = GeoTessera()
            >>> # Create full 128-band mosaic (default)
            >>> gt.merge_embeddings_for_region(
            ...     bounds=(-0.2, 51.4, 0.1, 51.6),
            ...     output_path="london_full_128band.tif"
            ... )

            >>> # Create selected band subset
            >>> gt.merge_embeddings_for_region(
            ...     bounds=(-122.6, 37.2, -121.7, 38.0),
            ...     output_path="sf_bay_subset.tif",
            ...     bands=[30, 60, 90]  # Selected bands
            ... )

        Note:
            Large regions require significant memory and processing time.
            Full 128-band outputs can be very large (>1GB for large regions).
            All outputs are stored as float32 to preserve the full precision
            of the dequantized embeddings. The output file includes full
            georeferencing metadata and can be used in any GIS software.
        """
        from .processing import merge_embeddings_for_region
        return merge_embeddings_for_region(
            gt=self,
            bounds=bounds,
            output_path=output_path,
            target_crs=target_crs,
            bands=bands,
            year=year
        )

    def find_tiles_for_geometry(
        self,
        geometry: Union[gpd.GeoDataFrame, "shapely.geometry.BaseGeometry"],
        year: int = 2024,
    ) -> List[Tuple[float, float]]:
        """Find all available tiles intersecting with a given geometry.

        Args:
            geometry: A GeoDataFrame or Shapely geometry (must be in EPSG:4326)
            year: Year of embeddings to search

        Returns:
            List of (lat, lon) tuples for tiles that intersect the geometry
        """

        # Convert to GeoDataFrame if needed
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry.crs != "EPSG:4326":
                gdf = geometry.to_crs("EPSG:4326")
            else:
                gdf = geometry
        elif hasattr(geometry, "bounds"):  # Shapely geometry
            gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
        else:
            raise TypeError("geometry must be a GeoDataFrame or Shapely geometry")

        # Get unified geometry
        unified_geom = gdf.unary_union

        # Get bounds of the geometry for efficient registry loading
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = bounds

        # Load only the registry blocks needed for this geometry
        self.registry.load_blocks_for_region((min_lon, min_lat, max_lon, max_lat), year)

        # Find intersecting tiles from the available embeddings in loaded blocks
        tiles = []
        for tile_year, lat, lon in self.registry.available_embeddings:
            if tile_year != year:
                continue

            # Create tile bounding box (tile coordinates represent center)
            tile_box = get_tile_box(lat, lon)

            # Check intersection
            if unified_geom.intersects(tile_box):
                tiles.append((lat, lon))

        return tiles

    def merge_embeddings_for_region_file(
        self,
        region_path: Union[str, Path],
        output_path: str,
        target_crs: str = "EPSG:4326",
        bands: Optional[List[int]] = None,
        year: int = 2024,
    ) -> Optional[str]:
        """Create a seamless mosaic of Tessera embeddings for a region file.

        Convenience method that loads a region from a file and creates a merged
        GeoTIFF. Supports all formats that GeoPandas can read including GeoJSON,
        Shapefile, GeoPackage, etc.

        Args:
            region_path: Path to region file (GeoJSON, Shapefile, etc.)
            output_path: Output path for the merged GeoTIFF
            target_crs: Target coordinate system (default: "EPSG:4326")
            bands: Band indices to include, or None for all bands
            year: Year of embeddings to use

        Returns:
            Path to created TIFF file, or None on error

        Example:
            >>> tessera = GeoTessera()
            >>> result = tessera.merge_embeddings_for_region_file(
            ...     region_path="study_area.geojson",
            ...     output_path="embeddings.tiff",
            ...     bands=[0, 1, 2]
            ... )
        """
        from .processing import merge_embeddings_for_region_file
        return merge_embeddings_for_region_file(
            gt=self,
            region_path=region_path,
            output_path=output_path,
            target_crs=target_crs,
            bands=bands,
            year=year
        )

    def extract_points(
        self,
        points: Union[List[Dict], pd.DataFrame, gpd.GeoDataFrame],
        year: int = 2024,
        include_coords: bool = False,
        progressbar: bool = True,
    ) -> pd.DataFrame:
        """Extract embedding values at specific point locations.

        This method efficiently extracts embeddings for multiple points by:
        1. Grouping points by their containing tiles
        2. Loading each tile only once
        3. Extracting values for all points within that tile

        Args:
            points: Points with 'lat'/'lon' columns/keys. Can include 'label' or other metadata.
                   Accepts list of dicts, pandas DataFrame, or GeoDataFrame.
            year: Year of embeddings to use
            include_coords: If True, includes lat/lon in output DataFrame
            progressbar: Show progress bar during extraction

        Returns:
            DataFrame with embeddings (128 columns named 'emb_0' to 'emb_127') plus any metadata from input.
            Points that fall outside available tiles will be excluded from results.
        """
        from .processing import extract_points_from_embeddings
        return extract_points_from_embeddings(
            gt=self,
            points=points,
            year=year,
            include_coords=include_coords,
            progressbar=progressbar
        )


    def get_tile_crs(self, lat: float, lon: float) -> str:
        """Get the CRS of a specific tile.

        Args:
            lat: Tile latitude
            lon: Tile longitude

        Returns:
            CRS string (e.g., 'EPSG:32630')
        """
        import rasterio

        # Convert world coordinates to tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        landmask_path = self._fetch_landmask(tile_lat, tile_lon, progressbar=False)
        with rasterio.open(landmask_path) as src:
            return str(src.crs)

    def get_tile_transform(self, lat: float, lon: float):
        """Get the rasterio transform for georeferencing a tile.

        Args:
            lat: Tile latitude
            lon: Tile longitude

        Returns:
            rasterio Affine transform
        """
        import rasterio

        # Convert world coordinates to tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        landmask_path = self._fetch_landmask(tile_lat, tile_lon, progressbar=False)
        with rasterio.open(landmask_path) as src:
            return src.transform
