"""Core GeoTessera functionality.

Simplified library focusing on:
1. Downloading tiles for lat/lon bounding boxes to numpy arrays
2. Exporting tiles to individual GeoTIFF files with accurate metadata

All other functionality has been moved to separate modules or removed.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np

from .registry import Registry, world_to_tile_coords

try:
    import importlib.metadata
    __version__ = importlib.metadata.version("geotessera")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class GeoTessera:
    """Simplified GeoTessera for downloading tiles and exporting GeoTIFFs.
    
    Core functionality:
    - Download tiles within a bounding box to numpy arrays
    - Export individual tiles as GeoTIFF files with correct metadata
    - Manage registry and data access
    """

    def __init__(
        self,
        dataset_version: str = "v1", 
        cache_dir: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
        auto_update: bool = True,
        manifests_repo_url: str = "https://github.com/ucam-eo/tessera-manifests.git"
    ):
        """Initialize GeoTessera with registry management.
        
        Args:
            dataset_version: Tessera dataset version (e.g., 'v1', 'v2')
            cache_dir: Directory for caching downloaded files
            registry_dir: Directory containing registry files
            auto_update: Whether to auto-update registry
            manifests_repo_url: Git repository URL for registry manifests
        """
        self.dataset_version = dataset_version
        self.registry = Registry(
            version=dataset_version,
            cache_dir=cache_dir,
            registry_dir=registry_dir,
            auto_update=auto_update,
            manifests_repo_url=manifests_repo_url
        )
        
    @property
    def version(self) -> str:
        """Get the GeoTessera library version."""
        return __version__

    def get_available_years(self) -> List[int]:
        """Get list of available years."""
        return self.registry.get_available_years()

    def fetch_embeddings(
        self, 
        bbox: Tuple[float, float, float, float],
        year: int = 2024,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Fetch all embedding tiles within a bounding box as numpy arrays.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to download
            progress_callback: Optional callback function(current, total) for progress tracking
            
        Returns:
            List of (tile_lat, tile_lon, embedding_array) tuples
            Each embedding_array is shape (H, W, 128) with dequantized values
        """
        # Load registry blocks for this region and get available tiles directly
        tiles_to_download = self.registry.load_blocks_for_region(bbox, year)
        
        # Download each tile with progress tracking
        results = []
        total_tiles = len(tiles_to_download)
        
        for i, (tile_lat, tile_lon) in enumerate(tiles_to_download):
            try:
                embedding = self.fetch_embedding(tile_lat, tile_lon, year)
                results.append((tile_lat, tile_lon, embedding))
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, total_tiles)
                    
            except Exception as e:
                print(f"Warning: Failed to download tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}")
                continue
                
        return results

    def fetch_embedding(self, lat: float, lon: float, year: int) -> np.ndarray:
        """Fetch and dequantize a single embedding tile.
        
        Args:
            lat: Tile center latitude
            lon: Tile center longitude  
            year: Year of embeddings
            
        Returns:
            Dequantized embedding array of shape (H, W, 128)
        """
        from .registry import tile_to_embedding_path
        
        # Ensure the block is loaded
        self.registry.ensure_block_loaded(year, lon, lat)
        
        # Get file paths
        embedding_path, scales_path = tile_to_embedding_path(lat, lon, year)
        
        # Fetch the files
        embedding_file = self.registry.fetch(embedding_path, progressbar=False)
        scales_file = self.registry.fetch(scales_path, progressbar=False)
        
        # Load and dequantize
        quantized_embedding = np.load(embedding_file)
        scales = np.load(scales_file)
        
        # Dequantize using scales
        # Handle both 2D scales (H, W) and 3D scales (H, W, 128)
        if scales.ndim == 2 and quantized_embedding.ndim == 3:
            # Broadcast 2D scales to match 3D embedding shape
            scales = scales[..., np.newaxis]  # Add channel dimension
        
        dequantized = quantized_embedding.astype(np.float32) * scales
        
        return dequantized

    def _get_utm_projection_from_landmask(self, lat: float, lon: float):
        """Get UTM projection info from corresponding landmask tile.
        
        Args:
            lat: Tile center latitude
            lon: Tile center longitude
            
        Returns:
            Tuple of (crs, transform) from landmask tile
            
        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile cannot be fetched or read
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for UTM projection retrieval: pip install rasterio")
            
        try:
            from .registry import tile_to_landmask_filename
            
            # Get landmask filename
            landmask_filename = tile_to_landmask_filename(lat, lon)
            
            # Ensure registry block is loaded
            self.registry.ensure_tile_block_loaded(lon, lat)
            
            # Fetch landmask file
            landmask_path = self.registry.fetch_landmask(landmask_filename, progressbar=False)
            
            # Extract CRS and transform
            with rasterio.open(landmask_path) as src:
                if src.crs is None:
                    raise RuntimeError(f"Landmask tile {landmask_filename} has no CRS information")
                if src.transform is None:
                    raise RuntimeError(f"Landmask tile {landmask_filename} has no transform information")
                return src.crs, src.transform
                
        except Exception as e:
            if isinstance(e, (ImportError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to get UTM projection from landmask for ({lat:.2f}, {lon:.2f}): {e}") from e

    def export_embedding_geotiff(
        self,
        lat: float,
        lon: float,
        output_path: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw"
    ) -> str:
        """Export a single embedding tile as a GeoTIFF file with native UTM projection.
        
        Args:
            lat: Tile center latitude
            lon: Tile center longitude
            output_path: Output path for GeoTIFF file
            year: Year of embeddings to export
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF
            
        Returns:
            Path to created GeoTIFF file
            
        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile or embedding data cannot be fetched
            FileNotFoundError: If registry files are missing
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF export: pip install rasterio")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fetch single tile
        embedding = self.fetch_embedding(lat, lon, year)
        
        # Select bands
        if bands is not None:
            data = embedding[:, :, bands].copy()
            band_count = len(bands)
        else:
            data = embedding.copy()
            band_count = 128
            
        # Get dimensions for GeoTIFF
        height, width = data.shape[:2]
        
        # Get UTM projection from landmask
        crs, transform = self._get_utm_projection_from_landmask(lat, lon)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=band_count,
            dtype='float32',
            crs=crs,
            transform=transform,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256
        ) as dst:
            # Write bands
            for i in range(band_count):
                dst.write(data[:, :, i], i + 1)
                
            # Add band descriptions
            if bands is not None:
                for i, band_idx in enumerate(bands):
                    dst.set_band_description(i + 1, f"Tessera_Band_{band_idx}")
            else:
                for i in range(128):
                    dst.set_band_description(i + 1, f"Tessera_Band_{i}")
                    
            # Add metadata
            dst.update_tags(
                TESSERA_DATASET_VERSION=self.dataset_version,
                TESSERA_YEAR=str(year),
                TESSERA_TILE_LAT=f"{lat:.2f}",
                TESSERA_TILE_LON=f"{lon:.2f}",
                TESSERA_DESCRIPTION="GeoTessera satellite embedding tile",
                GEOTESSERA_VERSION=__version__
            )
                    
        return str(output_path)

    def export_embedding_geotiffs(
        self,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """Export all embedding tiles in bounding box as individual GeoTIFF files with native UTM projections.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save GeoTIFF files
            year: Year of embeddings to export
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF
            progress_callback: Optional callback function(current, total) for progress tracking
            
        Returns:
            List of paths to created GeoTIFF files
            
        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tiles or embedding data cannot be fetched
            FileNotFoundError: If registry files are missing
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF export: pip install rasterio")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a wrapper callback to handle two-phase progress
        def fetch_progress_callback(current: int, total: int):
            if progress_callback:
                # Phase 1: Fetching tiles (0-50% of total progress)
                overall_progress = int((current / total) * 50)
                progress_callback(overall_progress, 100, f"Fetching tile {current}/{total}")
        
        # Fetch tiles with progress tracking
        if progress_callback:
            progress_callback(0, 100, "Loading registry blocks...")
        
        tiles = self.fetch_embeddings(bbox, year, fetch_progress_callback)
        
        if not tiles:
            print("No tiles found in bounding box")
            return []
        
        if progress_callback:
            progress_callback(50, 100, f"Fetched {len(tiles)} tiles, starting GeoTIFF export...")
            
        created_files = []
        total_tiles = len(tiles)
        
        for i, (tile_lat, tile_lon, embedding) in enumerate(tiles):
            # Create filename first for progress reporting
            filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}.tif"
            output_path = output_dir / filename
            
            # Update progress to show we're starting this file
            if progress_callback:
                export_progress = int(50 + (i / total_tiles) * 50)
                progress_callback(export_progress, 100, f"Creating {filename}...")
            
            # Select bands
            if bands is not None:
                data = embedding[:, :, bands].copy()
                band_count = len(bands)
            else:
                data = embedding.copy()
                band_count = 128
            
            # Get dimensions for GeoTIFF
            height, width = data.shape[:2]
            
            # Get UTM projection from landmask
            crs, transform = self._get_utm_projection_from_landmask(tile_lat, tile_lon)
            
            # Write GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=band_count,
                dtype='float32',
                crs=crs,
                transform=transform,
                compress=compress,
                tiled=True,
                blockxsize=256,
                blockysize=256
            ) as dst:
                # Write bands
                for i in range(band_count):
                    dst.write(data[:, :, i], i + 1)
                    
                # Add band descriptions
                if bands is not None:
                    for i, band_idx in enumerate(bands):
                        dst.set_band_description(i + 1, f"Tessera_Band_{band_idx}")
                else:
                    for i in range(128):
                        dst.set_band_description(i + 1, f"Tessera_Band_{i}")
                        
                # Add metadata
                dst.update_tags(
                    TESSERA_DATASET_VERSION=self.dataset_version,
                    TESSERA_YEAR=str(year),
                    TESSERA_TILE_LAT=f"{tile_lat:.2f}",
                    TESSERA_TILE_LON=f"{tile_lon:.2f}",
                    TESSERA_DESCRIPTION="GeoTessera satellite embedding tile",
                    GEOTESSERA_VERSION=__version__
                )
                        
            created_files.append(str(output_path))
            
            # Update progress for GeoTIFF export phase
            if progress_callback:
                # Phase 2: Exporting GeoTIFFs (50-100% of total progress)
                export_progress = int(50 + ((i + 1) / total_tiles) * 50)
                filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}.tif"
                progress_callback(export_progress, 100, f"Exported {filename} ({i + 1}/{total_tiles})")
            
        if progress_callback:
            progress_callback(100, 100, f"Completed! Exported {len(created_files)} GeoTIFF files")
            
        print(f"Exported {len(created_files)} GeoTIFF files to {output_dir}")
        return created_files