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

    def get_available_years(self) -> List[int]:
        """Get list of available years."""
        return self.registry.get_available_years()

    def download_bbox_tiles(
        self, 
        bbox: Tuple[float, float, float, float],
        year: int = 2024
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Download all tiles within a bounding box as numpy arrays.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to download
            
        Returns:
            List of (tile_lat, tile_lon, embedding_array) tuples
            Each embedding_array is shape (H, W, 128) with dequantized values
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Load registry blocks for this region
        self.registry.load_blocks_for_region(bbox, year)
        
        # Find all tiles in the bounding box
        tiles_to_download = []
        for emb_year, lat, lon in self.registry.available_embeddings:
            if emb_year != year:
                continue
                
            # Check if tile intersects with bbox
            tile_min_lon, tile_min_lat = lon - 0.05, lat - 0.05
            tile_max_lon, tile_max_lat = lon + 0.05, lat + 0.05
            
            if (tile_min_lon < max_lon and tile_max_lon > min_lon and
                tile_min_lat < max_lat and tile_max_lat > min_lat):
                tiles_to_download.append((lat, lon))
        
        # Download each tile
        results = []
        for tile_lat, tile_lon in tiles_to_download:
            try:
                embedding = self._fetch_embedding(tile_lat, tile_lon, year)
                results.append((tile_lat, tile_lon, embedding))
            except Exception as e:
                print(f"Warning: Failed to download tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}")
                continue
                
        return results

    def _fetch_embedding(self, lat: float, lon: float, year: int) -> np.ndarray:
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
        dequantized = quantized_embedding.astype(np.float32) * scales
        
        return dequantized

    def export_tiles_to_geotiffs(
        self,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw"
    ) -> List[str]:
        """Export all tiles in bounding box as individual GeoTIFF files.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save GeoTIFF files
            year: Year of embeddings to export
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF
            
        Returns:
            List of paths to created GeoTIFF files
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("rasterio required for GeoTIFF export: pip install rasterio")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tiles
        tiles = self.download_bbox_tiles(bbox, year)
        
        if not tiles:
            print("No tiles found in bounding box")
            return []
            
        created_files = []
        
        for tile_lat, tile_lon, embedding in tiles:
            # Select bands
            if bands is not None:
                data = embedding[:, :, bands].copy()
                band_count = len(bands)
            else:
                data = embedding.copy()
                band_count = 128
                
            # Create filename
            filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}.tif"
            output_path = output_dir / filename
            
            # Get tile bounds
            from .registry import get_tile_bounds
            west, south, east, north = get_tile_bounds(tile_lat, tile_lon)
            
            # Create georeferencing transform
            height, width = data.shape[:2]
            transform = from_bounds(west, south, east, north, width, height)
            
            # Write GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=band_count,
                dtype='float32',
                crs='EPSG:4326',
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
                    GEOTESSERA_VERSION=self.__class__.__module__.split('.')[0] + ' library'
                )
                        
            created_files.append(str(output_path))
            
        print(f"Exported {len(created_files)} GeoTIFF files to {output_dir}")
        return created_files