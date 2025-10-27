"""Core GeoTessera functionality.

The library focusses on:
1. Downloading Tessera tiles for lat/lon bounding boxes to numpy arrays
2. Exporting tiles to individual GeoTIFF files with accurate metadata
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Generator
import json
import numpy as np

from .registry import Registry

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("geotessera")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

class GeoTessera:
    """Library for downloading Tessera tiles and exporting GeoTIFFs.

    Core functionality:
    - Download tiles within a bounding box to numpy arrays
    - Export individual tiles as GeoTIFF files with correct metadata
    - Manage registry and data access
    """

    def __init__(
        self,
        dataset_version: str = "v1",
        cache_dir: Optional[Union[str, Path]] = None,
        registry_url: Optional[str] = None,
        registry_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize GeoTessera with Parquet registry.

        Args:
            dataset_version: Tessera dataset version (e.g., 'v1', 'v2')
            cache_dir: Directory for caching downloaded files
            registry_url: URL to download Parquet registry from (default: remote)
            registry_path: Local path to existing Parquet registry file
        """
        self.dataset_version = dataset_version
        self.registry = Registry(
            version=dataset_version,
            cache_dir=cache_dir,
            registry_url=registry_url,
            registry_path=registry_path,
        )

    @property
    def version(self) -> str:
        """Get the GeoTessera library version."""
        return __version__

    def embeddings_count(self, bbox: Tuple[float, float, float, float], year: int = 2024) -> int:
            """Get total number of embedding tiles within a bounding box.

            Args:
                bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
                year: Year of embeddings to consider

            Returns:
                Total number of tiles in the bounding box
            """
            tiles = self.registry.load_blocks_for_region(bbox, year)
            return len(tiles)

    # returns a generator
    def fetch_embeddings(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2024,
        progress_callback: Optional[callable] = None,
    ) -> Generator[Tuple[float, float, np.ndarray, object, object], None, None]:
        """Lazily fetches all embedding tiles within a bounding box with CRS information.
        Use as a generator to process tiles one at a time in a memory-efficient manner.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to download
            progress_callback: Optional callback function(current, total) for progress tracking

        Returns:
            Generator of (tile_lon, tile_lat, embedding_array, crs, transform) tuples where:
            - tile_lon: Tile center longitude
            - tile_lat: Tile center latitude
            - embedding_array: shape (H, W, 128) with dequantized values
            - crs: CRS object from rasterio (coordinate reference system)
            - transform: Affine transform from rasterio
        """
        # Load registry blocks for this region and get available tiles directly
        tiles_to_download = self.registry.load_blocks_for_region(bbox, year)

        # Download each tile with progress tracking
        total_tiles = len(tiles_to_download)

        for i, (tile_lon, tile_lat) in enumerate(tiles_to_download):
            try:
                # Create a sub-progress callback for this tile's downloads
                def tile_progress_callback(
                    current: int, total: int, status: str = None
                ):
                    if progress_callback:
                        # Map individual file progress to overall tile progress
                        tile_progress = (
                            i * 100 + (current / max(total, 1)) * 100
                        ) / total_tiles
                        tile_status = (
                            f"Tile {i + 1}/{total_tiles}: {status}"
                            if status
                            else f"Fetching tile {i + 1}/{total_tiles}"
                        )
                        progress_callback(int(tile_progress), 100, tile_status)

                embedding, crs, transform = self.fetch_embedding(
                    tile_lon, tile_lat, year, tile_progress_callback
                )

                yield tile_lon, tile_lat, embedding, crs, transform

                # Update progress for completed tile
                if progress_callback:
                    progress_callback(
                        (i + 1) * 100 // total_tiles,
                        100,
                        f"Completed tile {i + 1}/{total_tiles}",
                    )

            except Exception as e:
                print(
                    f"Warning: Failed to download tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}"
                )
                if progress_callback:
                    progress_callback(
                        (i + 1) * 100 // total_tiles,
                        100,
                        f"Failed tile {i + 1}/{total_tiles}",
                    )
                continue

        return None 

    def fetch_embedding(
        self,
        lon: float,
        lat: float,
        year: int,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, object, object]:
        """Fetch and dequantize a single embedding tile with CRS information.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude
            year: Year of embeddings
            progress_callback: Optional callback for download progress

        Returns:
            Tuple of (dequantized_embedding, crs, transform) where:
            - dequantized_embedding: array of shape (H, W, 128)
            - crs: CRS object from rasterio (coordinate reference system)
            - transform: Affine transform from rasterio
        """
        from .registry import tile_to_embedding_paths
        from pathlib import Path

        # Ensure the block is loaded
        self.registry.ensure_block_loaded(year, lon, lat)

        # Get file paths
        embedding_path, scales_path = tile_to_embedding_paths(lon, lat, year)

        # Fetch the files to temporary locations
        embedding_file = self.registry.fetch(
            embedding_path, progressbar=False, progress_callback=progress_callback
        )
        scales_file = self.registry.fetch(
            scales_path, progressbar=False, progress_callback=progress_callback
        )

        try:
            # Load and dequantize
            quantized_embedding = np.load(embedding_file)
            scales = np.load(scales_file)

            # Dequantize using scales
            # Handle both 2D scales (H, W) and 3D scales (H, W, 128)
            if scales.ndim == 2 and quantized_embedding.ndim == 3:
                # Broadcast 2D scales to match 3D embedding shape
                scales = scales[..., np.newaxis]  # Add channel dimension

            dequantized = quantized_embedding.astype(np.float32) * scales

            # Get CRS and transform from landmask
            crs, transform = self._get_utm_projection_from_landmask(lon, lat)

            return dequantized, crs, transform

        finally:
            # Clean up temporary files
            Path(embedding_file).unlink(missing_ok=True)
            Path(scales_file).unlink(missing_ok=True)

    def _reproject_geotiff_file(self, args):
        """Helper function to reproject a single GeoTIFF file.
                
        Args:
            args: Tuple containing (source_file, output_file, target_crs, source_resolution, compress)
        
        Returns:
            Tuple of (output_file, None) on success or (None, error_message) on failure
        """
        source_file, output_file, target_crs, source_resolution, compress = args
        
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            
            with rasterio.open(source_file) as src:
                # Calculate transform and dimensions for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds,
                    resolution=source_resolution
                )
                
                # Create reprojected file
                with rasterio.open(
                    output_file,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=target_crs,
                    transform=transform,
                    compress=compress,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                ) as dst:
                    # Reproject each band
                    for band_idx in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear
                        )
                    
                    # Copy metadata and band descriptions
                    dst.update_tags(**src.tags())
                    for band_idx in range(1, src.count + 1):
                        if src.descriptions and band_idx <= len(src.descriptions):
                            band_desc = src.descriptions[band_idx-1]
                            if band_desc:
                                dst.set_band_description(band_idx, band_desc)
            
            return output_file, None
        except Exception as e:
            return None, str(e)



    def _get_utm_projection_from_landmask(self, lon: float, lat: float):
        """Get UTM projection info from corresponding landmask tile.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude

        Returns:
            Tuple of (crs, transform) from landmask tile

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile cannot be fetched or read
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for UTM projection retrieval: pip install rasterio"
            )

        try:
            from .registry import tile_to_landmask_filename
            from pathlib import Path

            # Get landmask filename
            landmask_filename = tile_to_landmask_filename(lon, lat)

            # Ensure registry block is loaded
            self.registry.ensure_tile_block_loaded(lon, lat)

            # Fetch landmask file to temporary location
            landmask_path = self.registry.fetch_landmask(
                landmask_filename, progressbar=False
            )

            try:
                # Extract CRS and transform
                with rasterio.open(landmask_path) as src:
                    if src.crs is None:
                        raise RuntimeError(
                            f"Landmask tile {landmask_filename} has no CRS information"
                        )
                    if src.transform is None:
                        raise RuntimeError(
                            f"Landmask tile {landmask_filename} has no transform information"
                        )
                    return src.crs, src.transform

            finally:
                # Clean up temporary landmask file
                Path(landmask_path).unlink(missing_ok=True)

        except Exception as e:
            if isinstance(e, (ImportError, RuntimeError)):
                raise
            raise RuntimeError(
                f"Failed to get UTM projection from landmask for ({lon:.2f}, {lat:.2f}): {e}"
            ) from e

    def export_embedding_geotiff(
        self,
        lon: float,
        lat: float,
        output_path: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
    ) -> str:
        """Export a single embedding tile as a GeoTIFF file with native UTM projection.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude
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
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fetch single tile with CRS info
        embedding, crs, transform = self.fetch_embedding(lon, lat, year)

        # Select bands
        if bands is not None:
            data = embedding[:, :, bands].copy()
            band_count = len(bands)
        else:
            data = embedding.copy()
            band_count = 128

        # Get dimensions for GeoTIFF
        height, width = data.shape[:2]

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=band_count,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256,
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
                GEOTESSERA_VERSION=__version__,
            )

        return str(output_path)

    def export_embedding_geotiffs(
        self,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
        progress_callback: Optional[callable] = None,
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
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a wrapper callback to handle two-phase progress
        def fetch_progress_callback(current: int, total: int, status: str = None):
            if progress_callback:
                # Phase 1: Fetching tiles (0-50% of total progress)
                overall_progress = int((current / total) * 50)
                display_status = status or f"Fetching tile {current}/{total}"
                progress_callback(overall_progress, 100, display_status)

        # Fetch tiles with progress tracking
        if progress_callback:
            progress_callback(0, 100, "Loading registry blocks...")

        tiles = self.fetch_embeddings(bbox, year, fetch_progress_callback)

        if not tiles:
            print("No tiles found in bounding box")
            return []

        if progress_callback:
            progress_callback(
                50, 100, f"Fetched {len(tiles)} tiles, starting GeoTIFF export..."
            )

        created_files = []
        total_tiles = len(tiles)

        # Sequential GeoTIFF writing
        for i, (tile_lon, tile_lat, embedding, crs, transform) in enumerate(tiles):
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

            # Write GeoTIFF
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=band_count,
                dtype="float32",
                crs=crs,
                transform=transform,
                compress=compress,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                # Write bands
                for j in range(band_count):
                    dst.write(data[:, :, j], j + 1)

                # Add band descriptions
                if bands is not None:
                    for j, band_idx in enumerate(bands):
                        dst.set_band_description(j + 1, f"Tessera_Band_{band_idx}")
                else:
                    for j in range(128):
                        dst.set_band_description(j + 1, f"Tessera_Band_{j}")

                # Add metadata
                dst.update_tags(
                    TESSERA_DATASET_VERSION=self.dataset_version,
                    TESSERA_YEAR=str(year),
                    TESSERA_TILE_LAT=f"{tile_lat:.2f}",
                    TESSERA_TILE_LON=f"{tile_lon:.2f}",
                    TESSERA_DESCRIPTION="GeoTessera satellite embedding tile",
                    GEOTESSERA_VERSION=__version__,
                )

            created_files.append(str(output_path))

            # Update progress for GeoTIFF export phase
            if progress_callback:
                # Phase 2: Exporting GeoTIFFs (50-100% of total progress)
                export_progress = int(50 + ((i + 1) / total_tiles) * 50)
                progress_callback(
                    export_progress, 100, f"Exported {filename} ({i + 1}/{total_tiles})"
                )

        if progress_callback:
            progress_callback(
                100, 100, f"Completed! Exported {len(created_files)} GeoTIFF files"
            )

        print(f"Exported {len(created_files)} GeoTIFF files to {output_dir}")
        return created_files

    def merge_geotiffs_to_mosaic(
        self,
        geotiff_paths: List[str],
        output_path: Union[str, Path],
        target_crs: str = "EPSG:3857",
        compress: str = "lzw",
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Merge a list of GeoTIFF files into a single mosaic in the target CRS.

        Args:
            geotiff_paths: List of paths to GeoTIFF files to merge
            output_path: Path for output mosaic GeoTIFF
            target_crs: Target CRS for the mosaic (default: Web Mercator EPSG:3857)
            compress: Compression method for output GeoTIFF
            progress_callback: Optional callback function(current, total, status) for progress tracking

        Returns:
            Path to created mosaic file

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If merge fails
        """
        try:
            import rasterio
            from rasterio.merge import merge
            import tempfile
            import os
        except ImportError:
            raise ImportError(
                "rasterio required for mosaic creation: pip install rasterio"
            )

        if not geotiff_paths:
            raise RuntimeError("No GeoTIFF files provided")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine source resolution from first file
        with rasterio.open(geotiff_paths[0]) as first_src:
            source_resolution = min(abs(first_src.transform.a), abs(first_src.transform.e))

        # Create temporary directory for reprojected files
        with tempfile.TemporaryDirectory(prefix="geotessera_reproject_") as temp_dir:
            # Prepare reprojection arguments
            total_files = len(geotiff_paths)
            reproject_args = []
            reprojected_files = []
            
            for i, geotiff_file in enumerate(geotiff_paths):
                reprojected_file = os.path.join(temp_dir, f"reprojected_{i}.tif")
                reprojected_files.append(reprojected_file)
                reproject_args.append((geotiff_file, reprojected_file, target_crs, source_resolution, compress))
            
            if progress_callback:
                progress_callback(0, total_files * 2 + 2, "Starting reprojection...")
            
            # Sequential reprojection
            failed_files = []
            for i, args in enumerate(reproject_args):
                if progress_callback:
                    progress_callback(i, total_files * 2 + 2, f"Reprojecting file {i+1}/{total_files}...")
                
                _, error = self._reproject_geotiff_file(args)
                
                if error:
                    failed_files.append((geotiff_paths[i], error))
            
            if failed_files:
                error_msg = f"Failed to reproject {len(failed_files)} files: "
                error_msg += ", ".join([f"{Path(f).name}: {e}" for f, e in failed_files[:3]])
                if len(failed_files) > 3:
                    error_msg += f" and {len(failed_files) - 3} more"
                raise RuntimeError(error_msg)
            
            # Filter out any failed reprojections
            reprojected_files = [f for f in reprojected_files if os.path.exists(f)]

            if progress_callback:
                progress_callback(total_files, total_files * 2 + 2, "Opening files for merging...")

            # Open all reprojected files for merging
            src_files = [rasterio.open(f) for f in reprojected_files]
            
            try:
                if progress_callback:
                    progress_callback(total_files + 1, total_files * 2 + 2, "Merging tiles...")
                
                # Merge tiles
                mosaic_array, mosaic_transform = merge(src_files, method='first')
                
                # Get metadata from first file
                first_src = src_files[0]
                profile = first_src.profile.copy()
                profile.update({
                    'height': mosaic_array.shape[1],
                    'width': mosaic_array.shape[2],
                    'transform': mosaic_transform,
                    'dtype': mosaic_array.dtype,  # Use mosaic array dtype
                    'compress': compress,
                    'tiled': True,
                    'blockxsize': 512,
                    'blockysize': 512,
                })
                
                if progress_callback:
                    progress_callback(total_files * 2, total_files * 2 + 2, "Writing mosaic to disk...")
                
                # Write mosaic
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(mosaic_array)
                    
                    # Copy band descriptions from first file
                    for band_idx in range(1, mosaic_array.shape[0] + 1):
                        band_desc = first_src.descriptions[band_idx-1] if first_src.descriptions and band_idx <= len(first_src.descriptions) else None
                        if band_desc:
                            dst.set_band_description(band_idx, band_desc)
                    
                    # Update metadata
                    dst.update_tags(
                        TESSERA_TARGET_CRS=target_crs,
                        TESSERA_RESOLUTION=str(source_resolution),
                        TESSERA_TILE_COUNT=str(len(geotiff_paths)),
                        TESSERA_DESCRIPTION="GeoTessera satellite embedding mosaic",
                        GEOTESSERA_VERSION=__version__,
                    )
                
                if progress_callback:
                    progress_callback(total_files * 2 + 2, total_files * 2 + 2, "Complete")

            finally:
                # Close all source files
                for src in src_files:
                    src.close()

        return str(output_path)
    
    def apply_pca_to_embeddings(
        self,
        embeddings: List[Tuple[float, float, np.ndarray, object, object]],
        n_components: int = 3,
        standardize: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[float, float, np.ndarray, object, object, Dict]]:
        """Apply PCA to embedding tiles for visualization.
        
        Args:
            embeddings: List of (tile_lon, tile_lat, embedding_array, crs, transform) tuples
            n_components: Number of principal components to extract (default: 3 for RGB)
            standardize: Whether to standardize features before PCA
            progress_callback: Optional callback function(current, total, status) for progress tracking
            
        Returns:
            List of tuples with PCA-transformed data:
                (tile_lon, tile_lat, pca_array, crs, transform, pca_info)
            where pca_info contains:
                - explained_variance: Explained variance ratio for each component
                - total_variance: Total explained variance
                
        Raises:
            ImportError: If scikit-learn is not available
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn required for PCA visualization: pip install scikit-learn"
            )
        
        if not embeddings:
            return []
        
        pca_results = []
        total_tiles = len(embeddings)
        
        if progress_callback:
            progress_callback(0, total_tiles, "Starting PCA analysis...")
        
        for i, (tile_lon, tile_lat, embedding, crs, transform) in enumerate(embeddings):
            if progress_callback:
                progress_callback(i, total_tiles, f"Processing tile {i+1}/{total_tiles}: ({tile_lat:.2f}, {tile_lon:.2f})")
            
            # Reshape for PCA: (height, width, channels) -> (pixels, channels)
            height, width, n_bands = embedding.shape
            data_reshaped = embedding.reshape(-1, n_bands)
            
            # Handle NaN values by replacing with 0
            data_reshaped = np.nan_to_num(data_reshaped, nan=0.0)
            
            # Standardize data if requested
            if standardize:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_reshaped)
            else:
                data_scaled = data_reshaped
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)
            
            # Reshape back to image: (pixels, n_components) -> (height, width, n_components)
            pca_image = pca_result.reshape(height, width, n_components)
            
            # Create PCA info dictionary
            pca_info = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'total_variance': float(pca.explained_variance_ratio_.sum()),
                'n_components': n_components,
                'standardized': standardize,
            }
            
            pca_results.append((tile_lon, tile_lat, pca_image, crs, transform, pca_info))
        
        if progress_callback:
            progress_callback(total_tiles, total_tiles, "PCA analysis complete")
        
        return pca_results
    
    def export_pca_geotiffs(
        self,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path],
        year: int = 2024,
        n_components: int = 3,
        standardize: bool = True,
        compress: str = "lzw",
        normalize: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """Export PCA-transformed embeddings as GeoTIFF files.
        
        This method fetches embeddings, applies PCA transformation, and exports
        the results as GeoTIFF files suitable for RGB visualization.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save PCA GeoTIFF files
            year: Year of embeddings to export
            n_components: Number of principal components (default: 3 for RGB)
            standardize: Whether to standardize features before PCA
            compress: Compression method for GeoTIFF
            normalize: Whether to use global normalization across tiles (vs per-tile)
            progress_callback: Optional callback function(current, total, status)
            
        Returns:
            List of paths to created PCA GeoTIFF files
        """
        try:
            import rasterio
            from rasterio.enums import ColorInterp
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Fetch embeddings (0-40% progress)
        def fetch_progress(current, total, status=None):
            if progress_callback:
                overall_progress = int((current / total) * 40)
                progress_callback(overall_progress, 100, status or f"Fetching tile {current}/{total}")
        
        if progress_callback:
            progress_callback(0, 100, "Fetching embedding tiles...")
        
        embeddings = self.fetch_embeddings(bbox, year, fetch_progress)
        
        if not embeddings:
            if progress_callback:
                progress_callback(100, 100, "No tiles found in bounding box")
            return []
        
        # Phase 2: Apply PCA (40-70% progress)
        def pca_progress(current, total, status=None):
            if progress_callback:
                overall_progress = int(40 + (current / total) * 30)
                progress_callback(overall_progress, 100, status or f"Applying PCA to tile {current}/{total}")
        
        pca_results = self.apply_pca_to_embeddings(
            embeddings, n_components, standardize, pca_progress
        )
        
        # Phase 3: Export GeoTIFFs (70-100% progress)
        created_files = []
        total_tiles = len(pca_results)
        
        # Calculate global min/max if normalize is True (for consistent scaling across tiles)
        if normalize:
            # Global normalization: find min/max across ALL tiles first
            global_min = [float('inf')] * n_components
            global_max = [float('-inf')] * n_components
            
            for _, _, pca_img, _, _, _ in pca_results:
                for j in range(n_components):
                    comp = pca_img[:, :, j]
                    global_min[j] = min(global_min[j], np.nanmin(comp))
                    global_max[j] = max(global_max[j], np.nanmax(comp))
        
        for i, (tile_lon, tile_lat, pca_image, crs, transform, pca_info) in enumerate(pca_results):
            if progress_callback:
                export_progress = int(70 + (i / total_tiles) * 30)
                filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}_pca.tif"
                progress_callback(export_progress, 100, f"Writing {filename}...")
            
            # Always normalize PCA components to 0-255 for visualization
            pca_normalized = np.zeros_like(pca_image)
            for j in range(n_components):
                component = pca_image[:, :, j]
                
                if normalize:
                    # Use global min/max for consistent scaling across tiles
                    comp_min, comp_max = global_min[j], global_max[j]
                else:
                    # Use local min/max for per-tile normalization
                    comp_min, comp_max = np.nanmin(component), np.nanmax(component)
                
                if comp_max > comp_min:
                    pca_normalized[:, :, j] = (component - comp_min) / (comp_max - comp_min)
                else:
                    pca_normalized[:, :, j] = 0
            
            # Always convert to uint8 for visualization output
            output_data = (np.clip(pca_normalized, 0, 1) * 255).astype(np.uint8)
            dtype = 'uint8'
            
            # Create filename and path
            filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}_pca.tif"
            output_path = output_dir / filename
            
            # Get dimensions
            height, width = output_data.shape[:2]
            
            # Write PCA GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=n_components,
                dtype=dtype,
                crs=crs,
                transform=transform,
                compress=compress,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                # Write each component as a band
                for band_idx in range(n_components):
                    dst.write(output_data[:, :, band_idx], band_idx + 1)
                    
                    # Set band description with explained variance
                    variance_pct = pca_info['explained_variance'][band_idx] * 100
                    dst.set_band_description(
                        band_idx + 1,
                        f"PC{band_idx + 1} ({variance_pct:.1f}% variance)"
                    )
                
                # Set color interpretation for RGB visualization
                if n_components >= 3 and normalize:
                    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue][:n_components]
                
                # Add metadata
                dst.update_tags(
                    TESSERA_YEAR=str(year),
                    TESSERA_TILE_LON=str(tile_lon),
                    TESSERA_TILE_LAT=str(tile_lat),
                    PCA_COMPONENTS=str(n_components),
                    PCA_TOTAL_VARIANCE=f"{pca_info['total_variance']:.3f}",
                    PCA_EXPLAINED_VARIANCE=json.dumps(pca_info['explained_variance']),
                    PCA_STANDARDIZED=str(standardize),
                    PCA_NORMALIZED=str(normalize),
                    GEOTESSERA_VERSION=__version__,
                )
            
            created_files.append(str(output_path))
        
        if progress_callback:
            progress_callback(100, 100, f"Created {len(created_files)} PCA GeoTIFF files")
        
        return created_files
