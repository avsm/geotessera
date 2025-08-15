"""Data processing utilities for GeoTessera embeddings.

This module provides functions for processing and analyzing Tessera embeddings,
including point extraction, region-based merging, and data analysis operations.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import tempfile
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from .registry import world_to_tile_coords


def extract_points_from_embeddings(
    gt,  # GeoTessera instance
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
        gt: GeoTessera instance for data access
        points: Points with 'lat'/'lon' columns/keys. Can include 'label' or other metadata.
               Accepts list of dicts, pandas DataFrame, or GeoDataFrame.
        year: Year of embeddings to use
        include_coords: If True, includes lat/lon in output DataFrame
        progressbar: Show progress bar during extraction

    Returns:
        DataFrame with embeddings (128 columns named 'emb_0' to 'emb_127') plus any metadata from input.
        Points that fall outside available tiles will be excluded from results.
    """
    import rasterio
    from pyproj import Transformer

    # Convert input to consistent format
    if isinstance(points, list):
        points_df = pd.DataFrame(points)
    elif isinstance(points, gpd.GeoDataFrame):
        # Extract coordinates from geometry if needed
        if "lat" not in points.columns or "lon" not in points.columns:
            points = points.copy()
            points["lon"] = points.geometry.x
            points["lat"] = points.geometry.y
        points_df = pd.DataFrame(points.drop(columns="geometry"))
    else:
        points_df = points.copy()

    # Validate required columns
    if "lat" not in points_df.columns or "lon" not in points_df.columns:
        raise ValueError("Input must have 'lat' and 'lon' columns")

    # Group points by their containing tiles
    points_by_tile = defaultdict(list)
    for idx, point in points_df.iterrows():
        # Convert world coordinates to correct tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(point["lat"], point["lon"])
        
        # Round coordinates to handle floating-point precision issues
        tile_lat = round(tile_lat, 2)
        tile_lon = round(tile_lon, 2)
        
        points_by_tile[(tile_lat, tile_lon)].append((idx, point))

    # Process points tile by tile
    results = []
    processed_indices = set()

    if progressbar:
        tile_iterator = tqdm(points_by_tile.items(), desc="Processing tiles")
    else:
        tile_iterator = points_by_tile.items()

    for (tile_lat, tile_lon), tile_points in tile_iterator:
        try:
            # Ensure the block is loaded first
            gt.registry.ensure_block_loaded(year, tile_lon, tile_lat)
            
            # Skip if tile doesn't exist
            if not any(
                t_year == year and abs(t_lat - tile_lat) < 0.01 and abs(t_lon - tile_lon) < 0.01
                for t_year, t_lat, t_lon in gt.registry.available_embeddings
            ):
                continue

            # Fetch embedding and landmask for georeferencing
            embedding = gt.fetch_embedding(
                tile_lat, tile_lon, year, progressbar=False
            )
            landmask_path = gt._fetch_landmask(
                tile_lat, tile_lon, progressbar=False
            )

            # Get georeferencing info
            with rasterio.open(landmask_path) as src:
                transformer = Transformer.from_crs(
                    "EPSG:4326", src.crs, always_xy=True
                )
                h, w = src.height, src.width

                # Process each point
                for idx, point in tile_points:
                    if (
                        idx in processed_indices
                    ):  # Skip if already processed by another tile
                        continue

                    # Transform coordinates
                    px, py = transformer.transform(point["lon"], point["lat"])
                    row, col = src.index(px, py)

                    # Check if point is within tile bounds
                    if 0 <= row < h and 0 <= col < w:
                        # Extract embedding vector
                        emb_vector = embedding[row, col]

                        # Build result row
                        result = {f"emb_{i}": emb_vector[i] for i in range(128)}

                        # Add metadata
                        for col_name in points_df.columns:
                            if col_name not in ["lat", "lon"] or include_coords:
                                result[col_name] = point[col_name]

                        results.append(result)
                        processed_indices.add(idx)

        except Exception as e:
            if progressbar:
                print(
                    f"\\nWarning: Failed to process tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}"
                )
            continue

    if not results:
        print(
            "Warning: No embeddings were extracted. Check that points fall within available tiles."
        )
        return pd.DataFrame()

    # Create DataFrame with consistent column order
    results_df = pd.DataFrame(results)

    # Reorder columns: metadata first, then embeddings
    emb_cols = [f"emb_{i}" for i in range(128)]
    metadata_cols = [col for col in results_df.columns if col not in emb_cols]
    results_df = results_df[metadata_cols + emb_cols]

    if progressbar:
        print(
            f"Successfully extracted embeddings for {len(results_df)}/{len(points_df)} points"
        )

    return results_df


def merge_embeddings_for_region(
    gt,  # GeoTessera instance
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
        gt: GeoTessera instance for data access
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
        >>> merge_embeddings_for_region(
        ...     gt,
        ...     bounds=(-0.2, 51.4, 0.1, 51.6),
        ...     output_path="london_full_128band.tif"
        ... )

        >>> # Create selected band subset
        >>> merge_embeddings_for_region(
        ...     gt,
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
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject
        from rasterio.enums import Resampling
        from rasterio.merge import merge
    except ImportError:
        raise ImportError(
            "Please install rasterio for embedding merging: pip install rasterio"
        )

    min_lon, min_lat, max_lon, max_lat = bounds

    # Determine output configuration based on bands parameter
    if bands is None:
        # All 128 bands
        output_bands = None  # Will use all bands
        num_bands = 128
        print("Exporting all 128 bands")
    else:
        # Selected bands mode
        output_bands = bands
        num_bands = len(bands)
        print(f"Exporting {num_bands} selected bands")

        # Validate band indices
        if any(b < 0 or b > 127 for b in bands):
            raise ValueError("All band indices must be in range 0-127")

    # Load only the registry blocks needed for this region (much more efficient)
    gt.registry.load_blocks_for_region(bounds, year)

    # Find all embedding tiles that intersect with the bounds
    tiles_to_merge = []
    for emb_year, lat, lon in gt.registry.available_embeddings:
        if emb_year != year:
            continue

        # Check if tile intersects with bounds (tiles are centered on 0.05 grid)
        tile_min_lon, tile_min_lat = lon - 0.05, lat - 0.05
        tile_max_lon, tile_max_lat = lon + 0.05, lat + 0.05

        if (
            tile_min_lon < max_lon
            and tile_max_lon > min_lon
            and tile_min_lat < max_lat
            and tile_max_lat > min_lat
        ):
            tiles_to_merge.append((lat, lon))

    if not tiles_to_merge:
        raise ValueError(
            f"No embedding tiles found for the specified region in year {year}"
        )

    print(f"Found {len(tiles_to_merge)} embedding tiles to merge for year {year}")

    # Create temporary directory for georeferenced TIFF files
    temp_dir = tempfile.mkdtemp(prefix="geotessera_embed_merge_")

    try:
        # Step 1: Create properly georeferenced temporary TIFF files
        temp_tiff_paths = []

        for lat, lon in tiles_to_merge:
            try:
                # Get the numpy embedding
                embedding = gt.fetch_embedding(lat, lon, year, progressbar=True)

                # Get the corresponding landmask GeoTIFF for coordinate information
                # The landmask TIFF provides the optimal projection metadata for this tile
                landmask_path = gt._fetch_landmask(lat, lon, progressbar=False)

                # Read coordinate information from the landmask GeoTIFF metadata
                with rasterio.open(landmask_path) as landmask_src:
                    src_transform = landmask_src.transform
                    src_crs = landmask_src.crs
                    src_bounds = landmask_src.bounds
                    src_height, src_width = landmask_src.height, landmask_src.width

                # Extract the specified bands
                if output_bands is None:
                    # Use all 128 bands
                    vis_data = embedding.copy()
                else:
                    # Use selected bands
                    vis_data = embedding[:, :, output_bands].copy()

                # Ensure data is float32 (it should already be from fetch_embedding)
                vis_data = vis_data.astype(np.float32)

                # Create temporary georeferenced TIFF file
                temp_tiff_path = Path(temp_dir) / f"embed_{lat:.2f}_{lon:.2f}.tiff"

                # Handle potential coordinate system differences and reprojection
                if str(src_crs) != str(target_crs):
                    # Calculate transform for reprojection
                    dst_transform, dst_width, dst_height = (
                        calculate_default_transform(
                            src_crs,
                            target_crs,
                            src_width,
                            src_height,
                            left=src_bounds.left,
                            bottom=src_bounds.bottom,
                            right=src_bounds.right,
                            top=src_bounds.top,
                        )
                    )

                    # Create reprojected array
                    dst_data = np.zeros(
                        (dst_height, dst_width, num_bands), dtype=np.float32
                    )

                    # Reproject each band
                    for i in range(num_bands):
                        reproject(
                            source=vis_data[:, :, i],
                            destination=dst_data[:, :, i],
                            src_transform=src_transform,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,  # Use bilinear for smoother results
                        )

                    # Use reprojected data
                    final_data = dst_data
                    final_transform = dst_transform
                    final_crs = target_crs
                    final_height, final_width = dst_height, dst_width
                else:
                    # Use original coordinate system
                    final_data = vis_data
                    final_transform = src_transform
                    final_crs = src_crs
                    final_height, final_width = vis_data.shape[:2]

                # Write georeferenced TIFF file as float32
                with rasterio.open(
                    temp_tiff_path,
                    "w",
                    driver="GTiff",
                    height=final_height,
                    width=final_width,
                    count=num_bands,
                    dtype="float32",
                    crs=final_crs,
                    transform=final_transform,
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                ) as dst:
                    for i in range(num_bands):
                        dst.write(final_data[:, :, i], i + 1)

                temp_tiff_paths.append(str(temp_tiff_path))

            except Exception as e:
                # All errors during tile processing should be fatal
                raise RuntimeError(
                    f"Failed to process embedding tile ({lat}, {lon}): {e}"
                ) from e

        if not temp_tiff_paths:
            raise ValueError("No embedding tiles could be processed")

        print(f"Created {len(temp_tiff_paths)} temporary georeferenced TIFF files")

        # Step 2: Use rasterio.merge to properly merge the georeferenced TIFF files
        print("Merging georeferenced TIFF files...")

        # Open all TIFF files for merging
        src_files = [rasterio.open(path) for path in temp_tiff_paths]

        try:
            # Merge the files
            merged_array, merged_transform = merge(src_files, method="first")

            # Ensure the merged array is float32
            final_array = merged_array.astype(np.float32)

            # Write the merged result
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=final_array.shape[1],
                width=final_array.shape[2],
                count=final_array.shape[0],
                dtype="float32",
                crs=target_crs,
                transform=merged_transform,
                compress="lzw",
            ) as dst:
                dst.write(final_array)

            print(f"Merged embeddings saved to: {output_path}")
            print(
                f"Dimensions: {final_array.shape[2]}x{final_array.shape[1]} pixels, {final_array.shape[0]} bands"
            )
            print("Data type: float32")

            return output_path

        finally:
            # Close all source files
            for src in src_files:
                src.close()

    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)


def merge_embeddings_for_region_file(
    gt,  # GeoTessera instance
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
        gt: GeoTessera instance for data access
        region_path: Path to region file (GeoJSON, Shapefile, etc.)
        output_path: Output path for the merged GeoTIFF
        target_crs: Target coordinate system (default: "EPSG:4326")
        bands: Band indices to include, or None for all bands
        year: Year of embeddings to use

    Returns:
        Path to created TIFF file, or None on error

    Example:
        >>> tessera = GeoTessera()
        >>> result = merge_embeddings_for_region_file(
        ...     tessera,
        ...     region_path="study_area.geojson",
        ...     output_path="embeddings.tiff",
        ...     bands=[0, 1, 2]
        ... )
    """
    try:
        from .io import load_roi

        # Load region using the robust load_roi function
        gdf = load_roi(region_path)
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = bounds

        print(
            f"Region bounds: ({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})"
        )
        print(f"Region contains {len(gdf)} feature(s)")

        # Use the existing merge method with bounds
        output = merge_embeddings_for_region(
            gt,
            bounds=(min_lon, min_lat, max_lon, max_lat),
            output_path=output_path,
            target_crs=target_crs,
            bands=bands,
            year=year,
        )

        return output

    except Exception as e:
        print(f"Error processing region file: {e}")
        import traceback

        traceback.print_exc()
        return None


def merge_landmasks_for_region(
    gt,  # GeoTessera instance
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
        gt: GeoTessera instance for data access
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
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject
        from rasterio.enums import Resampling
        from rasterio.merge import merge
    except ImportError:
        raise ImportError(
            "Please install rasterio for TIFF merging: pip install rasterio"
        )

    min_lon, min_lat, max_lon, max_lat = bounds

    # Find all land mask tiles that intersect with the bounds
    tiles_to_merge = []
    for lat, lon in gt._list_available_landmasks():
        # Check if tile intersects with bounds (tiles are centered on 0.05 grid)
        tile_min_lon, tile_min_lat = lon - 0.05, lat - 0.05
        tile_max_lon, tile_max_lat = lon + 0.05, lat + 0.05

        if (
            tile_min_lon < max_lon
            and tile_max_lon > min_lon
            and tile_min_lat < max_lat
            and tile_max_lat > min_lat
        ):
            tiles_to_merge.append((lat, lon))

    if not tiles_to_merge:
        raise ValueError("No land mask tiles found for the specified region")

    print(f"Found {len(tiles_to_merge)} land mask tiles to merge")

    # Download all required land mask tiles
    tile_paths = []
    for lat, lon in tiles_to_merge:
        try:
            tile_path = gt._fetch_landmask(lat, lon, progressbar=True)
            tile_paths.append(tile_path)
        except Exception as e:
            print(f"Warning: Could not fetch land mask tile ({lat}, {lon}): {e}")
            continue

    if not tile_paths:
        raise ValueError("No land mask tiles could be downloaded")

    # Create temporary directory for reprojected tiles
    temp_dir = tempfile.mkdtemp(prefix="geotessera_merge_")

    try:
        # Reproject all tiles to target CRS if needed
        reprojected_paths = []

        for i, tile_path in enumerate(tile_paths):
            with rasterio.open(tile_path) as src:
                if str(src.crs) != target_crs:
                    # Reproject to target CRS
                    reprojected_path = Path(temp_dir) / f"reprojected_{i}.tiff"

                    # Calculate transform and dimensions for reprojection
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds
                    )

                    # Create reprojected raster
                    with rasterio.open(
                        reprojected_path,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=target_crs,
                        transform=transform,
                        compress="lzw",
                    ) as dst:
                        for band_idx in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, band_idx),
                                destination=rasterio.band(dst, band_idx),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=target_crs,
                                resampling=Resampling.nearest,
                            )

                    reprojected_paths.append(str(reprojected_path))
                else:
                    reprojected_paths.append(tile_path)

        # Merge all reprojected tiles
        with rasterio.open(reprojected_paths[0]) as src:
            merged_array, merged_transform = merge(
                [rasterio.open(path) for path in reprojected_paths]
            )

            # Check if this appears to be a land/water mask (binary values)
            is_binary_mask = (
                merged_array.min() >= 0
                and merged_array.max() <= 1
                and merged_array.dtype in ["uint8", "int8"]
            )

            if is_binary_mask:
                print(
                    "Detected binary land/water mask - converting to visible format"
                )
                # Convert binary mask to visible grayscale (0->0, 1->255)
                display_array = (merged_array * 255).astype("uint8")
            else:
                display_array = merged_array

            # Write merged result
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=display_array.shape[1],
                width=display_array.shape[2],
                count=display_array.shape[0],
                dtype=display_array.dtype,
                crs=target_crs,
                transform=merged_transform,
                compress="lzw",
            ) as dst:
                dst.write(display_array)

        print(f"Merged land mask saved to: {output_path}")
        return output_path

    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)