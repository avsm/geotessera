Architecture Guide
=================

This guide explains the internal architecture of GeoTessera and how the various components work together to provide efficient access to Tessera embeddings.

Overview
--------

GeoTessera is designed around a simple but powerful architecture that optimizes for:

- **Efficient data access**: Only download what you need
- **Projection preservation**: Maintain native UTM projections for accuracy
- **Scalability**: Handle large datasets with lazy loading
- **Flexibility**: Support both analysis and GIS workflows
- **Reliability**: Ensure data integrity with checksums

Core Architecture
-----------------

The library follows a layered architecture:

.. code-block::

    User Interface Layer
    ├── CLI Commands (geotessera download, visualize, etc.)
    └── Python API (GeoTessera class)
            ↓
    Core Processing Layer
    ├── GeoTessera class (main interface)
    ├── Registry (Parquet-based data discovery)
    └── Visualization (rendering and web maps)
            ↓
    Data Access Layer
    ├── Direct HTTP downloads (urllib)
    ├── Zarr v3 store (cloud-native streaming)
    ├── Rasterio (GeoTIFF I/O)
    └── GeoPandas (geospatial operations)
            ↓
    Storage Layer
    ├── Remote servers (https://s3.us-west-2.amazonaws.com/tessera-embeddings)
    ├── Zarr store (https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1/zarr)
    └── Local cache (~/.cache/geotessera/registry.parquet)

Coordinate System and Grid
--------------------------

Understanding the Tessera Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings are organized on a **0.1-degree grid system**:

**Grid Properties**:

- **Grid spacing**: 0.1° latitude × 0.1° longitude
- **Tile naming**: Named by center coordinates (e.g., ``grid_0.15_52.05``)
- **Coverage**: Each tile spans from (center - 0.05°) to (center + 0.05°)
- **Resolution**: Approximately 11km × 11km at the equator

**Coordinate Calculations**::

    # For a tile at center coordinates (lon, lat)
    west = lon - 0.05
    east = lon + 0.05  
    south = lat - 0.05
    north = lat + 0.05

**Grid Alignment**:

Tile centers are aligned to 0.1-degree boundaries::

    # Valid tile centers (examples)
    valid_centers = [
        (0.05, 52.05),   # Northwest Europe
        (0.15, 52.05),   # Adjacent tile
        (-0.05, 51.95),  # Southwest tile
    ]
    
    # Invalid centers (not on grid)
    invalid_centers = [
        (0.07, 52.03),   # Off-grid
        (0.1, 52.1),     # Off by 0.05°
    ]

Resolution and Pixel Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of pixels per tile varies with latitude due to the Earth's curvature:

.. code-block:: python

    import math
    
    def pixels_per_tile(latitude, resolution_meters=10):
        """Calculate approximate pixels per tile at given latitude."""
        # Earth circumference at equator (meters)
        earth_circumference = 40075000
        
        # Degrees per meter at equator
        degrees_per_meter = 360 / earth_circumference
        
        # Adjust for latitude (longitude only)
        lon_degrees_per_meter = degrees_per_meter / math.cos(math.radians(latitude))
        lat_degrees_per_meter = degrees_per_meter
        
        # Tile size in meters
        tile_width_meters = 0.1 / lon_degrees_per_meter
        tile_height_meters = 0.1 / lat_degrees_per_meter
        
        # Pixels in tile
        pixels_width = int(tile_width_meters / resolution_meters)
        pixels_height = int(tile_height_meters / resolution_meters)
        
        return pixels_width, pixels_height
    
    # Examples
    eq_pixels = pixels_per_tile(0)      # ~(1111, 1111) at equator
    uk_pixels = pixels_per_tile(52)     # ~(1823, 1111) in UK
    arctic_pixels = pixels_per_tile(80) # ~(6389, 1111) near poles

Data Format and Storage
-----------------------

Quantization System
~~~~~~~~~~~~~~~~~~~

Tessera embeddings are stored using a quantization system for efficiency:

**Storage Format**:

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):
   
   - Data type: ``int8`` (values -128 to 127)
   - Shape: ``(height, width, 128)``
   - Storage efficient: ~1MB per tile vs ~64MB unquantized

2. **Scale factors** (``grid_X.XX_Y.YY_scales.npy``):
   
   - Data type: ``float32``
   - Shape: ``(height, width)`` or ``(height, width, 128)``
   - Contains dequantization multipliers

**Dequantization Process**::

    import numpy as np
    
    # Load quantized data and scales
    quantized = np.load("grid_0.15_52.05.npy")         # int8
    scales = np.load("grid_0.15_52.05_scales.npy")     # float32
    
    # Dequantize
    if scales.ndim == 2:
        # Broadcast 2D scales to 3D
        scales = scales[..., np.newaxis]
    
    embedding = quantized.astype(np.float32) * scales
    
    # Result: (height, width, 128) float32 array

This process is handled automatically by ``GeoTessera.fetch_embedding()``, which now returns the dequantized embedding along with CRS and transform information from the corresponding landmask tile.

Metadata and Projections
~~~~~~~~~~~~~~~~~~~~~~~~

**Landmask Files** (``grid_X.XX_Y.YY.tiff``):

- Provide native UTM projection information for each tile
- Define precise geospatial transforms (no reprojection needed)
- Preserve original coordinate system for maximum accuracy
- Used for georeferencing when exporting to GeoTIFF
- Contain binary land/water masks

**Projection Selection**:

Each tile uses an appropriate UTM zone based on its location::

    def get_utm_zone(longitude):
        """Get UTM zone number for a longitude."""
        return int((longitude + 180) / 6) + 1
    
    def get_utm_epsg(longitude, latitude):
        """Get EPSG code for UTM projection."""
        zone = get_utm_zone(longitude)
        
        if latitude >= 0:
            # Northern hemisphere
            return f"EPSG:{32600 + zone}"
        else:
            # Southern hemisphere  
            return f"EPSG:{32700 + zone}"
    
    # Example: London at (0.15, 52.05)
    epsg = get_utm_epsg(0.15, 52.05)  # "EPSG:32631" (UTM Zone 31N)

Registry System
---------------

Parquet-Based Registry
~~~~~~~~~~~~~~~~~~~~~~

The registry uses a **Parquet file** for efficient data discovery and querying:

**Registry Structure**:

.. code-block::

    registry.parquet (single file with all metadata)
    ├── Columns:
    │   ├── lon, lat         # Tile center coordinates
    │   ├── year             # Data year (2017-2025)
    │   ├── hash             # SHA256 file integrity checksum
    │   ├── scales_hash      # SHA256 checksum for scales files
    │   └── file_size        # File size in bytes
    └── Rows: One per tile

**Querying the Registry**::

    import pandas as pd

    # Load registry
    registry = pd.read_parquet("registry.parquet")

    # Query tiles in a region
    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles = registry[
        (registry['lon'] >= bbox[0]) & (registry['lon'] <= bbox[2]) &
        (registry['lat'] >= bbox[1]) & (registry['lat'] <= bbox[3]) &
        (registry['year'] == 2024)
    ]

    # Examples of block-based filtering (internal optimization)
    def get_block_coordinates(lon, lat):
        """Get the 5x5 degree block coordinates for a point."""
        # Round down to nearest 5-degree boundary
        block_lon = int(lon // 5) * 5
        block_lat = int(lat // 5) * 5
        return block_lon, block_lat

**Registry Loading Process**:

1. **Download Parquet registry** (if not cached locally, ~few MB)
2. **Query tiles** for the requested bounding box using pandas
3. **Filter by year** if specified
4. **Return matching tiles** as a DataFrame
5. **Cache registry** in memory for subsequent requests

Registry Sources
~~~~~~~~~~~~~~~~

The registry can be loaded from multiple sources:

**1. Default Remote** (recommended)::

    # Downloads and caches registry.parquet automatically
    from geotessera import GeoTessera
    gt = GeoTessera()

    # Cached at: ~/.cache/geotessera/registry.parquet

**2. Local File**::

    gt = GeoTessera(registry_path="/path/to/registry.parquet")

**3. Local Directory**::

    # Looks for registry.parquet in the directory
    gt = GeoTessera(registry_dir="/path/to/registry-dir")

**4. Custom URL**::

    gt = GeoTessera(registry_url="https://example.com/registry.parquet")

**5. CLI Option**::

    geotessera download --cache-dir /custom/cache ...

Data Access Layer
-----------------

Direct HTTP Downloads
~~~~~~~~~~~~~~~~~~~~~

GeoTessera uses direct HTTP downloads with temporary file handling:

**Features**:

- **Zero persistent storage**: Tiles downloaded to temp files and cleaned up immediately
- **Integrity checking**: SHA256 verification for all downloads
- **Progress callbacks**: Real-time download feedback with speed and size info
- **Human-readable progress**: Download speeds shown in KB/s, MB/s format
- **Automatic cleanup**: try/finally blocks ensure no leftover temp files

**Cache Structure**::

    ~/.cache/geotessera/
    └── registry.parquet             # Only the registry is cached (~few MB)

    # Note: Embedding and landmask tiles are NOT cached
    # They are downloaded to temporary files and deleted after use

**Download Process**::

    import tempfile
    from urllib.request import urlopen
    from geotessera import dequantize_embedding

    def fetch_embedding(lon, lat, year):
        # 1. Query registry for tile metadata
        tile_info = registry.query_tile(lon, lat, year)

        # 2. Download to temporary files (or use local if exists in embeddings_dir)
        embedding_file, cleanup_embedding = registry.fetch(
            year=year, lon=lon, lat=lat, is_scales=False
        )
        scales_file, cleanup_scales = registry.fetch(
            year=year, lon=lon, lat=lat, is_scales=True
        )

        try:
            # 3. Load and dequantize
            quantized = np.load(embedding_file)
            scales = np.load(scales_file)
            embedding = dequantize_embedding(quantized, scales)

            # 4. Get CRS from landmask (also temporary or local)
            crs, transform = get_utm_projection_from_landmask(lon, lat)

            return embedding, crs, transform

        finally:
            # 5. Clean up temporary files (if they were temporary)
            if cleanup_embedding:
                Path(embedding_file).unlink(missing_ok=True)
            if cleanup_scales:
                Path(scales_file).unlink(missing_ok=True)

Temporary File Management
~~~~~~~~~~~~~~~~~~~~~~~~~

**Why Temporary Files?**

- Embedding tiles can be large (1-64MB per tile)
- Users typically process and export, not reuse raw tiles
- Eliminates need for cache management and cleanup
- Reduces disk space requirements to just the registry

**Cache Configuration**::

    from geotessera import GeoTessera

    # Control where registry is cached
    gt = GeoTessera(cache_dir="/custom/cache")

    # Default cache locations:
    # - Linux/macOS: ~/.cache/geotessera/
    # - Windows: %LOCALAPPDATA%/geotessera/

GeoTIFF Export Process
~~~~~~~~~~~~~~~~~~~~~~

When exporting to GeoTIFF, additional processing occurs:

**Export Workflow**:

1. **Fetch embedding data** (quantized + scales)
2. **Fetch landmask tile** for projection information  
3. **Extract native UTM projection** and transform from landmask
4. **Apply dequantization** to embedding data
5. **Preserve original coordinate system** (no reprojection)
6. **Select bands** (if specified)
7. **Write GeoTIFF** with native UTM CRS and accurate transform
8. **Apply compression** (LZW, DEFLATE, etc.)

**Projection Inheritance**::

    import rasterio
    
    def export_geotiff(embedding, landmask_path, output_path, bands=None):
        # Read projection from landmask
        with rasterio.open(landmask_path) as landmask:
            crs = landmask.crs
            transform = landmask.transform
            
        # Select bands
        if bands:
            embedding = embedding[:, :, bands]
            
        # Write GeoTIFF
        with rasterio.open(output_path, 'w',
                          driver='GTiff',
                          height=embedding.shape[0],
                          width=embedding.shape[1], 
                          count=embedding.shape[2],
                          dtype=embedding.dtype,
                          crs=crs,
                          transform=transform,
                          compress='lzw') as dst:
            
            for i in range(embedding.shape[2]):
                dst.write(embedding[:, :, i], i + 1)

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

**Large Region Handling**:

When processing large regions, GeoTessera uses several strategies:

- **Tile-by-tile processing**: Process one tile at a time to limit memory usage
- **Band selection**: Only load required bands to reduce memory footprint  
- **Generator patterns**: Use generators for large tile collections
- **Progress callbacks**: Provide feedback for long operations

**Example Memory-Efficient Processing**::

    def process_large_region(bbox, year, bands=None):
        """Process a large region without loading all tiles into memory."""
        gt = GeoTessera()

        # Step 1: Get tile list (metadata only, no data loaded)
        tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)

        # Step 2: Process tiles one at a time using generator
        for year, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(tiles_to_fetch):
            # Apply band selection early to reduce memory
            if bands:
                embedding = embedding[:, :, bands]

            # Process this tile
            result = process_single_tile(embedding)

            # Save or accumulate results
            save_tile_result(result, tile_lat, tile_lon)

            # Free memory
            del embedding

Network Optimization
~~~~~~~~~~~~~~~~~~~~

**Sequential Processing**:

The fetch_embeddings() generator processes tiles sequentially, which is optimal for most use cases::

    # Sequential processing (recommended for most cases)
    gt = GeoTessera()
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Returns generator - tiles are fetched one at a time
    for year, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(tiles_to_fetch):
        process_tile(embedding)  # Memory efficient

**Point Sampling**:

For sampling at specific locations, use the optimized point sampling method::

    # Efficient point sampling with automatic tile download
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]
    embeddings = gt.sample_embeddings_at_points(points, year=2024)

    # With metadata about which tile each point came from
    embeddings, metadata = gt.sample_embeddings_at_points(
        points, year=2024, include_metadata=True
    )

**Cache Efficiency**:

- **Pre-warming**: Download commonly used tiles in advance
- **Batch processing**: Group requests by geographic region
- **Size limits**: Respect server rate limits

Zarr Store (Cloud-Native Access)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GeoTesseraZarr`` class provides cloud-native access to embeddings
without downloading files. It implements the ``geoemb:`` convention for
geospatial embedding data stored in Zarr v3 format.

**Architecture**:

Data is organized by UTM zone, with each zone stored as a separate Zarr
group. The store automatically routes geographic queries to the correct zone::

    zarr store
    ├── Root attributes (geoemb:model, geoemb:build_version)
    ├── utm30/           # UTM Zone 30
    │   ├── time[:]      # Year coordinate array
    │   ├── embedding    # (time, y, x, band) float32
    │   └── ...
    ├── utm31/           # UTM Zone 31
    │   └── ...
    └── ...

**Access patterns**:

- **Point sampling**: ``sample_points()`` / ``sample_at()`` for extracting
  embeddings at specific coordinates across zones
- **Region reading**: ``read_region()`` for loading rectangular areas as
  mosaics with CRS and transform metadata
- **Zone access**: ``open_zone()`` returns an xarray Dataset with a
  ``.tessera`` accessor for direct manipulation

Datasets are cached per zone for the lifetime of the ``GeoTesseraZarr``
instance.

Future Extensions
~~~~~~~~~~~~~~~~~

The architecture supports future enhancements:

- **Temporal queries**: Multi-year analysis
- **Cloud optimization**: Direct cloud storage access
- **ML integration**: TensorFlow/PyTorch data loaders
- **Real-time updates**: Live data ingestion
- **Distributed processing**: Dask/Ray integration
