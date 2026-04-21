GeoTessera Documentation
========================

GeoTessera provides access to open geospatial embeddings from the `Tessera foundation model <https://github.com/ucam-eo/tessera>`_
(`paper <https://arxiv.org/abs/2506.20380>`_). Tessera processes Sentinel-1 and
Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m
resolution, compressing a full year of temporal-spectral features into dense
representations optimized for downstream geospatial analysis tasks.

Overview
--------

GeoTessera is built around a two-step workflow:

1. **Retrieve embeddings**: Fetch raw numpy arrays with CRS/transform information for a geographic bounding box
2. **Export to desired format**: Save as raw numpy arrays or convert to georeferenced GeoTIFF files with preserved projections

Key Features
------------

* **Global Coverage**: Access embeddings for any terrestrial location worldwide where data exists
* **Flexible Formats**: Export as numpy arrays for analysis or GeoTIFF for GIS integration
* **Cloud-Native Zarr Access**: Stream embeddings directly via ``GeoTesseraZarr`` without downloading files
* **Projection Preservation**: Native UTM projections preserved from landmask tiles
* **High Resolution**: 10m spatial resolution
* **Temporal Compression**: Full year of satellite observations in each embedding
* **Multi-spectral**: Combines Sentinel-1 SAR and Sentinel-2 optical data
* **Country Support**: Download by country name or custom regions
* **Resume Capability**: Both TIFF and NPY downloads skip existing files automatically
* **Efficient Registry**: Block-based lazy loading of only required data
* **Easy Access**: Python API and CLI with automatic caching

Installation
------------

Requires Python 3.12 or later. Install GeoTessera using pip::

    pip install geotessera

For development installation::

    git clone https://github.com/ucam-eo/geotessera
    cd geotessera
    pip install -e .

Quick Start
-----------

Check data availability first::

    # Generate coverage visualizations (creates PNG map, JSON data, and interactive HTML globe)
    geotessera coverage --output coverage_map.png
    # Creates: coverage_map.png, coverage.json, globe.html

    # View coverage for a specific year
    geotessera coverage --year 2024

    # Check coverage for a single country with precise boundary outline
    geotessera coverage --country "United Kingdom"
    geotessera coverage --country uk  # Also accepts country codes

Download embeddings in your preferred format::

    # Download as GeoTIFF (default, georeferenced, ready for GIS)
    geotessera download --bbox "-0.2,51.4,0.1,51.6" --year 2024 --output ./london_tiffs --bands 1,2,3

    # Download as quantized numpy arrays (for analysis, includes scales and landmask TIFFs)
    geotessera download --bbox "-0.2,51.4,0.1,51.6" --format npy --year 2024 --output ./london_arrays
    # NPY format includes: quantized .npy, _scales.npy, and landmask .tiff files

    # Download by country name with precise boundary filtering
    geotessera download --country "United Kingdom" --year 2024 --output ./uk_tiles

    # Download tiles from a region file (supports GeoJSON, Shapefile, or URLs)
    geotessera download --region-file example/CB.geojson --year 2024 --output ./cambridge
    geotessera download --region-file https://example.com/region.geojson --year 2024 --output ./remote_region


Python API usage::

    from geotessera import GeoTessera

    # Initialize client
    gt = GeoTessera()

    # Method 1: Fetch a single tile with CRS information
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
    print(f"CRS: {crs}")  # UTM projection

    # Method 2: Fetch all tiles in a bounding box
    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding.shape}")

    # Method 3: Sample embeddings at specific point locations
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]  # (lon, lat) tuples
    embeddings = gt.sample_embeddings_at_points(points, year=2024)
    print(f"Sampled embeddings shape: {embeddings.shape}")  # (3, 128)

    # Export as GeoTIFF files with preserved UTM projections
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
    files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./output",
        bands=[0, 1, 2]  # Export first 3 bands only
    )

Create PCA visualizations and web maps::

    # Create PCA mosaic from GeoTIFFs
    geotessera visualize ./london_tiffs pca_mosaic.tif

    # Create web tiles and serve interactively
    geotessera webmap pca_mosaic.tif --serve

Architecture Overview
---------------------

Coordinate System and Tile Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings use a **0.1-degree grid system**:

* **Tile size**: Each tile covers 0.1° × 0.1° (approximately 11km × 11km at the equator)
* **Tile naming**: Tiles are named by their **center coordinates** (e.g., ``grid_0.15_52.05``)
* **Tile bounds**: A tile at center (lon, lat) covers [lon ± 0.05°, lat ± 0.05°]
* **Resolution**: 10m per pixel (variable pixels per tile depending on latitude)

File Structure and Downloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you request embeddings, GeoTessera downloads files directly via HTTP to temporary locations:

**Embedding Files** (via ``fetch_embedding``):

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):

   * Shape: ``(height, width, 128)``
   * Data type: int8 (quantized for storage efficiency)
   * Contains the compressed embedding values

2. **Scale files** (``grid_X.XX_Y.YY_scales.npy``):

   * Shape: ``(height, width)`` or ``(height, width, 128)``
   * Data type: float32
   * Contains scale factors for dequantization

3. **Dequantization**: ``final_embedding = quantized_embedding * scales``

4. **Temporary Storage**: Files are downloaded to temp locations and automatically cleaned up after processing

**Landmask Files** (with CRS and masks for GeoTIFF export):

* **Landmask tiles** (``grid_X.XX_Y.YY.tiff``):

  * Provide UTM projection information
  * Define precise geospatial transforms
  * Contain land/water masks
  * Also downloaded to temp locations and cleaned up after use

The geotessera CLI can also export these into GeoTIFF format with each band
dequantised into 128-bands and with the GeoTIFF CRS metadata intact.

Data Flow
~~~~~~~~~

::

    User Request (lat/lon bbox)
        ↓
    Parquet Registry Lookup (find available tiles from registry.parquet)
        ↓
    Direct HTTP Downloads to Temp Files
        ├── embedding.npy (quantized) → temp file
        └── embedding_scales.npy → temp file
        ↓
    Dequantization (multiply arrays)
        ↓
    Automatic Cleanup (delete temp files)
        ↓
    Output Format
        ├── NumPy arrays → Direct analysis
        └── GeoTIFF → GIS integration

**Storage Note**: Only the Parquet registry (~few MB) is cached locally. All embedding data
is downloaded on-demand to temporary files and immediately cleaned up, resulting in zero
persistent storage overhead for tile data.

Registry System
~~~~~~~~~~~~~~~

GeoTessera uses a Parquet-based registry system for efficient data access:

* **Single Parquet file**: All tile metadata stored in one efficient ``registry.parquet`` file
* **Fast queries**: Uses pandas DataFrames for efficient spatial and temporal filtering
* **Block-based organization**: Internal 5×5 degree geographic blocks for efficient queries
* **Minimal storage**: Registry file is ~few MB and cached locally
* **Integrity checking**: SHA256 checksums ensure data integrity during downloads

The registry can be loaded from multiple sources:

1. **Default remote** (recommended, downloads and caches automatically)
2. **Local file** (via ``--registry-path`` parameter)
3. **Local directory** (via ``--registry-dir`` parameter, looks for ``registry.parquet``)
4. **Custom URL** (via ``--registry-url`` parameter)

Understanding Tessera Embeddings
--------------------------------

Each embedding tile:

* Covers a 0.1° × 0.1° area (approximately 11km × 11km at equator)
* Contains 128 channels of learned features per pixel
* Represents patterns from a full year of satellite observations
* Is stored in quantized format for efficient transmission and storage

The 128 channels capture various environmental features learned by the
Tessera foundation model, including vegetation patterns, water bodies,
urban structures, and seasonal changes.

Data Organization
-----------------

**Remote Server Structure**::

    https://s3.us-west-2.amazonaws.com/tessera-embeddings/
    ├── v1/                              # Dataset version
    │   ├── registry.parquet             # Parquet registry with all metadata
    │   ├── 2024/                        # Year
    │   │   ├── grid_0.15_52.05/         # Tile (named by center coords)
    │   │   │   ├── grid_0.15_52.05.npy              # Quantized embeddings
    │   │   │   └── grid_0.15_52.05_scales.npy       # Scale factors
    │   │   └── ...
    │   └── landmasks/
    │       ├── grid_0.15_52.05.tiff     # Landmask with projection info
    │       └── ...

**Local Cache Structure**::

    ~/.cache/geotessera/                 # Default cache location
    └── registry.parquet                 # Cached Parquet registry (~few MB)

    # Note: Embedding and landmask tiles are NOT cached persistently.
    # They are downloaded to temporary files and immediately cleaned up after use.

Embeddings are organized by:

* **Year**: 2017-2025 (depending on availability)
* **Location**: Global 0.1-degree grid system
* **Format**: NumPy arrays with shape (height, width, 128)

Cache Configuration
-------------------

Control where the Parquet registry is cached::

    from geotessera import GeoTessera

    # Use custom cache directory for registry
    gt = GeoTessera(cache_dir="/path/to/cache")

    # Use default cache location (recommended)
    gt = GeoTessera()

Or via CLI::

    # Specify custom cache directory
    geotessera download --cache-dir /path/to/cache ...

    # Use default cache location
    geotessera download ...

Default cache locations (when not specified):

* **Linux/macOS**: ``~/.cache/geotessera/``
* **Windows**: ``%LOCALAPPDATA%/geotessera/``

Documentation Sections
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   quickstart
   architecture
   tutorials
   cli_reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:
   
   GitHub Repository <https://github.com/ucam-eo/geotessera>
   Tessera Model <https://github.com/ucam-eo/tessera>
   Issue Tracker <https://github.com/ucam-eo/geotessera/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
