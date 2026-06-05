GeoTessera Documentation
========================

GeoTessera provides access to open geospatial embeddings from the `Tessera foundation model <https://github.com/ucam-eo/tessera>`_
(`paper <https://arxiv.org/abs/2506.20380>`_). Tessera processes Sentinel-1 and
Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m
resolution, compressing a full year of temporal-spectral features into dense
representations optimized for downstream geospatial analysis tasks.

.. important::

   **Two Tessera versions are now published.** Prefer the newer **1.1** model
   wherever it's available; it's a strict improvement over the legacy 1.0
   line. The 1.1 currently runs in a ``cambridge`` variant — test embeddings
   produced by the Cambridge team while the model is being rolled out. The
   legacy 1.0 line is frozen (no new years will be added). **Never mix
   embeddings from different versions or variants in the same downstream
   task**: the 128-channel feature spaces are independently learned and not
   interchangeable. Pick one ``(dataset_version, dataset_variant)`` pair per
   project. See :ref:`dataset-versions` for full details and the CLI/Python
   flags.

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

When you request embeddings, GeoTessera downloads files from the public S3
bucket (using anonymous, unsigned requests) into your chosen output directory,
where they persist for re-use:

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

4. **Persistent Storage**: Files are downloaded into your output directory and skipped on rerun, so interrupted downloads resume cleanly

**Landmask Files** (with CRS and masks for GeoTIFF export):

* **Landmask tiles** (``grid_X.XX_Y.YY.tiff``):

  * Provide UTM projection information
  * Define precise geospatial transforms
  * Contain land/water masks
  * Cached alongside the embedding tiles for re-use

The geotessera CLI can also export these into GeoTIFF format with each band
dequantised into 128-bands and with the GeoTIFF CRS metadata intact.

Data Flow
~~~~~~~~~

::

    User Request (lat/lon bbox, dataset_version, dataset_variant)
        ↓
    Per-version Manifest Lookup (filter manifest.parquet by year/lon/lat/variant)
        ↓
    Anonymous S3 Downloads (with CRC64NVMe verification on the wire)
        ├── embedding.npy (int8 quantized) → output_dir
        └── embedding_scales.npy (float32 scale factors) → output_dir
        ↓
    Dequantization at use time: float = quantized.astype('f4') * scales
        ↓
    Output Format
        ├── NumPy arrays + tessera_metadata.json sidecar → Direct analysis
        └── GeoTIFF (with TESSERA_DATASET_VERSION/VARIANT tags) → GIS integration

**Storage Note**: Manifest + landmask Parquets (~hundreds of MB combined) are
cached per-version under ``~/.cache/geotessera/{v1,v1.1}/``. Embedding tiles
land in the user-specified ``--output`` directory (resumable across runs via
existence checks).

Manifest System
~~~~~~~~~~~~~~~

GeoTessera uses a Parquet-based per-version manifest for efficient data access:

* **One manifest per dataset version**: ``s3://tessera-embeddings/{v1,v1.1}/manifest.parquet``.
  Each carries the file-scan inventory schema (``year, lon, lat, grid_size,
  scales_size, grid_path, ...``) plus explicit ``version`` and ``variant``
  columns so a single file covers every variant in that version.
* **Fast queries**: pandas/GeoPandas DataFrames with spatial R-tree on lon/lat
* **Block-based queries**: Internal 5×5° geographic blocks keep region lookups O(blocks)
* **Conditional fetches**: Per-version ETag sidecars enable ``If-None-Match``
  conditional GETs — refetches only happen when the bucket's ETag actually
  changes; otherwise the server returns 304 with no body.
* **Integrity checking**: End-to-end CRC64NVMe verification using S3's
  ``x-amz-checksum-crc64nvme`` response header on every download.

The manifest can be loaded from multiple sources:

1. **Default remote** (recommended, downloads and caches automatically per version)
2. **Local file** (via ``--registry-path`` parameter)
3. **Local directory** (via ``--registry-dir`` parameter, looks for ``manifest.parquet``)
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

.. _dataset-versions:

Dataset Versions and Variants
-----------------------------

GeoTessera ships embeddings under two orthogonal axes:

* **dataset version** — the trained Tessera model (``1.0`` or ``1.1``).
  Different versions have *different 128-channel feature spaces*: a feature
  vector from one version is **not comparable** to a vector from another.
* **dataset variant** — for a given version, an independent model run /
  release channel. The default is ``vultr`` (the production hosting on
  Vultr); ``cambridge`` is a test deployment by the Cambridge team for the
  1.1 line.

Currently published combinations on ``s3://tessera-embeddings/``:

+-------------+------------------+--------------+----------------+----------------------------------------------------------------+
| ``version`` | ``S3 path``      | ``variant``  | Years          | Notes                                                          |
+=============+==================+==============+================+================================================================+
| ``1.0``     | ``v1/``          | ``vultr``    | 2017–2025      | Legacy production line. Frozen — no new years will be added.   |
+-------------+------------------+--------------+----------------+----------------------------------------------------------------+
| ``1.1``     | ``v1.1/``        | ``cambridge``| 2017–2025      | Newer model. Cambridge test embeddings; active development.    |
+-------------+------------------+--------------+----------------+----------------------------------------------------------------+

Which one should I use?
~~~~~~~~~~~~~~~~~~~~~~~

**Prefer ``1.1`` / ``cambridge`` for new projects** where it's available
— it reflects the latest model and is where ongoing development happens.
Stick with ``1.0`` / ``vultr`` only if you're (a) reproducing prior
published work that used it, or (b) need a specific tile that the 1.1
deployment doesn't yet have.

.. warning::

   **Do not mix embeddings from different ``(version, variant)`` pairs in
   the same analysis.** Each (version, variant) is a distinct learned
   representation:

   * Cosine similarity, classification heads, clustering, PCA, or any
     downstream model trained on one set produces meaningless results if
     fed vectors from another.
   * Even tiles at the same lat/lon for the same year carry *different
     numeric values* across versions/variants. The grid geometry matches;
     the channel semantics do not.

   GeoTessera enforces a single ``(version, variant)`` per ``GeoTessera``
   instance and records the choice in the ``tessera_metadata.json``
   sidecar that every download writes — re-check that file before
   combining datasets from different runs.

Specifying version + variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CLI** — every data-fetching command (``download``, ``coverage``, ``info``)
accepts both flags::

    geotessera download \
        --dataset-version v1.1 \
        --dataset-variant cambridge \
        --region-file area.geojson \
        --year 2024 \
        --output ./tiles

``--dataset-version`` accepts either form: ``v1`` and ``1.0`` are aliases
(legacy S3 path uses ``v1/``); ``v1.1`` and ``1.1`` are aliases. The
internal normalised form (used in manifests and the metadata sidecar) is
``1.0`` / ``1.1``; the S3 path component is ``v1`` / ``v1.1``.

``--dataset-variant`` defaults to ``vultr`` so unflagged commands keep
working against the legacy line; pass ``cambridge`` (or any other
published variant) explicitly.

**Python API**::

    from geotessera import GeoTessera

    # Default: dataset_version='v1', dataset_variant='vultr' (legacy 1.0)
    gt = GeoTessera()

    # Recommended for new work:
    gt = GeoTessera(dataset_version='v1.1', dataset_variant='cambridge')

    # Either of these is also accepted:
    gt = GeoTessera(dataset_version='1.1', dataset_variant='cambridge')

    # Inspect what's loaded:
    print(gt.dataset_version, gt.dataset_variant)
    print(sorted(gt.registry.get_available_years()))

What gets recorded
~~~~~~~~~~~~~~~~~~

Every NPY download drops a ``tessera_metadata.json`` sidecar in the output
directory with the resolved ``(version, variant)``, the S3 URL prefix the
tiles came from, generation time, and tile count. Every exported GeoTIFF
is stamped with ``TESSERA_DATASET_VERSION``, ``TESSERA_DATASET_VERSION_PATH``,
and ``TESSERA_DATASET_VARIANT`` metadata tags. Use these as the source of
truth for which run produced a given file — local directory names alone
won't tell you (NPY tiles always land under ``global_0.1_degree_representation/``
regardless of variant, by design).

Coverage compositing
~~~~~~~~~~~~~~~~~~~~

For situations where you want to *visualise* multiple versions/variants
together (without combining them analytically), ``geotessera coverage
--by-source`` renders each ``(version, variant)`` group in its own colour
on the same map and produces an interactive ``globe.html`` with per-dataset
layer toggles. See the CLI reference for the full flag set.


Data Organization
-----------------

**Remote Server Structure** (S3, ``us-west-2``)::

    https://s3.us-west-2.amazonaws.com/tessera-embeddings/
    ├── v1/                                          # Dataset version 1.0
    │   ├── manifest.parquet                         # Per-version tile manifest
    │   ├── landmasks.parquet                        # Landmask manifest
    │   ├── global_0.1_degree_representation/        # vultr variant (default)
    │   │   └── 2024/grid_0.15_52.05/grid_0.15_52.05{,_scales}.npy
    │   └── global_0.1_degree_tiff_all/
    │       └── grid_0.15_52.05.tiff                 # Landmask TIFF
    └── v1.1/                                        # Dataset version 1.1
        ├── manifest.parquet
        ├── landmasks.parquet                        # Copy of v1's (same grid)
        └── global_0.1_degree_representation.cambridge/
            └── 2024/grid_0.15_52.05/grid_0.15_52.05{,_scales}.npy

Each ``manifest.parquet`` is scoped to one version and lists every
``(year, lon, lat)`` tile available for that version's variants. The
client downloads only the manifest matching its ``dataset_version`` and
filters by ``dataset_variant`` on load.

**Local Mirror Structure** (when downloading via ``geotessera download``)::

    output_dir/
    ├── tessera_metadata.json                        # version/variant provenance
    ├── global_0.1_degree_representation/            # Always this bare name,
    │   └── 2024/grid_0.15_52.05/                    # regardless of variant.
    │       ├── grid_0.15_52.05.npy
    │       └── grid_0.15_52.05_scales.npy
    └── global_0.1_degree_tiff_all/
        └── grid_0.15_52.05.tiff

**Local Cache Structure** (manifests + landmark manifests, per-version)::

    ~/.cache/geotessera/                             # Default cache location
    ├── v1/
    │   ├── manifest.parquet
    │   ├── manifest.parquet.etag                    # HTTP ETag for conditional GETs
    │   ├── landmasks.parquet
    │   └── landmasks.parquet.etag
    └── v1.1/
        ├── manifest.parquet
        ├── manifest.parquet.etag
        ├── landmasks.parquet
        └── landmasks.parquet.etag

The ``.etag`` sidecars enable conditional ``If-None-Match`` requests: the
client refetches only when the bucket's ETag has actually changed, and S3
returns ``304 Not Modified`` (zero body bytes) otherwise.

Embeddings are organized by:

* **Year**: 2017–2025 for both ``v1/vultr`` and ``v1.1/cambridge``
* **Location**: Global 0.1-degree grid system (same grid across all versions)
* **Format**: NumPy arrays with shape (height, width, 128) after dequantisation

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
