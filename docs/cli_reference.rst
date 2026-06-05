CLI Reference
=============

GeoTessera provides a comprehensive command-line interface for downloading, visualizing, and serving Tessera embeddings.

Global Options
--------------

Data-fetching commands (``download``, ``coverage``, ``info``) share the
dataset-selection options::

    --dataset-version TEXT    Tessera dataset version (default: v1).
                              Accepts v1, 1.0, v1.0, v1.1, 1.1 etc.
    --dataset-variant TEXT    Tessera dataset variant (default: vultr).
                              Known variants: vultr (1.0 default), cambridge (1.1).
    --verbose, -v             Enable verbose output
    --help                    Show help message

The ``download`` and ``coverage`` commands additionally accept manifest
location overrides (``info`` does not)::

    --cache-dir PATH          Custom cache directory for the manifest
    --registry-dir PATH       Directory containing manifest.parquet + landmasks.parquet

To point at a single local manifest file from Python, use the
``GeoTessera(registry_path=...)`` parameter (there is no equivalent CLI flag).

.. note::

   **Dataset versions and variants**: there are now two Tessera versions
   on S3 (``1.0`` and ``1.1``). Prefer ``1.1`` / ``cambridge`` for new work
   — the legacy 1.0 line is frozen. **Never mix versions or variants within
   the same downstream task** — they are independently learned feature
   spaces and not interchangeable. See :ref:`dataset-versions` in the main
   index for the full picture.

Cache Configuration
-------------------

Control where the Parquet manifest is cached:

.. code-block:: bash

    # Use custom cache directory for the per-version manifest
    geotessera download --cache-dir /path/to/cache ...

    # Use a locally-supplied manifest (e.g. a checked-in offline copy)
    geotessera download --registry-path /path/to/manifest.parquet ...

    # Default cache locations (if not specified):
    # - Linux/macOS: ~/.cache/geotessera/{v1,v1.1}/manifest.parquet
    # - Windows:     %LOCALAPPDATA%/geotessera/{v1,v1.1}/manifest.parquet

Note: Embedding tiles land in the user-supplied ``--output`` directory and
persist there for reuse. Only the per-version manifest + landmasks parquet
(~hundreds of MB combined) is kept in the cache directory.

Commands
--------

download
~~~~~~~~

Download embeddings for a region in numpy or GeoTIFF format.

**Usage**::

    geotessera download [OPTIONS]

**Required Options**:

* ``-o, --output PATH`` - Output directory [required]

**Region Definition** (one required):

* ``--bbox TEXT`` - Bounding box: 'min_lon,min_lat,max_lon,max_lat'
* ``--region-file PATH`` - GeoJSON/Shapefile to define region (supports local files or URLs)
* ``--country TEXT`` - Country name (e.g., 'United Kingdom', 'UK', 'GB')

**Format Options**:

* ``-f, --format TEXT`` - Output format: 'tiff' or 'npy' (default: tiff)
* ``--bands TEXT`` - Comma-separated band indices (default: all 128)
* ``--compress TEXT`` - Compression for TIFF format (default: lzw)

**Data Selection**:

* ``--year INT`` - Year of embeddings (default: 2024)
* ``--dataset-version TEXT`` - Tessera dataset version (``v1`` / ``1.0`` /
  ``v1.1`` / ``1.1``; default ``v1``). Pick **once per project**.
* ``--dataset-variant TEXT`` - Tessera dataset variant (default: ``vultr``).
  Pass ``cambridge`` for v1.1 test embeddings.

**Other Options**:

* ``--list-files`` - List all created files with details
* ``-v, --verbose`` - Verbose output

.. warning::

   Re-running ``download`` into a directory that already contains a different
   ``(version, variant)`` will silently mix tiles — the local layout is the
   same regardless of variant. Always use a fresh ``--output`` dir per
   ``(version, variant)`` and check the ``tessera_metadata.json`` sidecar
   if you're unsure which run produced the contents.

**Resume Behaviour**:

Both TIFF and NPY downloads automatically skip files that already exist on
disk, so interrupted downloads can be resumed by re-running the same command.
The progress bar reflects only remaining work.

**Examples**::

    # Download as GeoTIFF (georeferenced, for GIS)
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --year 2024 \
        --output ./london_tiffs
    # Next step: geotessera visualize ./london_tiffs pca_mosaic.tif

    # Download as numpy arrays (for analysis)
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --format npy \
        --year 2024 \
        --output ./london_arrays

    # Download specific bands only
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --bands "0,1,2,10,20,30" \
        --year 2024 \
        --output ./london_subset
    # Next step: geotessera visualize ./london_subset pca_mosaic.tif

    # Download by country name
    geotessera download \
        --country "United Kingdom" \
        --year 2024 \
        --output ./uk_tiles
    # Next step: geotessera visualize ./uk_tiles pca_mosaic.tif

    # Download using a region file
    geotessera download \
        --region-file cambridge.geojson \
        --format tiff \
        --year 2024 \
        --output ./cambridge_tiles
    # Next step: geotessera visualize ./cambridge_tiles pca_mosaic.tif

    # Download v1.1 / cambridge embeddings (recommended for new work)
    geotessera download \
        --dataset-version v1.1 \
        --dataset-variant cambridge \
        --region-file cambridge.geojson \
        --year 2024 \
        --output ./cambridge_v11
    # Note the tessera_metadata.json sidecar dropped alongside the tiles —
    # it records the (version, variant) provenance for this output dir.

**Output Formats**:

**TIFF Format** (``--format tiff``):
    - Creates georeferenced GeoTIFF files with native UTM projections
    - Each tile preserves its native UTM projection from landmask tiles
    - Includes accurate CRS and transform metadata
    - Suitable for GIS software (QGIS, ArcGIS, etc.)
    - Supports compression (lzw, deflate, none)
    - Registry directory structure: ``global_0.1_degree_representation/{year}/grid_{lon}_{lat}/grid_{lon}_{lat}_{year}.tiff``
    - Supports resume: interrupted downloads skip existing files

**NPY Format** (``--format npy``):
    - Downloads quantized numpy arrays (.npy) with separate scale files and landmask TIFFs
    - Registry directory structure: ``global_0.1_degree_representation/{year}/grid_{lon}_{lat}/``
    - Embedding files: ``grid_{lon}_{lat}.npy`` (int8 quantized)
    - Scale files: ``grid_{lon}_{lat}_scales.npy`` (float32)
    - Landmask TIFFs: ``global_0.1_degree_tiff_all/grid_{lon}_{lat}.tiff`` (CRS and transform)
    - Dequantize with: ``embedding = quantized.astype(np.float32) * scales``
    - Supports resume: interrupted downloads skip existing files

visualize
~~~~~~~~~

Create PCA visualization from multiband GeoTIFF files.

**Usage**::

    geotessera visualize INPUT_PATH OUTPUT_FILE [OPTIONS]

**Required Arguments**:

* ``INPUT_PATH`` - Path to GeoTIFF file or directory containing GeoTIFFs
* ``OUTPUT_FILE`` - Output PCA mosaic file (.tif)

**PCA Options**:

* ``--n-components INT`` - Number of PCA components (default: 3). Only first 3 used for RGB visualization - increase for analysis/research.
* ``--crs TEXT`` - Target CRS for reprojection (default: EPSG:3857)

**RGB Balance Options**:

* ``--balance TEXT`` - RGB balance method: histogram (default), percentile, or adaptive
* ``--percentile-low FLOAT`` - Lower percentile for percentile balance method (default: 2.0)
* ``--percentile-high FLOAT`` - Upper percentile for percentile balance method (default: 98.0)

**Examples**::

    # Create PCA visualization (3 components optimal for RGB)
    geotessera visualize tiles/ pca_mosaic.tif

    # Use histogram equalization for maximum contrast
    geotessera visualize tiles/ pca_balanced.tif --balance histogram

    # Use adaptive scaling based on variance
    geotessera visualize tiles/ pca_adaptive.tif --balance adaptive

    # Custom percentile range for outlier-robust scaling
    geotessera visualize tiles/ pca_custom.tif --percentile-low 5 --percentile-high 95

    # Use custom projection
    geotessera visualize tiles/ pca_mosaic.tif --crs EPSG:4326

    # PCA for research - compute more components for analysis
    # (still only uses first 3 for RGB, but saves variance info)
    geotessera visualize tiles/ pca_research.tif --n-components 10

**PCA Visualization Process**:

1. **Data Combination**: Combines all embedding data across tiles
2. **PCA Transformation**: Applies a single PCA transformation to the combined dataset
3. **RGB Mosaic**: Creates a unified RGB mosaic from the first 3 principal components
4. **Consistent Components**: Ensures consistent principal components across the entire region, eliminating tiling artifacts

**Balance Methods**:

* ``histogram`` - Histogram equalization for maximum contrast
* ``percentile`` - Uses percentile range for outlier-robust scaling
* ``adaptive`` - Adaptive scaling based on variance

**Next Steps**: After creating PCA visualization, use ``geotessera webmap`` to create interactive web tiles

webmap
~~~~~~

Create web tiles and viewer from a 3-band RGB mosaic.

**Usage**::

    geotessera webmap RGB_MOSAIC [OPTIONS]

**Required Arguments**:

* ``RGB_MOSAIC`` - 3-band RGB mosaic GeoTIFF file

**Options**:

* ``-o, --output PATH`` - Output directory
* ``--min-zoom INT`` - Min zoom for web tiles (default: 8)
* ``--max-zoom INT`` - Max zoom for web tiles (default: 15)
* ``--initial-zoom INT`` - Initial zoom level (default: 10)
* ``--force/--no-force`` - Force regeneration of tiles even if they exist
* ``--serve/--no-serve`` - Start web server immediately
* ``-p, --port INT`` - Port for web server (default: 8000)
* ``--region-file PATH`` - GeoJSON/Shapefile boundary to overlay (supports local files or URLs)
* ``--use-gdal-raster/--use-gdal2tiles`` - Use newer gdal raster tile vs gdal2tiles (default: gdal2tiles)

**Examples**::

    # Create web tiles from PCA mosaic and serve immediately
    geotessera webmap pca_mosaic.tif --serve

    # Create web tiles with custom zoom levels
    geotessera webmap pca_mosaic.tif --min-zoom 6 --max-zoom 18 --output webmap/

    # Add region boundary overlay
    geotessera webmap pca_mosaic.tif --region-file study_area.geojson --serve

    # Force regeneration of existing tiles
    geotessera webmap pca_mosaic.tif --force --serve

**Process**:
1. Reprojects mosaic to EPSG:3857 for web viewing if needed
2. Generates web tiles at specified zoom levels
3. Creates HTML viewer with Leaflet map
4. Optionally starts web server for immediate viewing


serve
~~~~~

Start a web server to serve visualization files.

**Usage**::

    geotessera serve DIRECTORY [OPTIONS]

**Required Arguments**:

* ``DIRECTORY`` - Directory containing web visualization files

**Options**:

* ``-p, --port INT`` - Port number for web server (default: 8000)
* ``--open/--no-open`` - Auto-open browser (default: open)
* ``--html TEXT`` - Specific HTML file to serve

**Examples**::

    # Serve web visualization and open browser
    geotessera serve ./london_web --open

    # Serve on specific port
    geotessera serve ./london_web --port 8080

    # Serve specific HTML file
    geotessera serve ./visualizations --html coverage.html

    # Serve without auto-opening browser
    geotessera serve ./london_web --no-open

**Notes**:
    - The server automatically finds HTML files (index.html, viewer.html, etc.)
    - Use Ctrl+C to stop the server
    - The server serves all files in the directory
    - Required for viewing Leaflet-based web maps

coverage
~~~~~~~~

Generate a world map showing Tessera embedding coverage.

**Usage**::

    geotessera coverage [OPTIONS]

**Output Options**:

* ``-o, --output PATH`` - Output PNG file path (default: tessera_coverage.png)

**Data Selection**:

* ``--year INT`` - Specific year to visualize (default: all years)
* ``--region-file PATH`` - GeoJSON/Shapefile to focus coverage map on specific region (supports local files or URLs)
* ``--country TEXT`` - Country name to focus coverage map on with precise boundary outline (e.g., 'United Kingdom', 'UK', 'GB')
* ``--dataset-version TEXT`` - Tessera dataset version. Pass ``all`` with
  ``--by-source`` to render every known version.
* ``--dataset-variant TEXT`` - Tessera dataset variant. Pass ``all`` with
  ``--by-source`` to render every variant.

**Multi-source rendering**:

* ``--by-source`` - Render each ``(version, variant)`` in a distinct hue with
  per-dataset checkboxes in the generated ``globe.html``. Each tile's shade
  encodes how many years of coverage it has (pale = 1 year, saturated =
  all years for that dataset). Without ``--dataset-version`` /
  ``--dataset-variant`` overrides, defaults to ``all`` on both axes —
  downloads every known version's manifest and renders them side-by-side.

**Visualization Options**:

* ``--tile-color TEXT`` - Color for tile rectangles (default: red)
* ``--tile-alpha FLOAT`` - Transparency of tiles 0.0-1.0 (default: 0.6)
* ``--tile-size FLOAT`` - Size multiplier for tiles (default: 1.0)
* ``--no-multi-year-colors`` - Disable multi-year color coding

**Map Options**:

* ``--width INT`` - Output image width in pixels (default: 2000)
* ``--no-countries`` - Don't show country boundaries

**Examples**::

    # STEP 1: Check coverage for your region (recommended first step)
    geotessera coverage --region-file study_area.geojson
    geotessera coverage --region-file colombia_aoi.gpkg
    geotessera coverage --country "United Kingdom"
    geotessera coverage --country "Colombia"

    # Check coverage for specific year only
    geotessera coverage --region-file study_area.shp --year 2024
    geotessera coverage --country "UK" --year 2024

    # Global coverage overview (all regions)
    geotessera coverage

    # Global coverage for specific year
    geotessera coverage --year 2024

    # Customize visualization
    geotessera coverage --region-file area.geojson --tile-alpha 0.3
    geotessera coverage --country "Germany" --tile-alpha 0.3

    # Show every dataset version on one map, with a layer-toggle UI in
    # globe.html. Each (version, variant) gets a distinct hue family;
    # within each hue, shade indicates how many years are covered.
    geotessera coverage --by-source --output ./multi
    # Output: multi/{tessera_coverage.png, coverage.json,
    #              coverage_texture_v1_vultr.png, coverage_v1_vultr_*.json,
    #              coverage_texture_v1.1_cambridge.png, coverage_v1.1_cambridge_*.json,
    #              globe.html}

    # Focus on a single (version, variant) but still get the per-dataset
    # legend / by-source coverage.json schema:
    geotessera coverage --by-source --dataset-version v1.1 \
        --dataset-variant cambridge --output ./v11_only

**Multi-Year Color Coding** (default when no specific year requested):
    - **Green**: All available years present for this tile
    - **Blue**: Only the latest year available for this tile  
    - **Orange**: Partial years coverage (some combination of years)

**Output**:
    - High-resolution PNG world map with available tile coverage
    - Colored rectangles show available tile locations (one per 0.1° × 0.1° tile)
    - **Boundary Visualization**: Country/region boundaries are precisely outlined when using ``--country`` or ``--region-file``
    - Global country boundaries are hidden when focusing on specific regions for cleaner visualization
    - Statistics and next-step hints shown after generation

**Next Steps**: After checking coverage, proceed to download data using the same region file or bounding box

info
~~~~

Display information about GeoTIFF files or the library.

**Usage**::

    geotessera info [OPTIONS]

**Options**:

* ``--tiles PATH`` - Analyze tile files/directory (GeoTIFF or NPY format)
* ``--geotiffs PATH`` - Alias for --tiles (deprecated)
* ``--dataset-version TEXT`` - Tessera dataset version (default: ``v1``)
* ``--dataset-variant TEXT`` - Tessera dataset variant (default: ``vultr``)
* ``-v, --verbose`` - Verbose output

**Examples**::

    # Show library information for the legacy 1.0/vultr line (frozen)
    geotessera info

    # Inspect v1.1/cambridge — same 2017–2025 year range, newer model
    geotessera info --dataset-version v1.1 --dataset-variant cambridge

    # Analyze downloaded tiles (GeoTIFF or NPY)
    geotessera info --tiles ./london_tiffs

    # Analyze single GeoTIFF file
    geotessera info --tiles ./london_tiffs/grid_51.45_-0.05.tif

    # Verbose library info
    geotessera info --verbose

**Output for Library Info**:
    - GeoTessera version
    - Available years in dataset
    - Registry information
    - Loaded blocks count

**Output for GeoTIFF Analysis**:
    - Total files analyzed
    - Years covered
    - Coordinate reference systems used
    - Bounding box of all files
    - Band count statistics
    - Individual tile information (with ``--verbose``)

Common Workflows
----------------

Basic Download and View
~~~~~~~~~~~~~~~~~~~~~~~

Complete workflow from coverage check to web visualization::

    # 1. Check data availability (RECOMMENDED FIRST STEP)
    geotessera coverage --year 2024 --output coverage.png

    # 2. Download data
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --year 2024 \
        --output ./london_data

    # 3. Create PCA visualization
    geotessera visualize ./london_data pca_mosaic.tif

    # 4. Create web tiles and serve
    geotessera webmap pca_mosaic.tif --serve

Analysis Workflow
~~~~~~~~~~~~~~~~~

Download for analysis purposes::

    # 1. Check coverage for your analysis region
    geotessera coverage --bbox "-0.1,52.0,0.1,52.2" --year 2024

    # 2. Download as numpy arrays
    geotessera download \
        --bbox "-0.1,52.0,0.1,52.2" \
        --format npy \
        --year 2024 \
        --output ./cambridge_analysis

    # 3. Process in Python
    python your_analysis_script.py

    # 4. Export results as GeoTIFF for visualization
    geotessera download \
        --bbox "-0.1,52.0,0.1,52.2" \
        --format tiff \
        --year 2024 \
        --output ./cambridge_viz

    # 5. Create PCA visualization and web map
    geotessera visualize ./cambridge_viz pca_analysis.tif
    geotessera webmap pca_analysis.tif --serve

GIS Workflow
~~~~~~~~~~~~

Prepare data for GIS software::

    # 1. Check coverage for your region first
    geotessera coverage --region-file study_area.geojson

    # 2. Download with specific bands for analysis
    geotessera download \
        --region-file study_area.geojson \
        --bands "10,20,30,40,50" \
        --format tiff \
        --compress lzw \
        --year 2024 \
        --output ./gis_data

    # 3. Create PCA visualization for overview
    geotessera visualize ./gis_data pca_overview.tif

    # 4. Analyze files before importing to GIS
    geotessera info --tiles ./gis_data --verbose

    # Files are now ready for QGIS, ArcGIS, etc.
    # Use pca_overview.tif for quick visual reference

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**"No tiles found in region"**:
    - Check coverage map first: ``geotessera coverage --year 2024``
    - Verify bounding box format: ``min_lon,min_lat,max_lon,max_lat``
    - Try a different year or larger region

**Slow downloads**:
    - Files are cached after first download
    - Use ``--verbose`` to see download progress
    - Check network connection

**Web visualization not working**:
    - Use ``geotessera serve`` instead of opening HTML directly
    - Check that tiles directory was created
    - Try ``--force`` to regenerate tiles

**Memory issues with large regions**:
    - Download smaller regions at a time
    - Use ``--bands`` to download only needed channels
    - Use ``npy`` format for smaller file sizes

**Permission errors**:
    - Check write permissions for output directory
    - Try using a different output directory
    - Set custom cache directory: ``--cache-dir /tmp/geotessera``

**GeoTIFF projection issues**:
    - Files use native UTM projections (varies by location from landmask tiles)
    - Each tile preserves its original projection for accuracy
    - Most GIS software handles reprojection automatically
    - Use ``geotessera info --tiles`` to check CRS for each tile
    - Different tiles may have different UTM zones

Getting Help
~~~~~~~~~~~~

For additional help::

    # Command-specific help
    geotessera download --help
    geotessera visualize --help

    # Version information
    geotessera version

    # Library information
    geotessera info --verbose

**Resources**:
    - GitHub Issues: https://github.com/ucam-eo/geotessera/issues
    - Documentation: https://geotessera.readthedocs.io/
    - Examples: See tutorials section of documentation
