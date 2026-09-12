geotessera package
==================

The GeoTessera package reads, exports, and displays Tessera embeddings.

Package Overview
----------------

.. automodule:: geotessera
   :members:
   :show-inheritance:
   :undoc-members:

API Reference
-------------

.. _geotessera-core:

:mod:`geotessera.core` -- Core Functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GeoTessera`` downloads individual tiles, samples points from local
GeoTIFF or NPY files, and exports rasters on their native UTM grids.
:meth:`~geotessera.GeoTessera.sample_embeddings_at_points` accepts an
explicit error policy. See :ref:`sampling-errors` for examples.

.. automodule:: geotessera.core
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-registry:

:mod:`geotessera.registry` -- Registry Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registry lists available tiles and supplies their download paths,
file sizes, and spatial bounds.

.. automodule:: geotessera.registry
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-visualization:

:mod:`geotessera.visualization` -- Visualization Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualization functions create coverage maps, RGB mosaics, and PCA
images. :func:`~geotessera.visualization.create_pca_mosaic` fits one
sampled PCA model across the inputs. See :doc:`cli_reference` for the
corresponding commands.

.. automodule:: geotessera.visualization
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-store:

:mod:`geotessera.store` -- Zarr Store Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GeoTesseraZarr`` reads points, regions, and patches from Zarr stores.
:meth:`~geotessera.store.GeoTesseraZarr.iter_region` yields row strips, and
:meth:`~geotessera.store.GeoTesseraZarr.export_geotiffs` writes one GeoTIFF
per intersecting UTM zone. See :doc:`zarr_quickstart` for examples.

.. automodule:: geotessera.store
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-tiles:

:mod:`geotessera.tiles` -- Tile Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Tile`` reads GeoTIFF and NPY tiles through the same interface.
``discover_tiles`` finds supported files in a directory.

.. automodule:: geotessera.tiles
   :members:
   :show-inheritance:
   :undoc-members:

.. _geotessera-cli:

:mod:`geotessera.cli` -- Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`cli_reference` for command syntax, options, and output behavior.

.. automodule:: geotessera.cli
   :members:
   :show-inheritance:
   :undoc-members:

Examples
--------

Basic Usage
~~~~~~~~~~~

Initialize and fetch embeddings::

    from geotessera import GeoTessera

    # Initialize client
    gt = GeoTessera()

    # Method 1: Fetch single tile with CRS information
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"CRS: {crs}")  # Native UTM projection

    # Method 2: Fetch region with projection info
    bbox = (-0.2, 51.4, 0.1, 51.6)

    # Step 1: Get list of tiles in region
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Step 2: Fetch the tiles (returns generator for memory efficiency)
    tiles = gt.fetch_embeddings(tiles_to_fetch)

    for year, tile_lon, tile_lat, embedding, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding.shape}, CRS: {crs}")

    # Method 3: Sample at specific points
    points = [(0.15, 52.05), (0.25, 52.15), (-0.05, 51.55)]
    embeddings = gt.sample_embeddings_at_points(points, year=2024)
    print(f"Sampled {len(points)} points: {embeddings.shape}")

Export to GeoTIFF
~~~~~~~~~~~~~~~~~

Export embeddings for GIS use with preserved projections::

    # Step 1: Get list of tiles to export
    bbox = (-0.2, 51.4, 0.1, 51.6)
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)

    # Step 2: Export all bands with native UTM projections
    files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./output",
    )

    # Export specific bands
    rgb_files = gt.export_embedding_geotiffs(
        tiles_to_fetch,
        output_dir="./rgb_output",
        bands=[0, 1, 2]  # Each tile preserves its native UTM projection
    )

    # Export single tile
    single_file = gt.export_embedding_geotiff(
        lon=0.15, lat=52.05,
        output_path="./single_tile.tif",
        year=2024,
        bands=[10, 20, 30]
    )

Create Visualizations
~~~~~~~~~~~~~~~~~~~~~

Generate visualizations::

    from geotessera.visualization import (
        create_rgb_mosaic,
        visualize_global_coverage
    )
    
    # Create RGB mosaic
    create_rgb_mosaic(
        geotiff_paths=rgb_files,
        output_path="mosaic.tif",
        bands=(0, 1, 2)
    )
    
    # Create coverage map
    visualize_global_coverage(
        tessera_client=gt,
        output_path="coverage.png",
        year=2024
    )
