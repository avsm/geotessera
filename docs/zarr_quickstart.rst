Zarr Quick Start
================

``GeoTesseraZarr`` reads embeddings from a local or remote Zarr store.
Queries select the required UTM zones and return dequantized float32 values.
GeoTessera requires Python 3.12 or later::

    pip install geotessera

Read one embedding
------------------

Create a client and inspect the available years::

    from geotessera import GeoTesseraZarr

    gt = GeoTesseraZarr()
    print(gt.years)

Read an embedding at a WGS84 longitude and latitude::

    vec, status = gt.probe(0.12, 52.20, year=2024)
    print(status)
    print(vec.shape)

``probe`` returns ``valid`` for an available embedding, ``water`` for open
water, ``nodata`` for an unwritten pixel, and ``outside`` beyond coverage.
``sample_at`` returns only the vector, with NaN values for missing embeddings.

Sample points
-------------

``sample_points`` reads a sequence of WGS84 longitude and latitude pairs
and returns an array with one row per input point::

    coords = [(0.12, 52.20), (-2.97, 53.44)]
    embeddings = gt.sample_points(coords, year=2024)

Points without an embedding return NaN rows. Points at a UTM zone boundary
can use the neighbouring zone when their own zone lacks coverage.

Read a region
-------------

``read_region`` takes WGS84 bounds in west, south, east, north order.
It returns an array, affine transform, and CRS on one native UTM grid::

    bbox = (0.05, 52.15, 0.20, 52.25)
    mosaic, transform, crs = gt.read_region(bbox, year=2024)

The bounding box selects a pixel window without resampling. Use
``export_geotiffs`` to export all intersecting zones when a region spans
more than one UTM zone.

Stream a large region
---------------------

``iter_region`` yields the same pixels as row strips, so the full region
does not need to fit in memory::

    for block, transform, crs in gt.iter_region(bbox, year=2024, strip_rows=512):
        print(block.shape)

``read_region_quantized`` returns int8 embeddings and their scale arrays.
Use it when the quantized representation is preferable to a float32 array.

Export GeoTIFFs
---------------

``export_geotiffs`` writes one file per intersecting UTM zone and returns
the output paths::

    files = gt.export_geotiffs(bbox, 2024, "region/", bands=[0, 1, 2])

Files are named ``tessera_YEAR_utmNN.tif`` and contain float32 embeddings,
NaN nodata, the native grid, band descriptions, and source metadata.
Output windows enclose the bounds within the available zone grids.
Each completed file replaces its destination. Rerunning repeats the export.

Omit ``bands`` to export all bands. Use ``depth`` to select a published
embedding prefix; band indices are zero-based within that prefix.
``compress`` selects GeoTIFF compression and defaults to ``"lzw"``.
``strip_rows`` limits the rows processed at a time and defaults to 128.

Set ``dry_run=True`` to return a list of estimates containing ``zone``,
``width``, ``height``, and ``uncompressed_bytes`` without reading embeddings::

    estimates = gt.export_geotiffs(bbox, 2024, "region/", dry_run=True)
    print(sum(item["uncompressed_bytes"] for item in estimates))

These estimates describe uncompressed output, not network transfer or
compressed file size. Split bounds that cross the antimeridian into two
requests.

The CLI provides the same export and can also create a web map directly::

    geotessera download --bbox '0.05,52.15,0.20,52.25' --year 2024 --output region/
    geotessera webmap --bbox '0.05,52.15,0.20,52.25' --year 2024 --output map/ --serve

See :doc:`cli_reference` for band selection, caching, and restart behavior.

Read a patch
------------

``read_patch`` returns a fixed-size square centred on a point::

    patch, transform, crs = gt.read_patch(0.12, 52.20, year=2024, size_px=256)

The point falls in pixel ``[size_px // 2, size_px // 2]``. A patch within one
UTM zone uses the native grid. A patch across a zone boundary uses a local
transverse Mercator grid and nearest-neighbour resampling by default.
The returned CRS describes the output grid.

A patch extending beyond the stored grid retains its requested position,
with NaN for uncovered pixels. Set ``dst_crs`` to use a particular projected
CRS; its units must be metres.

Matryoshka depths
-----------------

Stores for v2 can contain prefix arrays alongside the full embeddings.
Select a published prefix with ``depth``::

    from geotessera.registry import zarr_store_url

    gt2 = GeoTesseraZarr(zarr_store_url("v2"))
    embeddings16 = gt2.sample_points(coords, year=2024, depth=16)

The prefix equals the first dimensions of the full embedding. A store
without the requested depth reports an error and lists available depths.
Selecting fewer dimensions reduces the output size; transferred bytes
depend on the physical chunk layout.

Select a store
--------------

Use ``zarr_store_url`` to select another dataset version, or pass a local
store path::

    gt = GeoTesseraZarr(zarr_store_url("v1.1"))
    local = GeoTesseraZarr("/data/tessera.zarr")

Use one version and variant per analysis because their embedding spaces
are independently learned. See :ref:`dataset-versions`.

Caching
-------

Set ``cache_dir`` to persist metadata between runs. Byte-range reads of
sharded embeddings are cached within the process. Each store uses a
separate cache subdirectory::

    gt = GeoTesseraZarr(cache_dir="tessera-cache")

Set ``cache_max_size`` to bound the cache in bytes::

    gt = GeoTesseraZarr(
        cache_dir="tessera-cache", cache_max_size=2 * 1024**3
    )

Keep exported GeoTIFFs or completed web map directories for reuse between
runs. The read cache does not provide a persistent copy of the embeddings.

See also
--------

:doc:`quickstart` covers individual tile downloads.
:doc:`architecture` describes the Zarr layout and quantization.
The `examples repository <https://github.com/ucam-eo/geotessera-examples>`_
contains analysis workflows using this API.
