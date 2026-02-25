# STAC Index Command Design

**Date:** 2026-02-25
**Status:** Approved

## Overview

A new `geotessera-registry stac-index` CLI command that scans a directory of Zarr stores and generates a spec-compliant static STAC catalog. Clients (TZE viewer, QGIS, pystac, stac-browser) can use it to discover available years and UTM zones, then open individual stores.

## CLI Interface

```
geotessera-registry stac-index [--output-dir DIR] <zarr-dir>
```

- `<zarr-dir>` — Directory containing `utm{ZZ}_{YYYY}.zarr` stores
- `--output-dir` — Where to write STAC JSON files (defaults to `<zarr-dir>`)

## Output Structure

```
<output-dir>/
├── catalog.json                    # Root catalog
├── geotessera-2025/
│   ├── collection.json             # One collection per year
│   ├── utm29_2025/
│   │   └── utm29_2025.json         # One item per store
│   └── utm31_2025/
│       └── utm31_2025.json
└── ...
```

The Zarr stores themselves are not modified or moved.

## STAC Hierarchy

### Root Catalog

- **id:** `geotessera`
- **title:** `Geotessera Embedding Stores`
- **description:** Generated from store metadata
- **Links:** One child link per collection (year)

### Collection (one per year)

- **id:** `geotessera-{year}`
- **title:** `Geotessera {year}`
- **Spatial extent:** Union bounding box of all stores in that year (WGS84)
- **Temporal extent:** `{year}-01-01T00:00:00Z` to `{year}-12-31T23:59:59Z`
- **Links:** Parent (catalog), child links to items

### Item (one per Zarr store)

- **id:** `utm{ZZ}_{YYYY}` (matches store name without `.zarr`)
- **Geometry:** WGS84 polygon of the store's spatial extent (computed from transform + shape, reprojected from UTM)
- **Datetime:** `{year}-01-01T00:00:00Z`
- **Properties:**
  - `utm_zone` — UTM zone number
  - `crs_epsg` — EPSG code
  - `pixel_size_m` — Pixel size in meters
  - `grid_width` — Pixel width of the store
  - `grid_height` — Pixel height of the store
  - `n_bands` — Number of embedding bands (128)
  - `has_rgb_preview` — Boolean
  - `has_pca_preview` — Boolean
  - `geotessera_version` — Version string
- **Assets:**
  - `zarr` — Relative href to the `.zarr` directory, media type `application/x-zarr-v3`, role `data`

## Spatial Extent Computation

For each store, compute the WGS84 bounding box from Zarr metadata:

1. Read `transform` (affine) and `shape` from store attrs
2. Compute four UTM corners: origin, origin + width*px, origin - height*px, etc.
3. Reproject corners from EPSG to WGS84 using pyproj
4. Take the bounding box of the reprojected corners

This reuses the same projection logic already in `zarr_zone.py`.

## Dependencies

- **pystac** — Catalog/Collection/Item construction, link management, serialization
- **pyproj** — UTM to WGS84 reprojection (already a dependency)
- **zarr** — Reading store metadata (already a dependency)

`pystac` is added as a new dependency in `pyproject.toml`.

## Implementation Location

- New function `stac_index_command(args)` in `registry_cli.py`
- Helper `_zarr_store_to_stac_item()` for per-store metadata extraction
- Helper `_store_bbox_wgs84()` for spatial extent computation
- New subparser added alongside existing commands

## Error Handling

- Stores that can't be opened or lack required metadata are skipped with a warning
- If no valid stores are found, exit with an error message
- Overwrite existing STAC JSON files (idempotent regeneration)
