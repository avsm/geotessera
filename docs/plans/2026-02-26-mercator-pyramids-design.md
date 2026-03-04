# Web Mercator Pyramids via ndpyramid

## Problem

The current approach stores preview pyramids in UTM projection and reprojects
to Web Mercator per-pixel in the browser tile handler. This causes:

- Misalignment between tiles and the basemap
- Complex client code (65K coordinate transforms per 256px tile)
- Bugs in transform handling across pyramid levels
- Poor zoom transition behaviour

## Solution

Generate Web Mercator (EPSG:3857) pyramids server-side using ndpyramid's
`pyramid_reproject`. The tile handler becomes a trivial array slice — no
reprojection, no transform math, perfect basemap alignment by construction.

## Store layout

Existing UTM arrays are unchanged. Old UTM pyramid groups are replaced:

```
utm30_2024.zarr/
    embeddings             # int8   (northing, easting, band)  — UTM, unchanged
    scales                 # f32    (northing, easting)         — UTM, unchanged
    rgb                    # uint8  (northing, easting, rgba)   — UTM, unchanged
    pca_rgb                # uint8  (northing, easting, rgba)   — UTM, unchanged
    rgb_mercator/          # NEW — replaces rgb_pyramid/
        6/                 # zoom 6: coarsest (~1571 m/px at 50°N)
        7/
        ...
        12/                # zoom 12: finest stored (~24.5 m/px at 50°N)
    pca_rgb_mercator/      # NEW — replaces pca_rgb_pyramid/
        6/ .. 12/
```

Each zoom level is a zarr array:
- **Shape**: `(total_y_pixels, total_x_pixels, 4)` — covers the UTM zone's
  bounding box projected into EPSG:3857
- **Chunks**: `(256, 256, 4)` — one chunk = one XYZ tile = one HTTP fetch
- **Dtype**: uint8 (RGBA)

## Zoom level range

Computed from the base pixel size (10 m) and dataset centre latitude:

- **max_zoom** = 12 — 24.5 m/pixel, 2.5× coarser than base. MapLibre
  overzooms for z=13+. Double-click loads full-res embeddings anyway.
- **min_zoom** = 6 — ~1571 m/pixel. Whole UTM zone fits in a few tiles.
- **Total**: 7 levels (z=6 through z=12)

Storage estimate per zone (rgb only): ~4.6 GB for 7 levels. The finest
level (z=12) dominates.

## Store attributes

Remove:
- `has_rgb_pyramid`, `rgb_pyramid_levels`
- `has_pca_rgb_pyramid`, `pca_rgb_pyramid_levels`

Add:
- `has_rgb_mercator: true`
- `has_pca_rgb_mercator: true`
- `mercator_zoom_range: [6, 12]`
- `mercator_tile_bounds: {6: {x_min, x_max, y_min, y_max}, ...}` — per-level
  tile index offsets so the client knows which (x, y) maps to array index (0, 0)

## Python changes (`zarr_zone.py`)

### New function: `build_mercator_pyramid()`

1. Read UTM `rgb` (or `pca_rgb`) array from store
2. Wrap in xarray DataArray with CRS (from store `crs_epsg`) and affine
   transform (from store `transform`)
3. Call `ndpyramid.pyramid_reproject(ds, levels=7, projection='web-mercator',
   pixels_per_tile=256, resampling='bilinear')`
4. Write each level to `rgb_mercator/{z}` with chunks=(256, 256, 4)
5. Compute and store per-level tile bounds in attrs

### New CLI command: `add-mercator-pyramids`

```
geotessera add-mercator-pyramids /path/to/utm30_2024.zarr [--max-zoom 12]
```

- Opens existing store in `r+` mode
- Deletes old `rgb_pyramid/` and `pca_rgb_pyramid/` groups if present
- Generates Web Mercator pyramids for each preview array that exists
- Updates store attributes
- Works on stores with or without old UTM pyramids

### Remove

- `build_preview_pyramid()` — replaced
- `_coarsen_strip()` — no longer needed
- `PYRAMID_LEVELS` constant — replaced by zoom range computation
- `add_pyramids_to_existing_store()` — replaced by `add-mercator-pyramids`

### New dependency

- `ndpyramid[xesmf]` (includes rioxarray, xarray, xesmf for regridding)

## TypeScript changes (`maplibre-zarr-tessera`)

### `zarr-reader.ts`

- Open `rgb_mercator/{z}` arrays instead of `rgb_pyramid/{level}`
- Read `mercator_zoom_range` and `mercator_tile_bounds` from attrs
- Store in `StoreMetadata`

### `zarr-source.ts` — tile handler

The `handleTileRequest` simplifies to:

```typescript
async handleTileRequest(params, abortController) {
  const {z, x, y} = parseUrl(params.url);
  if (z < minZoom || z > maxZoom) return transparent();

  const arr = store.rgbMercatorArrs.get(z);
  const bounds = store.meta.mercatorTileBounds[z];

  // Offset: array (0,0) = tile (bounds.x_min, bounds.y_min)
  const ax = x - bounds.x_min;
  const ay = y - bounds.y_min;
  if (ax < 0 || ay < 0) return transparent();

  const region = await fetchRegion(arr,
    [[ay * 256, (ay + 1) * 256], [ax * 256, (ax + 1) * 256], null]);
  return { data: encodePng(region) };
}
```

No `pickPyramidLevel()`. No UTM projection. No bilinear interpolation.
No `fetchZarrChunks()`. No chunk cache. No concurrency semaphore.

### Remove from `zarr-source.ts`

- `pickPyramidLevel()` — zoom level maps directly to array
- `fetchZarrChunks()` / `fetchSingleChunk()` — one tile = one fetchRegion
- `zarrChunkCache` / `zarrChunkInflight` — MapLibre handles caching
- `acquireFetchSlot()` / `releaseFetchSlot()` — MapLibre limits concurrency
- `pyramidMeta()` for preview levels — not needed
- Per-pixel UTM→Mercator reprojection loop

### Keep unchanged

- Embedding loading (double-click → `loadFullChunk` in UTM)
- Classification overlays
- Grid/UTM boundary overlays
- Spatial query APIs (`getEmbeddingAt`, `getEmbeddingsInKernel`)
- Worker pool (used by embedding rendering)
- Event system

## Testing

1. Generate mercator pyramids for a test zone: `geotessera add-mercator-pyramids test.zarr`
2. Verify store structure: levels 6–12 exist, chunks are 256×256×4
3. Build viewer: `cd tze && pnpm build`
4. Visual: tiles align with basemap at all zoom levels
5. Visual: smooth zoom transitions (MapLibre cross-fades)
6. Visual: preview mode switch (rgb ↔ pca) reloads tiles
7. Double-click: still loads full-res embeddings
8. Classification: still works on embedding tiles
