# Web Mercator Pyramids Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace UTM-native preview pyramids with ndpyramid-generated Web Mercator pyramids, and simplify the client tile handler to a trivial array slice.

**Architecture:** Server-side `ndpyramid.pyramid_reproject()` generates EPSG:3857 pyramids from the existing UTM rgb/pca_rgb arrays. Each zoom level is stored with 256x256 chunks so one tile = one zarr chunk fetch. The client tile handler just parses z/x/y, offsets into the array, and returns the chunk — no reprojection.

**Tech Stack:** Python (ndpyramid, rioxarray, xarray, zarr), TypeScript (zarrita, maplibre-gl)

---

## Context

**Repositories:**
- `~/src/git/ucam-eo/geotessera/` — Python zarr generation (`geotessera/zarr_zone.py`, `geotessera/registry_cli.py`)
- `~/src/git/ucam-eo/tze/packages/maplibre-zarr-tessera/src/` — TypeScript tile viewer (`zarr-source.ts`, `zarr-reader.ts`, `types.ts`)

**Current state:** The code on the `zarr2` branch has a broken client-side UTM→Mercator reprojection in `zarr-source.ts`. Pyramids are generated in UTM space by `build_preview_pyramid()` in `zarr_zone.py` using 2x mean coarsening. The client opens them at `rgb_pyramid/{level}`.

**Design doc:** `docs/plans/2026-02-26-mercator-pyramids-design.md`

**Key numbers:**
- Base pixel size: 10m
- Useful zoom range: z=6 (coarsest) to z=12 (finest stored), 7 levels
- Zarr chunks: 256×256×4 (uint8 RGBA) = 256 KB per tile
- Storage: ~4.6 GB per UTM zone for rgb pyramids

---

## Task 1: Add ndpyramid dependency

**Files:**
- Modify: `geotessera/pyproject.toml`

**Step 1: Add ndpyramid to dependencies**

In `pyproject.toml`, add `"ndpyramid"` to the `dependencies` list. ndpyramid
pulls in xesmf for regridding; xarray and rioxarray are already dependencies.

```toml
dependencies = [
    ...existing deps...
    "ndpyramid",
]
```

**Step 2: Verify installation**

Run:
```bash
cd ~/src/git/ucam-eo/geotessera
uv sync
uv run python -c "import ndpyramid; print('ndpyramid OK')"
```
Expected: `ndpyramid OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add ndpyramid for Web Mercator pyramid generation"
```

---

## Task 2: Implement `build_mercator_pyramid()` in zarr_zone.py

**Files:**
- Modify: `geotessera/zarr_zone.py:1303-1502` (replace pyramid section)

This is the core Python change. Replace `build_preview_pyramid()`, `_coarsen_strip()`, and `PYRAMID_LEVELS` with a new `build_mercator_pyramid()` function.

**Step 1: Write the new function**

Replace lines 1303–1502 of `zarr_zone.py` (the entire "Preview pyramids" section) with:

```python
# =============================================================================
# Web Mercator preview pyramids (via ndpyramid)
# =============================================================================


def _compute_mercator_zoom_range(
    pixel_size_m: float,
    center_lat: float,
    max_zoom: int = 12,
) -> Tuple[int, int]:
    """Compute the useful Web Mercator zoom range for a dataset.

    Args:
        pixel_size_m: Base pixel size in metres (e.g. 10.0).
        center_lat: Centre latitude in degrees for cos(lat) correction.
        max_zoom: Finest zoom level to generate (default 12 ≈ 24.5 m/px).

    Returns:
        (min_zoom, max_zoom) inclusive.
    """
    cos_lat = math.cos(math.radians(center_lat))
    # Finest zoom where one tile pixel ≈ pixel_size_m
    finest = math.ceil(math.log2(40_075_016.686 * cos_lat / (256 * pixel_size_m)))
    # Cap at max_zoom; MapLibre will overzoom beyond this
    finest = min(finest, max_zoom)
    # Coarsest: 7 levels below finest, but no less than 0
    coarsest = max(0, finest - 6)
    return coarsest, finest


def _utm_array_to_xarray(
    store: "zarr.Group",
    array_name: str,
) -> "xarray.DataArray":
    """Wrap a UTM zarr preview array as a georeferenced xarray DataArray.

    Reads the array data and store attributes (transform, CRS) to create
    an xarray DataArray with proper spatial coordinates and CRS metadata
    that ndpyramid/rioxarray can reproject.

    Args:
        store: Zarr group opened in read mode.
        array_name: Name of the array (e.g. "rgb" or "pca_rgb").

    Returns:
        xarray DataArray with dims (y, x, band), CRS set, and spatial coords.
    """
    import xarray as xr
    import rioxarray  # noqa: F401 — needed for .rio accessor
    from rasterio.transform import Affine

    arr = store[array_name]
    attrs = dict(store.attrs)
    transform = attrs["transform"]
    epsg = attrs["crs_epsg"]

    h, w, c = arr.shape
    pixel_size = transform[0]
    origin_e = transform[2]
    origin_n = transform[5]

    # Pixel-centre coordinates
    x_coords = origin_e + (np.arange(w) + 0.5) * pixel_size
    y_coords = origin_n - (np.arange(h) + 0.5) * pixel_size

    data = np.asarray(arr[:])
    da = xr.DataArray(
        data,
        dims=["y", "x", "band"],
        coords={
            "y": y_coords,
            "x": x_coords,
            "band": np.arange(c),
        },
    )
    da = da.rio.write_crs(f"EPSG:{epsg}")
    da = da.rio.write_transform(Affine(*transform))
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def build_mercator_pyramid(
    store: "zarr.Group",
    preview_name: str,
    max_zoom: int = 12,
    console: Optional["rich.console.Console"] = None,
) -> dict:
    """Build Web Mercator pyramid from a UTM preview array using ndpyramid.

    Reads ``store[preview_name]`` (full-res RGBA uint8 in UTM), reprojects
    to EPSG:3857 at multiple zoom levels, and writes each level to
    ``{preview_name}_mercator/{zoom}`` with 256×256 chunks.

    Args:
        store: Zarr group opened in ``r+`` mode.
        preview_name: Source array name (``"rgb"`` or ``"pca_rgb"``).
        max_zoom: Finest zoom level to generate (default 12).
        console: Optional Rich Console for progress display.

    Returns:
        Dict with keys: zoom_range, tile_bounds, levels_written.
        Empty dict if source array is missing.
    """
    from ndpyramid import pyramid_reproject

    # --- Validate source ---
    try:
        _ = store[preview_name]
    except KeyError:
        logger.warning("build_mercator_pyramid: %r not found in store", preview_name)
        return {}

    # --- Compute zoom range from dataset metadata ---
    attrs = dict(store.attrs)
    pixel_size = attrs["transform"][0]
    epsg = attrs["crs_epsg"]

    # Estimate centre latitude of dataset
    transform = attrs["transform"]
    origin_e = transform[2]
    origin_n = transform[5]
    h, w = store[preview_name].shape[:2]
    center_e = origin_e + w * pixel_size / 2
    center_n = origin_n - h * pixel_size / 2

    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    center_lon, center_lat = to_wgs84.transform(center_e, center_n)

    min_zoom, max_zoom_actual = _compute_mercator_zoom_range(
        pixel_size, center_lat, max_zoom,
    )
    num_levels = max_zoom_actual - min_zoom + 1

    if console is not None:
        console.print(
            f"  Mercator pyramid: z={min_zoom}–{max_zoom_actual} "
            f"({num_levels} levels), centre={center_lat:.1f}°N"
        )

    # --- Delete existing mercator pyramid group ---
    mercator_name = f"{preview_name}_mercator"
    if mercator_name in store:
        del store[mercator_name]
    mercator_group = store.create_group(mercator_name)

    # --- Wrap UTM array as xarray DataArray ---
    if console is not None:
        console.print(f"  Reading {preview_name} array into xarray...")
    da = _utm_array_to_xarray(store, preview_name)

    # Convert to Dataset (ndpyramid expects Dataset)
    ds = da.to_dataset(name=preview_name)

    # --- Generate pyramid via ndpyramid ---
    if console is not None:
        console.print(f"  Running ndpyramid reproject (levels {min_zoom}–{max_zoom_actual})...")

    dt = pyramid_reproject(
        ds,
        levels=num_levels,
        pixels_per_tile=256,
    )

    # --- Write each level to zarr store ---
    tile_bounds = {}
    levels_written = 0

    for i in range(num_levels):
        zoom = min_zoom + i
        level_ds = dt[str(i)].ds

        level_da = level_ds[preview_name]
        data = level_da.values  # numpy array

        if data.ndim == 2:
            # Single-band — expand to (y, x, 1)
            data = data[:, :, np.newaxis]
        elif data.ndim == 3 and data.shape[0] <= 4:
            # ndpyramid may put band first: (band, y, x) → (y, x, band)
            data = np.moveaxis(data, 0, -1)

        oh, ow = data.shape[:2]
        nc = data.shape[2] if data.ndim == 3 else 1

        # Ensure uint8
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)

        chunk_h = min(256, oh)
        chunk_w = min(256, ow)

        out_arr = mercator_group.create_array(
            str(zoom),
            shape=(oh, ow, nc),
            chunks=(chunk_h, chunk_w, nc),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=None,
        )
        out_arr[:] = data

        # Compute tile bounds: which XYZ tile indices this array covers
        # ndpyramid level i at pixels_per_tile=256 covers specific tiles
        n_tiles_y = oh // 256
        n_tiles_x = ow // 256

        # For now store array dimensions; the client computes offsets
        # from the spatial coordinates stored by ndpyramid
        out_arr.attrs.update({
            "zoom": zoom,
            "shape": [oh, ow, nc],
            "n_tiles_y": n_tiles_y,
            "n_tiles_x": n_tiles_x,
        })

        tile_bounds[zoom] = {
            "n_tiles_y": n_tiles_y,
            "n_tiles_x": n_tiles_x,
        }
        levels_written += 1

        if console is not None:
            console.print(
                f"    z={zoom}: {oh}x{ow} ({n_tiles_y}x{n_tiles_x} tiles)"
            )

    # --- Store summary attrs ---
    mercator_group.attrs.update({
        "source": preview_name,
        "zoom_min": min_zoom,
        "zoom_max": max_zoom_actual,
        "num_levels": levels_written,
        "center_lon": float(center_lon),
        "center_lat": float(center_lat),
        "projection": "EPSG:3857",
    })

    return {
        "zoom_range": [min_zoom, max_zoom_actual],
        "tile_bounds": tile_bounds,
        "levels_written": levels_written,
    }
```

**Step 2: Test manually with an existing store**

```bash
cd ~/src/git/ucam-eo/geotessera
uv run python -c "
import zarr
from geotessera.zarr_zone import build_mercator_pyramid
store = zarr.open_group('<path-to-test-store>', mode='r+')
result = build_mercator_pyramid(store, 'rgb', max_zoom=12)
print(result)
"
```

Expected: prints zoom_range, tile_bounds dict, levels_written > 0.

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add build_mercator_pyramid using ndpyramid reproject"
```

---

## Task 3: Implement `add_mercator_pyramids_to_existing_store()`

**Files:**
- Modify: `geotessera/zarr_zone.py:1506-1568` (replace `add_pyramids_to_existing_store`)

**Step 1: Replace the function**

Replace `add_pyramids_to_existing_store()` with:

```python
def add_mercator_pyramids_to_existing_store(
    store_path: Path,
    max_zoom: int = 12,
    workers: Optional[int] = None,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Add Web Mercator pyramids for all existing preview arrays.

    Opens the store at *store_path* in ``r+`` mode.  For each preview
    array that exists (``rgb``, ``pca_rgb``), generates Web Mercator
    pyramids and updates store attributes.

    Deletes old UTM pyramids (``rgb_pyramid/``, ``pca_rgb_pyramid/``)
    if present.

    Args:
        store_path: Path to the ``.zarr`` directory.
        max_zoom: Finest zoom level to generate (default 12).
        workers: Unused (kept for API compat). ndpyramid handles parallelism.
        console: Optional Rich Console for progress display.
    """
    import zarr

    store = zarr.open_group(str(store_path), mode="r+")
    attrs = dict(store.attrs)

    # Delete old UTM pyramids
    for old_group in ["rgb_pyramid", "pca_rgb_pyramid"]:
        if old_group in store:
            del store[old_group]
            if console is not None:
                console.print(f"  Deleted old {old_group}/")

    # Remove old pyramid attrs
    old_attrs = [
        "has_rgb_pyramid", "rgb_pyramid_levels",
        "has_pca_rgb_pyramid", "pca_rgb_pyramid_levels",
    ]
    for attr_name in old_attrs:
        if attr_name in attrs:
            del store.attrs[attr_name]

    # Build mercator pyramids for each preview
    previews = [
        ("rgb", "has_rgb_preview"),
        ("pca_rgb", "has_pca_preview"),
    ]

    for preview_name, attr_flag in previews:
        if not attrs.get(attr_flag, False):
            continue

        if console is not None:
            console.print(f"  Building {preview_name} mercator pyramid...")

        result = build_mercator_pyramid(
            store, preview_name, max_zoom=max_zoom, console=console,
        )

        if result and result.get("levels_written", 0) > 0:
            store.attrs.update({
                f"has_{preview_name}_mercator": True,
                f"{preview_name}_mercator_zoom_range": result["zoom_range"],
            })

            if console is not None:
                zmin, zmax = result["zoom_range"]
                console.print(
                    f"  [green]{preview_name} mercator: "
                    f"z={zmin}–{zmax} ({result['levels_written']} levels)[/green]"
                )
```

**Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add add_mercator_pyramids_to_existing_store"
```

---

## Task 4: Update CLI to use mercator pyramids

**Files:**
- Modify: `geotessera/registry_cli.py:2528` (import line)
- Modify: `geotessera/registry_cli.py:2636-2682` (pyramid-only handler)
- Modify: `geotessera/registry_cli.py:2679` (function call)

**Step 1: Update the import**

At line 2528, change:
```python
from .zarr_zone import build_zone_stores, add_rgb_to_existing_store, add_pca_to_existing_store, add_pyramids_to_existing_store
```
to:
```python
from .zarr_zone import build_zone_stores, add_rgb_to_existing_store, add_pca_to_existing_store, add_mercator_pyramids_to_existing_store
```

**Step 2: Update the function call**

At line 2679, change:
```python
add_pyramids_to_existing_store(store_path, workers=args.workers, console=console)
```
to:
```python
add_mercator_pyramids_to_existing_store(store_path, console=console)
```

**Step 3: Update the `--pyramid` flag in `build_zone_stores` call**

Find where `build_zone_stores` is called with `pyramid=args.pyramid` (around line 2700-2720). The `pyramid=True` parameter currently triggers UTM pyramid building during a full store build. Update the call site in `build_zone_stores` to call `build_mercator_pyramid` instead of `build_preview_pyramid`. This is at `zarr_zone.py:776-788`:

Change:
```python
if pyramid:
    for preview_name in ["rgb", "pca_rgb"]:
        if preview_name in store:
            levels = build_preview_pyramid(
                store, preview_name, workers=workers, console=console,
            )
            if levels > 0:
                store.attrs.update({
                    f"has_{preview_name}_pyramid": True,
                    f"{preview_name}_pyramid_levels": levels + 1,
                })
                if console is not None:
                    console.print(f"  [green]{preview_name} pyramid: {levels} levels[/green]")
```

to:
```python
if pyramid:
    for preview_name in ["rgb", "pca_rgb"]:
        if preview_name in store:
            result = build_mercator_pyramid(
                store, preview_name, console=console,
            )
            if result and result.get("levels_written", 0) > 0:
                store.attrs.update({
                    f"has_{preview_name}_mercator": True,
                    f"{preview_name}_mercator_zoom_range": result["zoom_range"],
                })
                if console is not None:
                    zmin, zmax = result["zoom_range"]
                    console.print(f"  [green]{preview_name} mercator: z={zmin}-{zmax}[/green]")
```

**Step 4: Commit**

```bash
git add geotessera/zarr_zone.py geotessera/registry_cli.py
git commit -m "feat: CLI uses mercator pyramids for --pyramid and --pyramid-only"
```

---

## Task 5: Remove old UTM pyramid code

**Files:**
- Modify: `geotessera/zarr_zone.py`

**Step 1: Delete old functions**

Remove these functions entirely from `zarr_zone.py`:
- `_coarsen_strip()` (was around line 1313)
- `build_preview_pyramid()` (was around line 1326)

They were replaced by `build_mercator_pyramid()` in Task 2. Also remove the old `PYRAMID_LEVELS` constant.

Verify nothing else imports these:

```bash
cd ~/src/git/ucam-eo/geotessera
grep -r "build_preview_pyramid\|_coarsen_strip\|PYRAMID_LEVELS" geotessera/ --include="*.py"
```

Expected: no matches (or only the lines you're about to delete).

**Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "refactor: remove old UTM pyramid code (replaced by mercator)"
```

---

## Task 6: Update TypeScript types and reader

**Files:**
- Modify: `~/src/git/ucam-eo/tze/packages/maplibre-zarr-tessera/src/types.ts`
- Modify: `~/src/git/ucam-eo/tze/packages/maplibre-zarr-tessera/src/zarr-reader.ts`

**Step 1: Update StoreMetadata in types.ts**

Replace the pyramid-related fields:

```typescript
export interface StoreMetadata {
  url: string;
  utmZone: number;
  epsg: number;
  transform: [number, number, number, number, number, number];
  shape: [number, number, number];
  chunkShape: [number, number, number];
  nBands: number;
  hasRgb: boolean;
  hasPca: boolean;
  pcaExplainedVariance?: number[];
  // Mercator pyramids (replaces hasRgbPyramid/hasPcaPyramid)
  hasRgbMercator: boolean;
  hasPcaMercator: boolean;
  mercatorZoomRange: [number, number] | null;  // [minZoom, maxZoom]
  pyramidBasePixelSize: number;
}
```

**Step 2: Update zarr-reader.ts to open mercator arrays**

Replace the pyramid detection section (lines 76-113) with mercator detection:

```typescript
// Detect Web Mercator pyramid arrays
const rgbMercatorArrs = new Map<number, zarr.Array<zarr.DataType>>();
const pcaMercatorArrs = new Map<number, zarr.Array<zarr.DataType>>();
let hasRgbMercator = !!(attrs.has_rgb_mercator);
let hasPcaMercator = !!(attrs.has_pca_rgb_mercator);
const rgbMercZoomRange = (attrs.rgb_mercator_zoom_range as [number, number]) || null;
const pcaMercZoomRange = (attrs.pca_rgb_mercator_zoom_range as [number, number]) || null;
const mercatorZoomRange = rgbMercZoomRange ?? pcaMercZoomRange;

if (hasRgbMercator && rgbMercZoomRange) {
  for (let z = rgbMercZoomRange[0]; z <= rgbMercZoomRange[1]; z++) {
    try {
      const arr = await zarr.open(rootLoc.resolve(`rgb_mercator/${z}`), { kind: 'array' });
      rgbMercatorArrs.set(z, arr);
    } catch {
      break;
    }
  }
  if (rgbMercatorArrs.size === 0) hasRgbMercator = false;
}

if (hasPcaMercator && pcaMercZoomRange) {
  for (let z = pcaMercZoomRange[0]; z <= pcaMercZoomRange[1]; z++) {
    try {
      const arr = await zarr.open(rootLoc.resolve(`pca_rgb_mercator/${z}`), { kind: 'array' });
      pcaMercatorArrs.set(z, arr);
    } catch {
      break;
    }
  }
  if (pcaMercatorArrs.size === 0) hasPcaMercator = false;
}
```

Update the `meta` object:
```typescript
const meta: StoreMetadata = {
  url,
  utmZone,
  epsg,
  transform,
  shape: embArr.shape as [number, number, number],
  chunkShape: embArr.chunks as [number, number, number],
  nBands: (embArr.shape[2] as number) || 128,
  hasRgb,
  hasPca,
  pcaExplainedVariance: attrs.pca_explained_variance as number[] | undefined,
  hasRgbMercator,
  hasPcaMercator,
  mercatorZoomRange,
  pyramidBasePixelSize: Math.abs(transform[0]),
};
```

Update the `ZarrStore` interface and return:
```typescript
export interface ZarrStore {
  meta: StoreMetadata;
  embArr: zarr.Array<zarr.DataType>;
  scalesArr: zarr.Array<zarr.DataType>;
  rgbArr: zarr.Array<zarr.DataType> | null;
  pcaArr: zarr.Array<zarr.DataType> | null;
  rgbMercatorArrs: Map<number, zarr.Array<zarr.DataType>>;
  pcaMercatorArrs: Map<number, zarr.Array<zarr.DataType>>;
  chunkManifest: Set<string> | null;
}

// ... return statement:
return { meta, embArr, scalesArr, rgbArr, pcaArr, rgbMercatorArrs, pcaMercatorArrs, chunkManifest };
```

**Step 3: Commit**

```bash
cd ~/src/git/ucam-eo/tze
git add packages/maplibre-zarr-tessera/src/types.ts packages/maplibre-zarr-tessera/src/zarr-reader.ts
git commit -m "feat: update types and reader for mercator pyramid arrays"
```

---

## Task 7: Simplify the tile handler in zarr-source.ts

This is the big client-side payoff. Replace the broken UTM reprojection handler with a trivial mercator slice.

**Files:**
- Modify: `~/src/git/ucam-eo/tze/packages/maplibre-zarr-tessera/src/zarr-source.ts`

**Step 1: Replace handleTileRequest**

Replace the entire `handleTileRequest` method (currently ~120 lines of UTM reprojection) with:

```typescript
private async handleTileRequest(
  params: { url: string },
  abortController: AbortController,
): Promise<{ data: ArrayBuffer }> {
  const transparent = () => this.encodeRgbaToPng(new Uint8Array(256 * 256 * 4), 256, 256);

  if (!this.store || !this.proj) throw new Error('Store not initialized');

  try {
    // Parse z/x/y from URL: "zarr-xxx://{z}/{x}/{y}?t=..."
    const urlPath = params.url.split('://')[1]?.split('?')[0];
    if (!urlPath) return { data: await transparent() };
    const parts = urlPath.split('/');
    const z = parseInt(parts[0], 10);
    const x = parseInt(parts[1], 10);
    const y = parseInt(parts[2], 10);

    // Pick the mercator array for this zoom level
    const mode = this.opts.preview;
    const mercArrs = mode === 'pca'
      ? this.store.pcaMercatorArrs
      : this.store.rgbMercatorArrs;

    // Find the best available zoom level (exact or nearest coarser)
    let arr: typeof this.store.rgbArr = null;
    let useZ = z;
    while (useZ >= (this.store.meta.mercatorZoomRange?.[0] ?? 0)) {
      const candidate = mercArrs.get(useZ);
      if (candidate) { arr = candidate; break; }
      useZ--;
    }
    if (!arr) return { data: await transparent() };

    // Compute tile pixel offset within the array.
    // ndpyramid's pyramid_reproject at level i covers the full world grid
    // at that zoom level, but the stored array only covers the data extent.
    // The array shape tells us how many pixels are stored.
    const arrShape = arr.shape as number[];
    const arrH = arrShape[0];
    const arrW = arrShape[1];

    // At zoom useZ, tiles are 256px. Compute how many tiles the array spans.
    const nTilesY = Math.ceil(arrH / 256);
    const nTilesX = Math.ceil(arrW / 256);

    // ndpyramid generates a global grid — we need to figure out
    // which tile (x, y) at zoom useZ corresponds to array index (0, 0).
    // The array attrs store this, or we compute from the stored coordinates.
    const arrAttrs = arr.attrs as Record<string, unknown>;
    const tileOffsetX = (arrAttrs.tile_offset_x as number) ?? 0;
    const tileOffsetY = (arrAttrs.tile_offset_y as number) ?? 0;

    // If zoom level was downgraded, scale tile coordinates
    const zoomDiff = z - useZ;
    const scaledX = Math.floor(x / Math.pow(2, zoomDiff));
    const scaledY = Math.floor(y / Math.pow(2, zoomDiff));

    // Array-local tile indices
    const ax = scaledX - tileOffsetX;
    const ay = scaledY - tileOffsetY;

    if (ax < 0 || ay < 0 || ax >= nTilesX || ay >= nTilesY) {
      return { data: await transparent() };
    }

    // Fetch the 256x256 chunk
    const r0 = ay * 256;
    const r1 = Math.min(r0 + 256, arrH);
    const c0 = ax * 256;
    const c1 = Math.min(c0 + 256, arrW);

    if (abortController.signal.aborted) throw new DOMException('Aborted', 'AbortError');

    const view = await fetchRegion(arr, [[r0, r1], [c0, c1], null]);

    if (abortController.signal.aborted) throw new DOMException('Aborted', 'AbortError');

    // Build RGBA tile
    const tileW = 256, tileH = 256;
    const rgba = new Uint8Array(tileW * tileH * 4);
    const src = new Uint8Array(view.data.buffer, view.data.byteOffset, view.data.byteLength);
    const srcH = r1 - r0;
    const srcW = c1 - c0;
    const nCh = view.shape.length >= 3 ? view.shape[2] : 3;

    for (let row = 0; row < srcH; row++) {
      for (let col = 0; col < srcW; col++) {
        const si = (row * srcW + col) * nCh;
        const di = (row * tileW + col) * 4;
        rgba[di]     = src[si];
        rgba[di + 1] = src[si + 1] ?? 0;
        rgba[di + 2] = src[si + 2] ?? 0;
        rgba[di + 3] = (nCh >= 4) ? src[si + 3] : 255;
      }
    }

    return { data: await this.encodeRgbaToPng(rgba, tileW, tileH) };
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err;
    this.debug('error', `Tile render failed: ${(err as Error).message}`);
    return { data: await transparent() };
  }
}
```

**Step 2: Remove dead code**

Delete these methods/fields from `zarr-source.ts` as they are no longer used by the tile handler:

- `pickPyramidLevel()` method
- `fetchSingleChunk()` method
- `fetchZarrChunks()` method
- `acquireFetchSlot()` / `releaseFetchSlot()` methods
- `zarrChunkCache` field and `MAX_ZARR_CACHE` constant
- `zarrChunkInflight` field
- `fetchQueue` / `fetchActive` / `MAX_CONCURRENT_FETCHES` fields
- `tileToLngLatBounds()` method (no longer needed — mercator tiles are pre-aligned)

Keep:
- `pyramidMeta()` — still used by embedding code (`loadFullChunk`, `chunkPixelBounds`)
- `getPreviewArrayInfo()` — still used for level-0 preview queries (if any)
- `computeZoomRange()` — update to use `mercatorZoomRange` from metadata
- `reloadPreviewSource()` — still used for preview mode switching
- Everything related to embeddings, classification, overlays

**Step 3: Update computeZoomRange()**

Replace with:
```typescript
private computeZoomRange(): { minzoom: number; maxzoom: number } {
  if (!this.store?.meta.mercatorZoomRange) return { minzoom: 0, maxzoom: 18 };
  const [minzoom, maxzoom] = this.store.meta.mercatorZoomRange;
  // Allow MapLibre to overzoom 2 levels beyond stored data
  return { minzoom, maxzoom: Math.min(22, maxzoom + 2) };
}
```

**Step 4: Update addTo() source registration**

In `addTo()`, after computing zoom range, the `map.addSource` call stays the same. But remove the `mlConfig.MAX_PARALLEL_IMAGE_REQUESTS = 6` line — MapLibre's defaults are fine when each tile is a single chunk fetch.

**Step 5: Update references to old pyramid fields**

Search for `hasRgbPyramid`, `hasPcaPyramid`, `pyramidLevels`, `rgbPyramidArrs`, `pcaPyramidArrs` in `zarr-source.ts` and replace with mercator equivalents. Key places:

- First tile log message: update to show mercator zoom range
- `getPreviewArrayInfo()`: update to use mercator arrays for levels > 0
  - Actually, this method is only used by `fetchSingleChunk` which we're deleting. If it's also used by embedding code, keep it for level 0 only.

**Step 6: Build and verify**

```bash
cd ~/src/git/ucam-eo/tze
pnpm build
```

Expected: builds clean with no TypeScript errors.

**Step 7: Commit**

```bash
git add packages/maplibre-zarr-tessera/src/zarr-source.ts
git commit -m "feat: simplify tile handler to trivial mercator array slice

Removes ~400 lines of UTM reprojection, chunk caching, concurrency
limiting, and pyramid level selection. Each tile is now a single
256x256 zarr chunk fetch with no coordinate math."
```

---

## Task 8: Update cram test for mercator pyramids

**Files:**
- Modify: `~/src/git/ucam-eo/geotessera/tests/zarr.t`

**Step 1: Update the pyramid test section**

Replace the "Pyramid Building" test section (lines 171-215) with a mercator pyramid test:

```
Test: Mercator Pyramid Building on Preview Arrays
--------------------------------------------------

Build a zone store with RGB from the Cambridge tiles:

  $ geotessera-registry zarr-build \
  >   "$TESTDIR/cb_tiles_zarr" \
  >   --output-dir "$TESTDIR/zarr_mercator_test" \
  >   --year 2024 \
  >   --rgb 2>&1 | grep -E '(RGB preview|Zone)' | head -3 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)

Add mercator pyramids to the store:

  $ geotessera-registry zarr-build \
  >   "$TESTDIR/cb_tiles_zarr" \
  >   --output-dir "$TESTDIR/zarr_mercator_test" \
  >   --pyramid-only 2>&1 | grep -E '(mercator|Mercator|Pyramids)' | head -3 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)

Verify mercator pyramid structure exists in the zarr store:

  $ ZARR_STORE=$(find "$TESTDIR/zarr_mercator_test" -name "*.zarr" -type d | head -1)
  $ uv run python -c "
  > import zarr
  > store = zarr.open_group('$ZARR_STORE', mode='r')
  > attrs = dict(store.attrs)
  > print(f'has_rgb_mercator: {attrs.get(\"has_rgb_mercator\", False)}')
  > zoom_range = attrs.get('rgb_mercator_zoom_range', None)
  > print(f'zoom_range: {zoom_range}')
  > mercator = store['rgb_mercator']
  > levels = sorted(k for k in mercator.keys())
  > print(f'levels: {levels}')
  > first_level = mercator[levels[0]]
  > print(f'first_level_shape: {first_level.shape}')
  > print(f'first_level_chunks: {first_level.chunks}')
  > "
  has_rgb_mercator: True
  zoom_range: * (glob)
  levels: * (glob)
  first_level_shape: * (glob)
  first_level_chunks: * (glob)
```

**Step 2: Run tests**

```bash
cd ~/src/git/ucam-eo/geotessera/tests
uv run cram zarr.t
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add tests/zarr.t
git commit -m "test: update cram test for mercator pyramids"
```

---

## Task 9: End-to-end visual verification

**No code changes.** This task verifies the full pipeline works.

**Step 1: Generate mercator pyramids for a real store**

```bash
cd ~/src/git/ucam-eo/geotessera
geotessera-registry zarr-build <existing-store-dir> \
  --output-dir <output-dir> \
  --pyramid-only
```

**Step 2: Serve and test in browser**

```bash
cd ~/src/git/ucam-eo/tze
pnpm dev
```

Open browser, load a zone with mercator pyramids. Verify:

1. Tiles appear at all zoom levels (z=6 through z=12)
2. Tiles align precisely with basemap features
3. Smooth zoom transitions (MapLibre cross-fades)
4. Preview mode switch (rgb ↔ pca) reloads tiles
5. Double-click still loads full-res embeddings
6. Classification overlays still work
7. No blank screens during zoom transitions
8. Basemap switcher still works (satellite/terrain/streets/dark)

---

Plan complete and saved to `docs/plans/2026-02-26-mercator-pyramids-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
