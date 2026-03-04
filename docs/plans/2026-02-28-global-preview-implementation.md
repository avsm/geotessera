# Global Preview Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken `build_global_preview` with a chunk-aligned, dask-orchestrated, incrementally-updatable implementation that never races on zarr chunk writes.

**Architecture:** Fixed 360x180 degree global grid at 0.0001 deg resolution. Each UTM zone is reprojected chunk-by-chunk (512x512 output tiles) using `dask.delayed` + `rasterio.warp.reproject`. Tasks write directly to non-overlapping zarr chunk positions, eliminating concurrent write races. Pyramid levels are updated incrementally per-zone. The store is created once and updated in place.

**Tech Stack:** zarr v3, dask (delayed + threaded scheduler), rasterio, pyproj, numpy. Existing deps only.

**Design doc:** `docs/plans/2026-02-28-global-preview-redesign.md`

---

### Task 1: Add Constants and `_ensure_global_store()`

**Files:**
- Modify: `geotessera/zarr_zone.py` (add constants near top after line 30, add new function after line 1092)

**Step 1: Add global grid constants after `RGB_PREVIEW_BANDS` (line 31)**

```python
# Global preview grid (fixed extent, never changes)
GLOBAL_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
GLOBAL_BASE_RES = 0.0001  # degrees (~10m at equator)
GLOBAL_LEVEL0_W = 3_600_000  # ceil(360 / 0.0001)
GLOBAL_LEVEL0_H = 1_800_000  # ceil(180 / 0.0001)
GLOBAL_CHUNK = 512
GLOBAL_NUM_BANDS = 4
GLOBAL_DEFAULT_LEVELS = 7
GLOBAL_BATCH_CHUNK_ROWS = 64  # chunk-rows per dask compute batch
```

**Step 2: Add `_ensure_global_store()` function**

Add after the `read_utm_region()` function (after line 1092). This function idempotently creates the global store with all pyramid levels pre-allocated:

```python
def _ensure_global_store(store_path: Path, num_levels: int) -> None:
    """Create the global preview store with fixed dimensions if it doesn't exist.

    If the store already exists with correct dimensions, this is a no-op.
    Creates level 0 through level (num_levels-1), each with an 'rgb' array
    and a 'band' coordinate array.
    """
    import json as _json

    import zarr
    from zarr.codecs import BloscCodec

    if store_path.exists():
        # Validate existing store has the right shape
        root = zarr.open_group(str(store_path), mode="r")
        if "0/rgb" in root:
            shape = root["0/rgb"].shape
            if shape == (GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W, GLOBAL_NUM_BANDS):
                return  # Store exists and is correct
            raise ValueError(
                f"Existing store has shape {shape}, expected "
                f"({GLOBAL_LEVEL0_H}, {GLOBAL_LEVEL0_W}, {GLOBAL_NUM_BANDS})"
            )

    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)

    h, w = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    band_data = np.arange(GLOBAL_NUM_BANDS, dtype=np.int32)

    for lvl in range(num_levels):
        if h < 1 or w < 1:
            break
        # Create level group
        level_dir = os.path.join(str(store_path), str(lvl))
        os.makedirs(level_dir, exist_ok=True)
        group_meta = os.path.join(level_dir, "zarr.json")
        if not os.path.exists(group_meta):
            with open(group_meta, "w") as f:
                _json.dump(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}},
                    f,
                )
        # Re-open to pick up the group
        root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)

        root.create_array(
            f"{lvl}/rgb",
            shape=(h, w, GLOBAL_NUM_BANDS),
            chunks=(GLOBAL_CHUNK, GLOBAL_CHUNK, GLOBAL_NUM_BANDS),
            dtype=np.uint8,
            fill_value=np.uint8(0),
            compressors=BloscCodec(cname="zstd", clevel=3),
            dimension_names=["lat", "lon", "band"],
        )
        root.create_array(
            f"{lvl}/band",
            data=band_data,
            chunks=(GLOBAL_NUM_BANDS,),
        )
        h //= 2
        w //= 2

    # Write multiscales metadata
    from topozarr.metadata import create_multiscale_metadata

    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
    actual_levels = len([k for k in root.keys() if k.isdigit()])
    ms_attrs = create_multiscale_metadata(actual_levels, "EPSG:4326", "mean")
    ms_attrs["multiscales"]["crs"] = "EPSG:4326"
    west, south, east, north = GLOBAL_BOUNDS
    ms_attrs["spatial"] = {
        "bounds": [west, south, east, north],
        "resolution": GLOBAL_BASE_RES,
    }
    root.attrs.update(ms_attrs)
    zarr.consolidate_metadata(str(store_path))
```

**Step 3: Verify it works with a quick smoke test**

Run:
```bash
cd /Users/avsm/src/git/ucam-eo/geotessera
uv run python -c "
from pathlib import Path
import tempfile, shutil
from geotessera.zarr_zone import _ensure_global_store, GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
tmp = Path(tempfile.mkdtemp()) / 'test.zarr'
_ensure_global_store(tmp, num_levels=3)
import zarr
r = zarr.open_group(str(tmp), mode='r')
print(f'level0 shape: {r[\"0/rgb\"].shape}')
print(f'level1 shape: {r[\"1/rgb\"].shape}')
print(f'level2 shape: {r[\"2/rgb\"].shape}')
print(f'band: {list(r[\"0/band\"][:].tolist())}')
print(f'bounds: {dict(r.attrs)[\"spatial\"][\"bounds\"]}')
shutil.rmtree(tmp.parent)
"
```

Expected output:
```
level0 shape: (1800000, 3600000, 4)
level1 shape: (900000, 1800000, 4)
level2 shape: (450000, 900000, 4)
band: [0, 1, 2, 3]
bounds: [-180.0, -90.0, 180.0, 90.0]
```

**Step 4: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add global grid constants and _ensure_global_store()"
```

---

### Task 2: Implement `_reproject_chunk()` (single-chunk reprojection)

**Files:**
- Modify: `geotessera/zarr_zone.py` (add function after `_ensure_global_store`)

**Step 1: Write `_reproject_chunk()`**

This is the core unit of work. It reprojects one 512x512 output chunk from UTM source data and writes directly to the zarr array. It replaces `_reproject_tile()`.

```python
def _reproject_chunk(
    global_arr,
    chunk_row: int,
    chunk_col: int,
    src_arr,
    src_epsg: int,
    src_pixel: float,
    src_origin_e: float,
    src_origin_n: float,
    src_h: int,
    src_w: int,
    to_utm,
) -> bool:
    """Reproject one 512x512 output chunk from UTM source and write to global_arr.

    Each call writes to a unique (chunk_row, chunk_col) position, so concurrent
    calls to different positions are safe.

    Args:
        global_arr: Zarr array handle for the global level-0 rgb array.
        chunk_row: Output chunk row index (pixel row = chunk_row * GLOBAL_CHUNK).
        chunk_col: Output chunk col index (pixel col = chunk_col * GLOBAL_CHUNK).
        src_arr: Zarr array handle for the UTM zone's rgb array.
        src_epsg: EPSG code of the source UTM zone.
        src_pixel: Source pixel size in metres.
        src_origin_e: Easting of source origin (top-left corner).
        src_origin_n: Northing of source origin (top-left corner).
        src_h: Source array height in pixels.
        src_w: Source array width in pixels.
        to_utm: pyproj.Transformer from EPSG:4326 to source EPSG.

    Returns:
        True if any non-zero data was written, False if chunk was all-empty.
    """
    from affine import Affine
    from rasterio.enums import Resampling
    import rasterio.warp

    west, south, east, north = GLOBAL_BOUNDS
    row0 = chunk_row * GLOBAL_CHUNK
    col0 = chunk_col * GLOBAL_CHUNK
    tile_h = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_H - row0)
    tile_w = min(GLOBAL_CHUNK, GLOBAL_LEVEL0_W - col0)
    if tile_h <= 0 or tile_w <= 0:
        return False

    # Geographic extent of this output chunk
    tile_west = west + col0 * GLOBAL_BASE_RES
    tile_north = north - row0 * GLOBAL_BASE_RES
    tile_east = tile_west + tile_w * GLOBAL_BASE_RES
    tile_south = tile_north - tile_h * GLOBAL_BASE_RES

    dst_transform = Affine(
        GLOBAL_BASE_RES, 0, tile_west,
        0, -GLOBAL_BASE_RES, tile_north,
    )

    # Back-project chunk corners to UTM to find source window
    sample_lons = [tile_west, tile_east, tile_west, tile_east,
                   (tile_west + tile_east) / 2]
    sample_lats = [tile_north, tile_north, tile_south, tile_south,
                   (tile_north + tile_south) / 2]
    try:
        utm_xs, utm_ys = to_utm.transform(sample_lons, sample_lats)
    except Exception:
        return False

    if any(not math.isfinite(v) for v in list(utm_xs) + list(utm_ys)):
        return False

    # Compute source window with padding
    pad = 16
    r_min = max(0, int((src_origin_n - max(utm_ys)) / src_pixel) - pad)
    r_max = min(src_h, int(math.ceil(
        (src_origin_n - min(utm_ys)) / src_pixel
    )) + pad)
    c_min = max(0, int((min(utm_xs) - src_origin_e) / src_pixel) - pad)
    c_max = min(src_w, int(math.ceil(
        (max(utm_xs) - src_origin_e) / src_pixel
    )) + pad)

    if r_max <= r_min or c_max <= c_min:
        return False

    window = np.asarray(src_arr[r_min:r_max, c_min:c_max, :])
    if not window.any():
        return False

    src_data = np.transpose(window.astype(np.float32), (2, 0, 1))
    del window

    win_transform = Affine(
        src_pixel, 0, src_origin_e + c_min * src_pixel,
        0, -src_pixel, src_origin_n - r_min * src_pixel,
    )

    dst_data = np.full(
        (GLOBAL_NUM_BANDS, tile_h, tile_w), np.nan, dtype=np.float32,
    )

    try:
        rasterio.warp.reproject(
            source=src_data,
            destination=dst_data,
            src_transform=win_transform,
            src_crs=f"EPSG:{src_epsg}",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
    except Exception:
        return False

    del src_data
    dst_data = np.nan_to_num(dst_data, nan=0.0)
    dst_data = np.clip(dst_data, 0, 255).astype(np.uint8)
    out = np.transpose(dst_data, (1, 2, 0))  # (tile_h, tile_w, 4)
    del dst_data

    if not out.any():
        return False

    # Write directly to the global array at this chunk's position.
    # This is the ONLY writer for this (chunk_row, chunk_col), so no races.
    global_arr[row0 : row0 + tile_h, col0 : col0 + tile_w, :] = out
    return True
```

**Step 2: Verify it compiles**

Run:
```bash
uv run python -c "from geotessera.zarr_zone import _reproject_chunk; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add _reproject_chunk() for chunk-aligned reprojection"
```

---

### Task 3: Implement `_reproject_zone()` (batched dask orchestration)

**Files:**
- Modify: `geotessera/zarr_zone.py` (add function after `_reproject_chunk`)

**Step 1: Write `_reproject_zone()`**

This function orchestrates reprojection of a single zone into the global store using dask.delayed in batches:

```python
def _reproject_zone(
    store_path: Path,
    zone_num: int,
    zone_store_path: Path,
    zone_epsg: int,
    zone_transform: list,
    zone_shape: tuple,
    workers: int,
    console: Optional["rich.console.Console"] = None,
) -> Tuple[int, int, int, int]:
    """Reproject one zone's RGB into level 0 of the global store.

    Uses dask.delayed to parallelise chunk-level reprojection tasks.
    Each task writes to a unique chunk position, eliminating write races.
    Tasks are submitted in batches of GLOBAL_BATCH_CHUNK_ROWS chunk-rows
    to bound dask graph size and provide progress feedback.

    Returns:
        (row_start, row_end, col_start, col_end) of the affected region
        in pixel coordinates, snapped to chunk boundaries.
    """
    import dask
    import zarr
    from pyproj import Transformer

    src_pixel = zone_transform[0]
    src_origin_e = zone_transform[2]
    src_origin_n = zone_transform[5]
    src_h, src_w = zone_shape[:2]

    west, south, east, north = GLOBAL_BOUNDS

    # Compute the zone's WGS84 bounding box from its metadata
    to_4326 = Transformer.from_crs(
        f"EPSG:{zone_epsg}", "EPSG:4326", always_xy=True,
    )
    corners_utm = [
        (src_origin_e, src_origin_n),
        (src_origin_e + src_w * src_pixel, src_origin_n),
        (src_origin_e, src_origin_n - src_h * src_pixel),
        (src_origin_e + src_w * src_pixel, src_origin_n - src_h * src_pixel),
    ]
    mid_e = src_origin_e + src_w * src_pixel / 2
    mid_n = src_origin_n - src_h * src_pixel / 2
    corners_utm += [
        (mid_e, src_origin_n),
        (mid_e, src_origin_n - src_h * src_pixel),
        (src_origin_e, mid_n),
        (src_origin_e + src_w * src_pixel, mid_n),
    ]
    corners_4326 = [to_4326.transform(e, n) for e, n in corners_utm]
    lons = [c[0] for c in corners_4326]
    lats = [c[1] for c in corners_4326]

    zlon_min, zlon_max = min(lons), max(lons)
    zlat_min, zlat_max = min(lats), max(lats)

    # Snap to chunk boundaries (expand outward)
    col_start = max(0, (int(math.floor((zlon_min - west) / GLOBAL_BASE_RES))
                        // GLOBAL_CHUNK * GLOBAL_CHUNK))
    col_end = min(GLOBAL_LEVEL0_W,
                  ((int(math.ceil((zlon_max - west) / GLOBAL_BASE_RES))
                    + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK))
    row_start = max(0, (int(math.floor((north - zlat_max) / GLOBAL_BASE_RES))
                        // GLOBAL_CHUNK * GLOBAL_CHUNK))
    row_end = min(GLOBAL_LEVEL0_H,
                  ((int(math.ceil((north - zlat_min) / GLOBAL_BASE_RES))
                    + GLOBAL_CHUNK - 1) // GLOBAL_CHUNK * GLOBAL_CHUNK))

    if col_end <= col_start or row_end <= row_start:
        if console is not None:
            console.print(f"    [yellow]Zone {zone_num}: no output region[/yellow]")
        return (0, 0, 0, 0)

    n_chunk_rows = (row_end - row_start) // GLOBAL_CHUNK
    n_chunk_cols = (col_end - col_start) // GLOBAL_CHUNK
    chunk_row_start = row_start // GLOBAL_CHUNK
    chunk_col_start = col_start // GLOBAL_CHUNK

    if console is not None:
        total_chunks = n_chunk_rows * n_chunk_cols
        console.print(
            f"    Zone {zone_num:02d}: rows {row_start}-{row_end}, "
            f"cols {col_start}-{col_end} "
            f"({n_chunk_rows}x{n_chunk_cols} = {total_chunks} chunks)"
        )

    # Open zarr handles (shared across tasks — zarr handles are thread-safe
    # for reads; writes to non-overlapping regions are safe)
    global_root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
    global_arr = global_root["0/rgb"]
    zone_store = zarr.open_group(str(zone_store_path), mode="r")
    src_arr = zone_store["rgb"]

    to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{zone_epsg}", always_xy=True,
    )

    # Process in batches of chunk-rows
    chunks_written = 0
    chunks_total = n_chunk_rows * n_chunk_cols

    for batch_start in range(0, n_chunk_rows, GLOBAL_BATCH_CHUNK_ROWS):
        batch_end = min(batch_start + GLOBAL_BATCH_CHUNK_ROWS, n_chunk_rows)

        tasks = []
        for cr_offset in range(batch_start, batch_end):
            cr = chunk_row_start + cr_offset
            for cc_offset in range(n_chunk_cols):
                cc = chunk_col_start + cc_offset
                task = dask.delayed(_reproject_chunk)(
                    global_arr=global_arr,
                    chunk_row=cr,
                    chunk_col=cc,
                    src_arr=src_arr,
                    src_epsg=zone_epsg,
                    src_pixel=src_pixel,
                    src_origin_e=src_origin_e,
                    src_origin_n=src_origin_n,
                    src_h=src_h,
                    src_w=src_w,
                    to_utm=to_utm,
                )
                tasks.append(task)

        results = dask.compute(*tasks, scheduler="threads",
                               num_workers=workers)
        batch_written = sum(1 for r in results if r)
        chunks_written += batch_written

        if console is not None:
            done = min((batch_end) * n_chunk_cols, chunks_total)
            pct = int(100 * done / chunks_total)
            console.print(
                f"      [{pct:3d}%] {done}/{chunks_total} chunks "
                f"({chunks_written} with data)"
            )

    return (row_start, row_end, col_start, col_end)
```

**Step 2: Verify it compiles**

Run:
```bash
uv run python -c "from geotessera.zarr_zone import _reproject_zone; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add _reproject_zone() with batched dask orchestration"
```

---

### Task 4: Implement `_coarsen_zone_pyramid()` (incremental pyramid)

**Files:**
- Modify: `geotessera/zarr_zone.py` (add function after `_reproject_zone`)

**Step 1: Write `_coarsen_zone_pyramid()`**

This updates pyramid levels 1-N for the region affected by a single zone:

```python
def _coarsen_zone_pyramid(
    store_path: Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    num_levels: int,
    workers: int,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Update pyramid levels 1 through num_levels-1 for the affected region.

    Reads from the previous level and writes coarsened data to the current
    level, processing in row-strips parallelised with dask.delayed.

    Args:
        store_path: Path to the global zarr store.
        row_start, row_end: Affected pixel row range at level 0.
        col_start, col_end: Affected pixel col range at level 0.
        num_levels: Total number of pyramid levels.
        workers: Number of parallel workers.
    """
    import dask
    import zarr

    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)

    prev_row_start, prev_row_end = row_start, row_end
    prev_col_start, prev_col_end = col_start, col_end

    for lvl in range(1, num_levels):
        prev_arr_path = f"{lvl - 1}/rgb"
        cur_arr_path = f"{lvl}/rgb"

        if prev_arr_path not in root or cur_arr_path not in root:
            break

        prev_arr = root[prev_arr_path]
        cur_arr = root[cur_arr_path]
        cur_h, cur_w = cur_arr.shape[:2]

        # Map the affected region to this level (halve coordinates)
        # Snap to chunk boundaries at this level
        lr_start = max(0, (prev_row_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lr_end = min(cur_h, ((prev_row_end // 2 + GLOBAL_CHUNK - 1)
                             // GLOBAL_CHUNK * GLOBAL_CHUNK))
        lc_start = max(0, (prev_col_start // 2) // GLOBAL_CHUNK * GLOBAL_CHUNK)
        lc_end = min(cur_w, ((prev_col_end // 2 + GLOBAL_CHUNK - 1)
                             // GLOBAL_CHUNK * GLOBAL_CHUNK))

        if lr_end <= lr_start or lc_end <= lc_start:
            break

        if console is not None:
            console.print(
                f"    Level {lvl}: rows {lr_start}-{lr_end}, "
                f"cols {lc_start}-{lc_end}"
            )

        strip_h = GLOBAL_CHUNK

        def _coarsen_strip(r0, _prev_arr=prev_arr, _cur_arr=cur_arr,
                           _lc_start=lc_start, _lc_end=lc_end,
                           _cur_h=cur_h):
            r1 = min(r0 + strip_h, _cur_h)
            sr0 = r0 * 2
            sr1 = min(sr0 + (r1 - r0) * 2, _prev_arr.shape[0])
            src_w_region = (_lc_end - _lc_start) * 2
            sc0 = _lc_start * 2
            sc1 = min(sc0 + src_w_region, _prev_arr.shape[1])
            strip = np.asarray(
                _prev_arr[sr0:sr1, sc0:sc1, :]
            ).astype(np.float32)
            th = strip.shape[0] // 2
            tw = strip.shape[1] // 2
            if th == 0 or tw == 0:
                return
            coarsened = (
                strip[: th * 2, : tw * 2, :]
                .reshape(th, 2, tw, 2, GLOBAL_NUM_BANDS)
                .mean(axis=(1, 3))
            )
            result = np.clip(coarsened, 0, 255).astype(np.uint8)
            _cur_arr[r0 : r0 + th, _lc_start : _lc_start + tw, :] = result

        strip_starts = list(range(lr_start, lr_end, strip_h))
        tasks = [dask.delayed(_coarsen_strip)(r0) for r0 in strip_starts]
        dask.compute(*tasks, scheduler="threads", num_workers=workers)

        # Prepare for next level
        prev_row_start, prev_row_end = lr_start, lr_end
        prev_col_start, prev_col_end = lc_start, lc_end
```

**Step 2: Verify it compiles**

Run:
```bash
uv run python -c "from geotessera.zarr_zone import _coarsen_zone_pyramid; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add _coarsen_zone_pyramid() for incremental pyramid updates"
```

---

### Task 5: Rewrite `build_global_preview()` as orchestrator

**Files:**
- Modify: `geotessera/zarr_zone.py` — replace the existing `build_global_preview()` function (lines 1094-1310) and delete `_reproject_tile()` (lines 1313-1424) and `_write_global_store()` (lines 1427-1817)

**Step 1: Replace `build_global_preview()`**

Delete the old `build_global_preview`, `_reproject_tile`, and `_write_global_store` functions. Replace with:

```python
def build_global_preview(
    zarr_dir: Path,
    year: int,
    zones: Optional[List[int]] = None,
    num_levels: int = GLOBAL_DEFAULT_LEVELS,
    workers: int = 4,
    console: Optional["rich.console.Console"] = None,
) -> Path:
    """Create or update global EPSG:4326 preview store from per-zone UTM stores.

    Reprojects each zone's RGB array from UTM to WGS84 into a fixed
    global grid, then updates the pyramid levels for the affected region.

    The output store is always ``{zarr_dir}/global_rgb_{year}.zarr``.
    If the store already exists, only the specified zones are re-processed
    (incremental update).  The store uses a fixed 360x180 degree extent
    so adding zones never changes the array dimensions.

    Args:
        zarr_dir: Directory containing per-zone ``.zarr`` stores.
        year: Year to filter zone store filenames.
        zones: Optional list of UTM zone numbers to include. If *None*,
            all matching stores are used.
        num_levels: Number of resolution levels in the output pyramid.
        workers: Number of parallel reprojection workers.
        console: Optional Rich Console for status messages.

    Returns:
        Path to the global zarr store.
    """
    import gc

    import zarr

    if console is not None:
        console.print(
            f"[bold]Building global preview (year={year})[/bold]"
        )

    # ------------------------------------------------------------------
    # 1. Discover zone stores
    # ------------------------------------------------------------------
    pattern = re.compile(rf"^utm(\d{{2}})_{year}\.zarr$")
    zone_stores: Dict[int, Path] = {}

    for entry in sorted(zarr_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m is None:
            continue
        zone_num = int(m.group(1))
        if zones is not None and zone_num not in zones:
            continue
        zone_stores[zone_num] = entry

    if not zone_stores:
        msg = f"No zone stores found in {zarr_dir} for year {year}"
        if zones is not None:
            msg += f" (zones filter: {zones})"
        raise FileNotFoundError(msg)

    if console is not None:
        console.print(
            f"  Found {len(zone_stores)} zone store(s): "
            f"{sorted(zone_stores.keys())}"
        )

    # ------------------------------------------------------------------
    # 2. Read zone metadata (attrs only, no pixel data)
    # ------------------------------------------------------------------
    zone_infos: Dict[int, dict] = {}

    for zone_num, store_path in sorted(zone_stores.items()):
        store = zarr.open_group(str(store_path), mode="r")
        attrs = dict(store.attrs)

        if "rgb" not in store:
            if console is not None:
                console.print(
                    f"  [yellow]Zone {zone_num}: no rgb array, skipping[/yellow]"
                )
            continue

        zone_infos[zone_num] = {
            "store_path": store_path,
            "epsg": int(attrs["crs_epsg"]),
            "transform": list(attrs["transform"]),
            "shape": store["rgb"].shape,
        }

    if not zone_infos:
        raise FileNotFoundError(
            "No zone stores with rgb arrays found"
        )

    # ------------------------------------------------------------------
    # 3. Ensure the global store exists
    # ------------------------------------------------------------------
    output_path = zarr_dir / f"global_rgb_{year}.zarr"

    if console is not None:
        console.print(f"  Output: {output_path}")

    _ensure_global_store(output_path, num_levels)

    if console is not None:
        console.print(
            f"  Global grid: {GLOBAL_LEVEL0_W}x{GLOBAL_LEVEL0_H} "
            f"@ {GLOBAL_BASE_RES} deg, {num_levels} levels"
        )

    # ------------------------------------------------------------------
    # 4. Reproject each zone and update pyramid
    # ------------------------------------------------------------------
    for zone_num, zinfo in sorted(zone_infos.items()):
        if console is not None:
            console.print(
                f"\n  [bold]Processing zone {zone_num:02d} "
                f"(EPSG:{zinfo['epsg']})[/bold]"
            )

        row_start, row_end, col_start, col_end = _reproject_zone(
            store_path=output_path,
            zone_num=zone_num,
            zone_store_path=zinfo["store_path"],
            zone_epsg=zinfo["epsg"],
            zone_transform=zinfo["transform"],
            zone_shape=zinfo["shape"],
            workers=workers,
            console=console,
        )

        if row_end <= row_start or col_end <= col_start:
            continue

        if console is not None:
            console.print(f"    Building pyramid...")

        _coarsen_zone_pyramid(
            store_path=output_path,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            num_levels=num_levels,
            workers=workers,
            console=console,
        )

        gc.collect()

    # ------------------------------------------------------------------
    # 5. Re-consolidate metadata
    # ------------------------------------------------------------------
    zarr.consolidate_metadata(str(output_path))

    if console is not None:
        console.print(
            f"\n  [bold green]Global store updated: {output_path}[/bold green]"
        )
        console.print(
            f"  Zones processed: {sorted(zone_infos.keys())}"
        )

    return output_path
```

**Step 2: Verify no import errors**

Run:
```bash
uv run python -c "from geotessera.zarr_zone import build_global_preview; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: rewrite build_global_preview() as incremental zone-by-zone orchestrator

Replaces monolithic build with per-zone dask-orchestrated reprojection.
Deletes _reproject_tile() and _write_global_store()."
```

---

### Task 6: Update CLI in `registry_cli.py`

**Files:**
- Modify: `geotessera/registry_cli.py` — update `global_preview_command()` (lines 2659-2700) and CLI parser (lines 3390-3430)

**Step 1: Update `global_preview_command()`**

The new signature drops `--output` (output is derived from zarr_dir + year):

```python
def global_preview_command(args):
    """Build global EPSG:4326 preview store from per-zone UTM stores."""
    from .zarr_zone import build_global_preview

    zarr_dir = Path(args.zarr_dir)
    year = args.year
    num_levels = args.levels
    num_workers = args.workers

    # Parse zones into list of ints
    zones = None
    if args.zones:
        try:
            zones = [int(z.strip()) for z in args.zones.split(",")]
        except ValueError:
            console.print("[red]Error: --zones must be comma-separated integers[/red]")
            return 1

    console.print(
        f"[bold]Building global preview store[/bold]\n"
        f"  Input:   {zarr_dir}\n"
        f"  Year:    {year}\n"
        f"  Levels:  {num_levels}\n"
        f"  Workers: {num_workers}"
    )
    if zones:
        console.print(f"  Zones:   {', '.join(str(z) for z in zones)}")

    result = build_global_preview(
        zarr_dir=zarr_dir,
        year=year,
        zones=zones,
        num_levels=num_levels,
        workers=num_workers,
        console=console,
    )

    console.print(f"\n[bold green]Global preview store: {result}[/bold green]")
    return 0
```

**Step 2: Update CLI parser**

Remove the `--output` argument from the parser. The output section
currently at lines 3400-3405 should be deleted:

```python
    # Remove these lines:
    # global_preview_parser.add_argument(
    #     "--output",
    #     type=Path,
    #     required=True,
    #     help="Output path for global preview store",
    # )
```

**Step 3: Verify CLI parse works**

Run:
```bash
uv run geotessera-registry global-preview --help
```

Expected: help text without `--output`, showing `zarr_dir`, `--year`, `--zones`, `--levels`, `--workers`.

**Step 4: Commit**

```bash
git add geotessera/registry_cli.py
git commit -m "feat: update global-preview CLI to drop --output flag

Output path is now derived from zarr_dir + year. Supports incremental
updates by re-running with different --zones."
```

---

### Task 7: Update cram test in `tests/zarr.t`

**Files:**
- Modify: `tests/zarr.t` (lines 185-220)

**Step 1: Update the test**

The test needs to drop the `--output` flag and adjust expectations for
the new fixed-grid store. The test uses a small zone store built from
Cambridge tiles, so the output will be a sparse global store.

Replace lines 185-220 with:

```
Build global preview store from the zone store:

  $ geotessera-registry global-preview \
  >   "$TESTDIR/zarr_global_test" \
  >   --year 2024 \
  >   --levels 3 \
  >   --workers 2 2>&1 | grep -E '(Building|Found|Processing|Global store)' | head -4 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)
  * (glob)

Verify global preview store structure has multiscales metadata:

  $ uv run python -c "
  > import json
  > with open('$TESTDIR/zarr_global_test/global_rgb_2024.zarr/zarr.json') as f:
  >     meta = json.load(f)
  > ms = meta['attributes']['multiscales']
  > print(f'crs: {ms[\"crs\"]}')
  > print(f'num_levels: {len(ms[\"layout\"])}')
  > print(f'has_consolidated: {\"consolidated_metadata\" in meta}')
  > cm = meta['consolidated_metadata']['metadata']
  > has_rgb = any('rgb' in k for k in cm)
  > print(f'has_rgb_arrays: {has_rgb}')
  > first_arr = [v for k, v in cm.items() if 'rgb' in k][0]
  > has_blosc = any(c.get('name') == 'blosc' for c in first_arr.get('codecs', []))
  > print(f'has_blosc: {has_blosc}')
  > print(f'dimension_names: {first_arr.get(\"dimension_names\")}')
  > sp = meta['attributes']['spatial']
  > print(f'bounds: {sp[\"bounds\"]}')
  > "
  crs: EPSG:4326
  num_levels: 3
  has_consolidated: True
  has_rgb_arrays: True
  has_blosc: True
  dimension_names: ['lat', 'lon', 'band']
  bounds: [-180.0, -90.0, 180.0, 90.0]
```

**Step 2: Run the cram test**

Run:
```bash
cd /Users/avsm/src/git/ucam-eo/geotessera/tests && uv run cram zarr.t
```

The test will likely fail on the glob patterns for the console output —
update the glob lines based on actual output. The key assertion is that
`bounds: [-180.0, -90.0, 180.0, 90.0]` appears (fixed global extent)
and all other structural checks pass.

**Step 3: Iterate on glob patterns until test passes**

Cram tests use `*` and `(glob)` for flexible matching. Adjust the grep
filters and glob patterns based on the actual output from the new code.

**Step 4: Commit**

```bash
git add tests/zarr.t
git commit -m "test: update global-preview cram test for fixed global grid"
```

---

### Task 8: Clean up deleted code and verify end-to-end

**Files:**
- Modify: `geotessera/zarr_zone.py` — verify `_reproject_tile` and `_write_global_store` are fully removed
- Verify: `scripts/patch_global_bounds.py` — still works (uses its own bounds computation, not affected)

**Step 1: Verify old functions are gone**

Run:
```bash
uv run python -c "
from geotessera import zarr_zone
for name in ['_reproject_tile', '_write_global_store']:
    assert not hasattr(zarr_zone, name), f'{name} still exists!'
print('Old functions removed: OK')
"
```

Expected: `Old functions removed: OK`

**Step 2: Run full cram test suite**

Run:
```bash
cd /Users/avsm/src/git/ucam-eo/geotessera/tests && uv run cram *.t
```

All tests should pass.

**Step 3: Verify `dask` is available as a dependency**

Run:
```bash
uv run python -c "import dask; print(f'dask {dask.__version__}')"
```

If dask is not installed, add it:
```bash
uv add dask
```

**Step 4: Commit any final cleanup**

```bash
git add -u
git commit -m "chore: clean up deleted functions and verify dependencies"
```

---

### Task 9: Smoke test with a real zone store (manual)

This task is for manual verification, not automated.

**Step 1: Run on a single zone**

```bash
geotessera-registry global-preview /path/to/zarr/v0/ \
    --year 2025 --zones 30 --workers 4
```

Verify:
- No OOM (watch with `top` or `htop`)
- Console shows progress per batch
- Output store exists at `/path/to/zarr/v0/global_rgb_2025.zarr`
- Pyramid levels 0-6 all have data in the zone 30 column range

**Step 2: Run on a second zone (incremental)**

```bash
geotessera-registry global-preview /path/to/zarr/v0/ \
    --year 2025 --zones 31 --workers 4
```

Verify:
- Store was not recreated (level 0 shape unchanged)
- Zone 31's column range now has data
- Zone 30's data is still intact

**Step 3: Verify in viewer**

Load the store in the tze viewer and verify:
- No missing streaks
- Both zones render contiguously
- Pyramid levels display at appropriate zoom levels
