# Zarr v3 Sharding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch zone stores from 1024x1024 uncompressed chunks to (4,4) inner chunks inside (256,256) shards with zstd compression, enabling efficient single-pixel lookups.

**Architecture:** Add two constants (SHARD_SIZE=256, INNER_CHUNK=4). Update `create_zone_store()` to use shards+compression. Snap grid to shard multiples. Update chunk iteration to use SHARD_SIZE. Buffer writes to shard alignment. Reads are transparent (zarr handles sharding).

**Tech Stack:** zarr 3.x (ShardingCodec, BloscCodec), numpy

**Design doc:** `docs/plans/2026-03-02-zarr-sharding-design.md`

---

### Task 1: Add Constants and Update Grid Snapping

**Files:**
- Modify: `geotessera/zarr_zone.py:30-41` (constants block)
- Modify: `geotessera/zarr_zone.py:184-230` (`compute_zone_grid()`)

**Step 1: Add constants**

After line 41 (`GLOBAL_BATCH_CHUNK_ROWS = 64`), add:

```python
# Zone store sharding
SHARD_SIZE = 256   # shard spatial dimension (pixels), aligned to tile size
INNER_CHUNK = 4    # inner chunk spatial dimension (pixels)
```

**Step 2: Snap grid dimensions to shard multiples**

In `compute_zone_grid()`, after computing `width_px` and `height_px`
(lines 226-227), snap up to the nearest multiple of SHARD_SIZE:

```python
width_px = round((extent_right - origin_easting) / pixel_size)
height_px = round((origin_northing - extent_bottom) / pixel_size)

# Snap to shard boundary (256 pixels) so shards are never partial
width_px = math.ceil(width_px / SHARD_SIZE) * SHARD_SIZE
height_px = math.ceil(height_px / SHARD_SIZE) * SHARD_SIZE
```

**Step 3: Verify import works**

Run: `uv run python -c "from geotessera.zarr_zone import SHARD_SIZE, INNER_CHUNK, compute_zone_grid; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: add sharding constants and snap zone grid to shard multiples"
```

---

### Task 2: Update `create_zone_store()` to Use Sharding + Compression

**Files:**
- Modify: `geotessera/zarr_zone.py:290-382` (`create_zone_store()`)

**Step 1: Update array creation**

Replace the three `create_array` calls for embeddings, scales, and rgb
with sharded+compressed versions.

Change embeddings (lines 308-316):
```python
from zarr.codecs import BloscCodec

store.create_array(
    "embeddings",
    shape=(zone_grid.height_px, zone_grid.width_px, N_BANDS),
    chunks=(INNER_CHUNK, INNER_CHUNK, N_BANDS),
    shards=(SHARD_SIZE, SHARD_SIZE, N_BANDS),
    dtype=np.int8,
    fill_value=np.int8(0),
    compressors=BloscCodec(cname="zstd", clevel=3),
    dimension_names=["northing", "easting", "band"],
)
```

Change scales (lines 317-325):
```python
store.create_array(
    "scales",
    shape=(zone_grid.height_px, zone_grid.width_px),
    chunks=(INNER_CHUNK, INNER_CHUNK),
    shards=(SHARD_SIZE, SHARD_SIZE),
    dtype=np.float32,
    fill_value=np.float32("nan"),
    compressors=BloscCodec(cname="zstd", clevel=3),
    dimension_names=["northing", "easting"],
)
```

Change rgb (lines 329-337):
```python
store.create_array(
    name,
    shape=(zone_grid.height_px, zone_grid.width_px, 4),
    chunks=(INNER_CHUNK, INNER_CHUNK, 4),
    shards=(SHARD_SIZE, SHARD_SIZE, 4),
    dtype=np.uint8,
    fill_value=np.uint8(0),
    compressors=BloscCodec(cname="zstd", clevel=3),
    dimension_names=["northing", "easting", "rgba"],
)
```

**Step 2: Test store creation**

Run: `uv run python -c "
from geotessera.zarr_zone import create_zone_store, ZoneGrid, TileInfo, SHARD_SIZE
import tempfile, pathlib
zg = ZoneGrid(zone=31, year=2024, canonical_epsg=32631,
    origin_easting=500000, origin_northing=6000000,
    width_px=SHARD_SIZE*2, height_px=SHARD_SIZE*2, pixel_size=10, tiles=[])
with tempfile.TemporaryDirectory() as td:
    store = create_zone_store(zg, pathlib.Path(td))
    emb = store['embeddings']
    print(f'shape={emb.shape}, chunks={emb.chunks}, shards={emb.shards}')
    print(f'dtype={emb.dtype}')
    sc = store['scales']
    print(f'scales: shape={sc.shape}, chunks={sc.chunks}, shards={sc.shards}')
print('OK')
"`

Expected: shape, chunks=(4,4,128), shards=(256,256,128), OK

**Step 3: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: use zarr v3 sharding with zstd compression for zone stores"
```

---

### Task 3: Update `add_rgb_to_existing_store()` to Use Sharding

**Files:**
- Modify: `geotessera/zarr_zone.py:945-967` (`add_rgb_to_existing_store()`)

**Step 1: Update the rgb array creation in `add_rgb_to_existing_store()`**

The `except KeyError` block (lines 957-967) creates the rgb array with
old-style chunks. Update to match the new sharded layout:

```python
except KeyError:
    from zarr.codecs import BloscCodec
    emb_shape = store["embeddings"].shape
    store.create_array(
        "rgb",
        shape=(emb_shape[0], emb_shape[1], 4),
        chunks=(INNER_CHUNK, INNER_CHUNK, 4),
        shards=(SHARD_SIZE, SHARD_SIZE, 4),
        dtype=np.uint8,
        fill_value=np.uint8(0),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dimension_names=["northing", "easting", "rgba"],
    )
```

**Step 2: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: use sharding for rgb array in add_rgb_to_existing_store"
```

---

### Task 4: Update Chunk Iteration to Use SHARD_SIZE

**Files:**
- Modify: `geotessera/zarr_zone.py:791-831` (`_sample_chunk_stats()`)
- Modify: `geotessera/zarr_zone.py:883-937` (`write_preview_pass()`)

Both functions use `emb_arr.chunks[:2]` to get chunk iteration size.
With sharding, `.chunks` returns the inner chunk (4,4) not the shard
(256,256). They need to iterate at shard granularity.

**Step 1: Update `_sample_chunk_stats()` signature**

The function receives `chunk_h` and `chunk_w` from the caller. No change
needed to the function itself — the caller just needs to pass SHARD_SIZE.

**Step 2: Update `compute_stretch_from_store()`**

In `compute_stretch_from_store()` (around line 843), change:
```python
chunk_h, chunk_w = emb_arr.chunks[:2]
```
to:
```python
chunk_h, chunk_w = SHARD_SIZE, SHARD_SIZE
```

**Step 3: Update `write_preview_pass()`**

In `write_preview_pass()` (around line 910), change:
```python
chunk_h, chunk_w = emb_arr.chunks[:2]
```
to:
```python
chunk_h, chunk_w = SHARD_SIZE, SHARD_SIZE
```

**Step 4: Verify import**

Run: `uv run python -c "from geotessera.zarr_zone import compute_stretch_from_store, write_preview_pass; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "fix: iterate at shard granularity instead of inner chunk size"
```

---

### Task 5: Update Module Docstring

**Files:**
- Modify: `geotessera/zarr_zone.py:1-16` (module docstring)

**Step 1: Update the store layout comment at the top of the file**

Replace lines 7-15:
```python
"""Zone-wide Zarr format for consolidated Tessera embeddings.

This module provides tools for building and reading Zarr v3 stores that
consolidate all tiles within a UTM zone into a single store per year.
This enables efficient spatial subsetting and cloud-native access.

Store layout (sharded, zstd-compressed):
    utm{zone:02d}_{year}.zarr/
        embeddings        # int8    (H, W, 128)  chunks=(4,4,128)   shards=(256,256,128)
        scales            # float32 (H, W)       chunks=(4,4)       shards=(256,256)
        rgb               # uint8   (H, W, 4)    chunks=(4,4,4)     shards=(256,256,4)   [optional]

NaN in scales indicates no-data (water or no coverage).
Per-pixel inner chunks enable O(2KB) single-pixel lookups via HTTP range
requests.  Tile-aligned shards (256x256) keep file counts reasonable.
"""
```

**Step 2: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "docs: update module docstring for sharded store layout"
```

---

### Task 6: Run Full Cram Tests

**Files:**
- Test: `tests/zarr.t`

**Step 1: Run the cram test suite**

Run: `cd tests && uv run cram zarr.t`

Expected: All tests pass. The zarr.t tests download real tiles,
build zone stores, and verify structure. They should work with
sharded stores since reads are transparent. If tests reference
specific chunk sizes in assertions, update them.

Note: The zone store build test (`zarr-build`) and global preview
test will exercise both the new sharded write path and the read
path. This is the primary integration test.

**Step 2: Fix any failures**

If a test checks `.chunks` metadata, it will now see (4,4,...) instead
of (1024,1024,...). Update assertions accordingly.

**Step 3: Commit any test fixes**

```
git add tests/zarr.t
git commit -m "test: update assertions for sharded zone stores"
```

---
