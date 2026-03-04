# Shard-First Parallel Zarr-Build Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the tile-first zarr-build pipeline with a shard-first approach where each shard is an independent parallel work item that mmap-reads tile slices, applies masking, and writes once.

**Architecture:** Pre-compute a shard index mapping each shard to its overlapping tile slices. Use `_run_parallel` (existing ThreadPoolExecutor + Rich progress helper) to write all shards in parallel. Tile embeddings are mmap'd read-only (kernel page cache handles sharing); scales/landmask masking is applied per-shard on small 256×256 slices.

**Tech Stack:** numpy (mmap), rasterio (windowed reads), zarr v3 (sharded stores), concurrent.futures (via existing `_run_parallel`)

---

### Task 1: Add `ShardTileOverlap` and `ShardSpec` dataclasses

**Files:**
- Modify: `geotessera/zarr_zone.py:53-81` (after existing dataclasses)

**Step 1: Add the dataclasses**

Add after the `ZoneGrid` dataclass (line 81):

```python
@dataclass
class ShardTileOverlap:
    """One tile's contribution to one shard — precomputed slice coordinates."""

    embedding_path: str
    scales_path: str
    landmask_path: str
    # Tile-local region to read
    t_row_start: int
    t_row_end: int
    t_col_start: int
    t_col_end: int
    # Shard-buffer region to write into
    s_row_start: int
    s_row_end: int
    s_col_start: int
    s_col_end: int


@dataclass
class ShardSpec:
    """Everything a shard worker needs to write one complete shard."""

    sr: int  # shard row index
    sc: int  # shard col index
    row_px: int  # pixel row in zone grid (sr * SHARD_SIZE)
    col_px: int  # pixel col in zone grid (sc * SHARD_SIZE)
    tiles: List[ShardTileOverlap]
```

**Step 2: Verify import**

Run: `uv run python -c "from geotessera.zarr_zone import ShardSpec, ShardTileOverlap; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: add ShardSpec and ShardTileOverlap dataclasses"
```

---

### Task 2: Implement `build_shard_index()`

**Files:**
- Modify: `geotessera/zarr_zone.py` — add function after `tile_pixel_offset` (after line 255)

**Step 1: Write `build_shard_index`**

```python
def build_shard_index(
    tile_infos: List[TileInfo],
    zone_grid: ZoneGrid,
) -> List[ShardSpec]:
    """Build a shard index: for each non-empty shard, list overlapping tile slices.

    Pure arithmetic — no file I/O.  Returns only shards that have at least
    one tile overlap (empty shards are left as zarr fill values).
    """
    # Collect overlaps keyed by (shard_row, shard_col)
    shard_map: Dict[Tuple[int, int], List[ShardTileOverlap]] = {}

    for ti in tile_infos:
        row, col = tile_pixel_offset(ti, zone_grid)
        h, w = ti.height, ti.width

        sr_start = row // SHARD_SIZE
        sr_end = (row + h - 1) // SHARD_SIZE
        sc_start = col // SHARD_SIZE
        sc_end = (col + w - 1) // SHARD_SIZE

        for sr in range(sr_start, sr_end + 1):
            for sc in range(sc_start, sc_end + 1):
                shard_top = sr * SHARD_SIZE
                shard_left = sc * SHARD_SIZE

                # Tile-local region overlapping this shard
                t_row_start = max(0, shard_top - row)
                t_row_end = min(h, shard_top + SHARD_SIZE - row)
                t_col_start = max(0, shard_left - col)
                t_col_end = min(w, shard_left + SHARD_SIZE - col)

                # Shard-buffer region
                s_row_start = max(0, row - shard_top)
                s_row_end = s_row_start + (t_row_end - t_row_start)
                s_col_start = max(0, col - shard_left)
                s_col_end = s_col_start + (t_col_end - t_col_start)

                ov = ShardTileOverlap(
                    embedding_path=ti.embedding_path,
                    scales_path=ti.scales_path,
                    landmask_path=ti.landmask_path,
                    t_row_start=t_row_start,
                    t_row_end=t_row_end,
                    t_col_start=t_col_start,
                    t_col_end=t_col_end,
                    s_row_start=s_row_start,
                    s_row_end=s_row_end,
                    s_col_start=s_col_start,
                    s_col_end=s_col_end,
                )
                shard_map.setdefault((sr, sc), []).append(ov)

    # Convert to sorted list of ShardSpecs
    specs = []
    for (sr, sc), overlaps in sorted(shard_map.items()):
        specs.append(ShardSpec(
            sr=sr, sc=sc,
            row_px=sr * SHARD_SIZE,
            col_px=sc * SHARD_SIZE,
            tiles=overlaps,
        ))
    return specs
```

**Step 2: Write a unit test**

Create a small inline test to verify the index is built correctly with
synthetic tile data.  This tests the arithmetic, not file I/O.

```
uv run python -c "
from geotessera.zarr_zone import (
    build_shard_index, TileInfo, ZoneGrid, SHARD_SIZE,
)
from rasterio.transform import Affine

# One tile at pixel offset (100, 200), size 300x400
# Should span shard rows 0-1, shard cols 0-2
ti = TileInfo(
    lon=0.0, lat=0.0, year=2024, epsg=32630,
    transform=Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 50000.0),
    height=300, width=400,
    landmask_path='', embedding_path='', scales_path='',
)
grid = ZoneGrid(
    zone=30, year=2024, canonical_epsg=32630,
    origin_easting=0.0, origin_northing=51000.0,
    width_px=1024, height_px=1024,  # 4x4 shards
    pixel_size=10.0,
)
specs = build_shard_index([ti], grid)
# tile at row=100, col=200 with h=300, w=400
# covers pixels [100:400, 200:600]
# shard rows: 100//256=0 to 399//256=1 -> rows 0,1
# shard cols: 200//256=0 to 599//256=2 -> cols 0,1,2
shard_keys = [(s.sr, s.sc) for s in specs]
assert shard_keys == [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)], f'got {shard_keys}'

# Check shard (0,0): covers pixels [0:256, 0:256]
# Tile contributes [100:256, 200:256] in pixel space
# Tile-local: rows 0..156, cols 0..56
# Shard-local: rows 100..256, cols 200..256
s00 = specs[0]
ov = s00.tiles[0]
assert (ov.t_row_start, ov.t_row_end) == (0, 156), f'got {ov.t_row_start}..{ov.t_row_end}'
assert (ov.t_col_start, ov.t_col_end) == (0, 56), f'got {ov.t_col_start}..{ov.t_col_end}'
assert (ov.s_row_start, ov.s_row_end) == (100, 256), f'got {ov.s_row_start}..{ov.s_row_end}'
assert (ov.s_col_start, ov.s_col_end) == (200, 256), f'got {ov.s_col_start}..{ov.s_col_end}'
print(f'OK: {len(specs)} shard specs')
"
```

Expected: `OK: 6 shard specs`

**Step 3: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: add build_shard_index for shard-first tile lookup"
```

---

### Task 3: Implement `_load_landmask_slice()`

**Files:**
- Modify: `geotessera/zarr_zone.py` — add near `apply_landmask_to_scales` (around line 597)

**Step 1: Write the function**

```python
def _load_landmask_slice(
    landmask_path: str,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    """Load a sub-region of a landmask GeoTIFF using a rasterio window.

    Returns a 2D uint8 array where 0 = water.  If the landmask cannot be
    read (missing file, shape mismatch, etc.) returns all-ones (all land)
    so that no pixels are masked.
    """
    import rasterio
    from rasterio.windows import Window

    try:
        with rasterio.open(landmask_path) as src:
            window = Window.from_slices(
                (row_start, row_end), (col_start, col_end),
            )
            return src.read(1, window=window)
    except Exception as e:
        logger.warning(f"Failed to read landmask slice from {landmask_path}: {e}")
        return np.ones((row_end - row_start, col_end - col_start), dtype=np.uint8)
```

**Step 2: Verify import**

Run: `uv run python -c "from geotessera.zarr_zone import _load_landmask_slice; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: add _load_landmask_slice for windowed landmask reads"
```

---

### Task 4: Implement `_write_one_shard()`

**Files:**
- Modify: `geotessera/zarr_zone.py` — add after `_load_landmask_slice`

**Step 1: Write the function**

```python
def _write_one_shard(
    spec: ShardSpec,
    store: "zarr.Group",
) -> None:
    """Assemble and write a single shard — one self-contained unit of work.

    Mmap-reads tile embeddings (kernel page cache handles sharing between
    workers), loads scales/landmask slices, applies masking on the small
    shard-sized region, and writes once to the zarr store.
    """
    emb_buf = np.zeros((SHARD_SIZE, SHARD_SIZE, N_BANDS), dtype=np.int8)
    scales_buf = np.full((SHARD_SIZE, SHARD_SIZE), np.float32("nan"))

    for ov in spec.tiles:
        # Embedding: mmap read-only, slice out shard region (page-cache shared)
        emb = np.load(ov.embedding_path, mmap_mode="r")
        emb_buf[ov.s_row_start : ov.s_row_end, ov.s_col_start : ov.s_col_end, :] = (
            emb[ov.t_row_start : ov.t_row_end, ov.t_col_start : ov.t_col_end, :]
        )

        # Scales: mmap + copy the small slice so we can mutate it
        scales_mmap = np.load(ov.scales_path, mmap_mode="r")
        s = scales_mmap[
            ov.t_row_start : ov.t_row_end, ov.t_col_start : ov.t_col_end
        ].copy()

        # Landmask: windowed read for just this region
        lm = _load_landmask_slice(
            ov.landmask_path,
            ov.t_row_start, ov.t_row_end,
            ov.t_col_start, ov.t_col_end,
        )
        s[lm == 0] = np.float32("nan")
        s[~np.isfinite(s)] = np.float32("nan")

        scales_buf[ov.s_row_start : ov.s_row_end, ov.s_col_start : ov.s_col_end] = s

    # Single zarr write per array — one shard, one pass
    r, c = spec.row_px, spec.col_px
    store["embeddings"][r : r + SHARD_SIZE, c : c + SHARD_SIZE, :] = emb_buf
    store["scales"][r : r + SHARD_SIZE, c : c + SHARD_SIZE] = scales_buf
```

**Step 2: Verify import**

Run: `uv run python -c "from geotessera.zarr_zone import _write_one_shard; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: add _write_one_shard for self-contained parallel shard writes"
```

---

### Task 5: Wire into `build_zone_stores` and replace `_write_tiles_batched`

**Files:**
- Modify: `geotessera/zarr_zone.py` — the `build_zone_stores` function

**Step 1: Replace the `_write_tiles_batched` call in `build_zone_stores`**

Find the current block (around line 879-883):
```python
        # Write tiles batched by shard — each shard written exactly once
        tiles_written, errors = _write_tiles_batched(
            store, tile_infos, zone_grid,
            workers=workers, console=console,
        )
```

Replace with:
```python
        # Build shard index: precompute which tiles overlap each shard
        shard_specs = build_shard_index(tile_infos, zone_grid)
        if console is not None:
            n_total = zone_grid.height_px // SHARD_SIZE * (zone_grid.width_px // SHARD_SIZE)
            console.print(
                f"  {len(shard_specs):,} non-empty shards "
                f"(of {n_total:,} total)"
            )

        # Write shards in parallel — each shard is independent
        results = _run_parallel(
            lambda spec: _write_one_shard(spec, store),
            shard_specs, workers, console,
            label="Writing shards",
        )
        tiles_written = len(tile_infos)
        errors = len(shard_specs) - len(results)
```

**Step 2: Remove the unused `from rich.progress import ...` block**

The `build_zone_stores` function has an import of `Progress, SpinnerColumn,
BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn` (around line
852-855) that was only used by the old tile-writing progress bar.
`_run_parallel` handles its own progress imports.  Remove these lines:

```python
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        MofNCompleteColumn, TimeElapsedColumn,
    )
```

**Step 3: Verify import and basic functionality**

Run: `uv run python -c "from geotessera.zarr_zone import build_zone_stores; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "feat: wire shard-first parallel writes into build_zone_stores"
```

---

### Task 6: Run the cram tests

**Step 1: Run all tests**

Run: `cd tests && uv run cram *.t`

The `zarr.t` test runs `geotessera-registry zarr-build` on 4 Cambridge
tiles — this exercises the full shard-first pipeline end-to-end.

Expected: all tests pass.  If a test fails, investigate and fix before
proceeding.

**Step 2: Commit any test fixes**

If fixes were needed:
```
git add -u
git commit -m "fix: adjust for shard-first pipeline"
```

---

### Task 7: Clean up dead code

**Files:**
- Modify: `geotessera/zarr_zone.py`

**Step 1: Remove `_write_tiles_batched`**

The entire `_write_tiles_batched` function (currently ~155 lines) is no
longer called.  Remove it.  Keep `write_tile_to_store` and
`_read_single_tile` — they are still used by the per-tile zarr export
path (individual tile .zarr archives in the `download --format zarr` flow)
and tests.

**Step 2: Verify nothing imports the removed function**

Run: `uv run python -c "import geotessera.zarr_zone; print('OK')"`
Expected: `OK`

Run: `grep -r '_write_tiles_batched' geotessera/`
Expected: no matches

**Step 3: Run tests again**

Run: `cd tests && uv run cram *.t`
Expected: all pass

**Step 4: Commit**

```
git add geotessera/zarr_zone.py
git commit -m "refactor: remove _write_tiles_batched (replaced by shard-first pipeline)"
```
