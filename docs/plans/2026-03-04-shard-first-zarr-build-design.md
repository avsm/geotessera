# Shard-first parallel zarr-build

**Date**: 2026-03-04
**Status**: Design

## Problem

The current zarr-build pipeline iterates tile-first: for each tile, read it
from NFS, scatter its data across ~20 shard buffers, then flush completed
shard rows.  This causes:

1. **NFS thrashing** — parallel tile reads open many large .npy files
   concurrently on a network mount, saturating NFS with random I/O.
2. **Coordination overhead** — tracking which shard rows are "complete"
   requires drain/flush logic that blocks the main thread.
3. **Unnecessary mutation** — the full 1100×1100×128 embedding array is
   loaded and mutated (zeroing no-data pixels) even though every downstream
   reader already checks `isnan(scales)`.

## Design

Invert the loop: iterate **shard-first**.  Each shard is an independent unit
of work that reads only the tile slices it needs, assembles a complete buffer,
and writes once.

### Pipeline

```
gather_tile_infos()                    # unchanged — tile metadata, no I/O
  → compute_zone_grid()                # unchanged — zone pixel grid
  → build_shard_index()                # NEW: precompute tile→shard overlaps
  → create_zone_store()                # unchanged — zarr v3 store
  → _run_parallel(write_one_shard)     # NEW: embarrassingly parallel
  → [optional RGB preview passes]      # unchanged
```

### Key insight: skip embedding mutation

Every reader (JS viewer, `compute_rgb_chunk`, `_sample_chunk_stats`) already
guards on `isnan(scales)`.  The `embedding[np.isnan(scales)] = 0` mutation is
unnecessary.  This means:

- **Embeddings** (155MB per tile, int8) need zero mutations → `np.load(mmap_mode='r')`
- **Scales** (5MB per tile, float32) need landmask + inf→NaN → applied on the
  small shard-sized slice (~262KB), not the full tile
- **Landmask** — windowed rasterio read for just the shard slice

### Data structures

```python
@dataclass
class ShardTileOverlap:
    """One tile's contribution to one shard."""
    embedding_path: str
    scales_path: str
    landmask_path: str
    # Tile-local region to read
    t_row_start: int
    t_row_end: int
    t_col_start: int
    t_col_end: int
    # Shard-buffer region to write
    s_row_start: int
    s_row_end: int
    s_col_start: int
    s_col_end: int

@dataclass
class ShardSpec:
    """Everything a worker needs to write one shard."""
    sr: int                           # shard row index
    sc: int                           # shard col index
    row_px: int                       # = sr * SHARD_SIZE
    col_px: int                       # = sc * SHARD_SIZE
    tiles: List[ShardTileOverlap]     # 1–4 tiles typically
```

### Shard index construction (`build_shard_index`)

For each tile, use `tile_pixel_offset` to get (row, col) in the zone grid.
Compute which shards the tile overlaps (row÷256 .. (row+h-1)÷256, same for
cols).  For each overlapping shard, compute the exact tile-local and
shard-local slice coordinates.  Collect into `ShardSpec` objects.

Only shards with ≥1 tile overlap get a spec.  Empty shards are left as zarr
fill values (0 for embeddings, NaN for scales).

Pure arithmetic, no I/O.  O(n_tiles × shards_per_tile).

### Shard writer (`_write_one_shard`)

Each worker:

1. Allocate `emb_buf = zeros(256, 256, 128, int8)` and
   `scales_buf = full(256, 256, NaN, float32)`
2. For each overlapping tile:
   a. `np.load(embedding_path, mmap_mode='r')` → slice out shard region
   b. `np.load(scales_path, mmap_mode='r')` → slice + `.copy()`
   c. Windowed rasterio read of landmask for same region
   d. Apply `scales[landmask == 0] = NaN; scales[~isfinite] = NaN`
   e. Composite into buffers
3. Single zarr write: `store["embeddings"][r:r+256, c:c+256, :] = emb_buf`
4. Single zarr write: `store["scales"][r:r+256, c:c+256] = scales_buf`

### Parallelism

Use the existing `_run_parallel` helper (ThreadPoolExecutor + as_completed +
Rich progress).  Each shard is one work item.  No shared mutable state
between workers — the zarr store handles concurrent writes to different
shards, and mmap'd tile reads share kernel page cache.

### NFS behaviour

Adjacent shards (e.g. sr=3,sc=16 and sr=3,sc=17) often overlap the same
tile files.  The first worker to access a tile faults its pages into the
kernel buffer cache from NFS.  Subsequent workers for adjacent shards hit
page cache — no NFS round-trip.  The kernel evicts cold pages under memory
pressure.

### Memory budget

Per worker: 256×256×128 (8MB emb) + 256×256×4 (0.3MB scales) + mmap
overhead ≈ **~9MB per thread**.  With 24 workers: ~216MB.  Mmap'd tile
pages are managed by the kernel from the 256GB+ available.

### Progress bars

| Phase              | Label                  | Total              |
|--------------------|------------------------|--------------------|
| Shard index        | "Building shard index" | n_tiles            |
| Writing shards     | "Writing shards"       | n_non_empty_shards |
| RGB stretch        | "Computing stretch"    | (existing)         |
| RGB preview        | "Writing RGB preview"  | (existing)         |

### What changes

- **New**: `ShardSpec`, `ShardTileOverlap`, `build_shard_index()`,
  `_write_one_shard()`, `_load_landmask_slice()`
- **Replaced**: `_write_tiles_batched` → `_run_parallel(write_one_shard)`
  call in `build_zone_stores`
- **Kept**: `_read_single_tile` and `write_tile_to_store` remain for
  backward compat / tests but are not used in the shard path
- **Unchanged**: `gather_tile_infos`, `compute_zone_grid`,
  `create_zone_store`, RGB preview pipeline, CLI, store format

### Risks

- **rasterio thread safety**: windowed reads from the same GeoTIFF in
  multiple threads.  Mitigated: each thread opens its own file handle.
- **NFS file handle limits**: many threads opening the same .npy files.
  Mitigated: mmap uses a single kernel-level mapping; Python `np.load`
  opens and closes the fd during load.
- **zarr concurrent shard writes**: different shards write to different
  files, so no contention at the filesystem level.
