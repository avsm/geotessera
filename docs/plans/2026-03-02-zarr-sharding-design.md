# Zarr v3 Sharding for Per-Pixel Embedding Lookups

**Date**: 2026-03-02
**Status**: Proposed

## Problem

The current zone stores use 1024x1024 uncompressed chunks. Reading a single
pixel's 128-band embedding requires fetching the entire chunk (~128KB for
embeddings, ~4MB for a full 1024x1024x128 read). This makes point queries
expensive, especially over HTTP range requests.

## Design

Switch to zarr v3 sharding: small inner chunks (4x4 pixels) inside
tile-aligned shards (256x256 pixels). This gives O(2KB) single-pixel reads
while keeping file counts and spatial batch reads reasonable.

### Store Layout

```
utm{zone:02d}_{year}.zarr/
    embeddings    # int8    (H, W, 128)  chunks=(4,4,128) shards=(256,256,128)
    scales        # float32 (H, W)       chunks=(4,4)     shards=(256,256)
    rgb           # uint8   (H, W, 4)    chunks=(4,4,4)   shards=(256,256,4)
    easting       # float64 (W,)         single chunk, no sharding
    northing      # float64 (H,)         single chunk, no sharding
    band          # int32   (128,)        single chunk, no sharding
```

### Inner Chunk: (4,4,128)

Benchmarked against alternatives:

| Inner chunk   | Index/shard | File size | Pixel read | Read amplification |
|---------------|-------------|-----------|------------|--------------------|
| (1,1,128)     | 1024 KB     | 4188 KB   | 948 µs     | 1x                 |
| **(4,4,128)** | **64 KB**   | **2590 KB** | **15 ms** | **16x**          |
| (8,8,128)     | 16 KB       | 2498 KB   | 13 ms      | 64x                |
| (16,16,128)   | 4 KB        | 2762 KB   | 5 ms       | 256x               |

(4,4,128) chosen for the balance: 64KB shard index (~2.5% overhead), good
zstd compression, and 2KB per point read (128 int8 values × 16 pixels).

### Shard: (256,256,128)

Aligned with tessera tile boundaries (256x256 pixels at 10m). One shard =
one tile's worth of data. Benefits:

- Tile writes are whole-shard writes (no read-modify-write)
- File count matches current chunk file count
- Natural unit for HTTP caching

### Compression

`BloscCodec(cname='zstd', clevel=3)` on inner chunks within shards.
Embeddings are high-entropy but zstd at level 3 handles the mixed
zero/non-zero data well (~2x compression on semi-sparse tiles).

### Grid Snapping

Zone grid dimensions must be multiples of 256 (shard size). The grid
computation in `compute_zone_grid()` will snap outward to 256-pixel
boundaries. This adds up to 2.55km padding per edge (256 × 10m).

## Access Patterns

### Single-pixel lookup (optimised case)
1. Compute shard index from (row, col) → shard (row//256, col//256)
2. Read shard index (64KB, cached after first access)
3. Seek to inner chunk at (row%256//4, col%256//4) → decompress 2KB
4. Extract pixel from 4x4x128 block

### Tile read (256x256 region)
1. One shard covers the full tile
2. Decompress 4096 inner chunks (contiguous in shard file)
3. Assemble into (256,256,128) array

### Large region read
1. May span multiple shards
2. Each shard decompresses only the needed inner chunks
3. Zarr handles this transparently

## Write Strategy: Shard-Buffered Writes

To avoid slow per-pixel or per-tile partial shard writes, buffer tile
data in memory and write each shard as a single complete unit.

### Current flow
```
for tile in tiles:
    read tile → write directly to store[row:row+h, col:col+w, :]
```
Each tile write may touch 1-4 shards (if tile straddles shard boundaries),
causing read-modify-write on partial shards.

### New flow
```
shard_buffers = {}  # keyed by (shard_row, shard_col)
for tile in tiles:
    read tile → scatter into shard buffer(s)
flush all shard buffers → write each as one contiguous shard
```

Since tiles are 256x256 and shards are 256x256, most tiles map to exactly
one shard. Tiles that straddle shard boundaries (at zone grid edges where
tile origins don't align with shard grid) write to 1-4 buffers. Each
buffer is flushed once all contributing tiles are written.

**Memory**: Each shard buffer is 256×256×128 int8 = 8MB for embeddings,
256×256×4 = 256KB for scales. With sequential tile processing, at most
~4 shard buffers are active = ~33MB peak.

Alternatively, since shard boundaries align with tile boundaries in most
cases, we can simply sort tiles by shard position and write each tile as a
full shard slice directly.

## Code Changes Required

### Constants
- `SHARD_SIZE = 256` — shard spatial dimension
- `INNER_CHUNK = 4` — inner chunk spatial dimension

### `create_zone_store()`
- Add `chunks=` and `shards=` parameters to array creation
- Add `compressors=BloscCodec(cname='zstd', clevel=3)`
- Snap grid dimensions to 256-pixel multiples

### `compute_zone_grid()`
- Snap `width_px` and `height_px` up to nearest multiple of 256
  (SHARD_SIZE)

### `write_tile_to_store()` / tile writing
- Buffer writes to shard alignment, flush complete shards
- Or: write 256x256-aligned tile data directly (if tile and shard align)

### Chunk iteration code
- `_sample_chunk_stats()`, `write_preview_pass()`: iterate over
  shard-aligned boundaries (256x256) instead of `emb_arr.chunks[:2]`
  (which now returns inner chunk size 4x4)
- Use `SHARD_SIZE` constant for iteration

### `_reproject_chunk()` and global preview
- Reads from `src_arr[r_min:r_max, ...]` — no change needed, zarr
  handles sharded reads transparently

### `Tile._load_from_zone_zarr()` (tiles.py)
- Reads `store["embeddings"][r:r+h, c:c+w, :]` — no change needed,
  zarr handles sharded reads transparently

## Warnings

1. **Write speed**: Shard-buffered writes should be comparable to current
   uncompressed writes since each shard is written once as a complete unit.
   Compression adds some CPU cost but zstd level 3 is fast.
2. **No backwards compatibility**: Old stores must be regenerated.
3. **Partial edge shards**: Zone grid edges may not be fully populated.
   fill_value handles this (int8 0 for embeddings, NaN for scales).
