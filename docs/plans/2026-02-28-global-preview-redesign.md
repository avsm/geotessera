# Global Preview Redesign

## Problem

The current `build_global_preview` generates corrupted output with missing
streaks of tiles. Root cause: multiple threads write to the same zarr chunk
concurrently via `ThreadPoolExecutor`. When two `_reproject_tile` tasks map
to overlapping 512x512 zarr chunks, one thread's partial write is
clobbered by the other's. The 512-pixel chunk boundaries create the
characteristic streak pattern.

Secondary problems:
- Monolithic: must rebuild from scratch for every run
- No per-year separation in the pipeline design
- Bounds computed from zone union, not fixed — adding zones changes array shape
- Pyramid coarsening reads/writes full width strips, wasting memory

## Design Goals

1. **Correct**: no concurrent writes to the same chunk, ever
2. **Incremental**: process one UTM zone at a time, updating an existing store
3. **Memory-bounded**: O(workers x chunk_size) peak memory during reprojection
4. **Parallel**: use dask for task parallelism, rasterio releases the GIL
5. **Per-year**: each year gets its own store with stable dimensions

## Fixed Global Grid

The store covers the full globe with a fixed pixel grid:

| Parameter | Value |
|-----------|-------|
| Bounds | (-180, -90, 180, 90) |
| Resolution | 0.0001 deg (~10m at equator) |
| Level 0 shape | 1,800,000 x 3,600,000 x 4 |
| Chunks | 512 x 512 x 4 |
| dtype | uint8 |
| fill_value | 0 (transparent black) |
| Pyramid levels | 7 (levels 0-6) |

Pixel coordinate system:
- Column j -> longitude: -180 + j * 0.0001
- Row i -> latitude: 90 - i * 0.0001 (north-up, row 0 = north pole)

The fixed grid means:
- Array shape never changes regardless of which zones are processed
- Each UTM zone maps to a deterministic column range
- Unwritten chunks are sparse (zarr v3 doesn't store all-zero chunks)
- Adding a new zone is just filling in a region, no rebuild needed

## Store Layout

```
global_rgb_{year}.zarr/
  zarr.json              # multiscales + spatial.bounds (always [-180,-90,180,90])
  0/
    rgb/ c/...           # 1,800,000 x 3,600,000 x 4
    band/ c/0            # [0,1,2,3] int32
  1/
    rgb/ c/...           # 900,000 x 1,800,000 x 4
    band/
  ...
  6/
    rgb/ c/...           # 28,125 x 56,250 x 4
    band/
```

Output path: always `{zarr_dir}/global_rgb_{year}.zarr`.

## Architecture

### Per-Zone Processing

Each zone is processed independently and sequentially:

```
for zone in zones:
    1. Read zone store metadata (attrs only)
    2. Compute zone's output region (chunk-aligned row/col slices)
    3. Reproject zone's RGB into level 0 of global store
    4. Update pyramid levels 1-6 for the affected region
    5. Re-consolidate metadata
```

Zones are processed sequentially to avoid any possibility of two zones
writing to the same chunk. Within each zone, reprojection of individual
chunks runs in parallel.

### Phase 1: Reprojection (Level 0)

For each zone:

1. **Compute output region**: From the zone's EPSG code and store metadata,
   determine which rows and columns of the global grid this zone covers.
   Snap to chunk boundaries (round start down, end up to nearest multiple
   of 512).

2. **Batch chunk tasks**: Divide the output region into batches of
   `BATCH_ROWS` chunk-rows (default 64). Each batch contains
   `BATCH_ROWS x n_chunk_cols` chunk tasks.

3. **Execute batch**: For each batch, use `dask.delayed` to create one
   task per output chunk. Each task:
   - Computes the chunk's WGS84 extent from its (row, col) position
   - Back-projects corner points to UTM to find the source window
   - Reads the source window from the UTM zarr store
   - Calls `rasterio.warp.reproject()` for that 512x512 tile
   - Writes the result directly to the global zarr array at the
     chunk's position

   `dask.compute(*batch_tasks, scheduler='threads', num_workers=N)`

4. **Memory per task**: ~6.5MB peak (source window + reproject buffers +
   output chunk). With N workers: N x 6.5MB. With default 4 workers: 26MB.

5. **Progress**: Report after each batch completes.

**Why each task writes directly to zarr** (rather than building a dask
array and calling `to_zarr`): avoids zarr v3 compatibility issues with
dask's zarr integration, and makes the write-per-chunk guarantee explicit.
Since tasks write to non-overlapping chunk-aligned positions, there are no
races.

### Phase 2: Pyramid Coarsening

After a zone's level 0 is complete, update pyramid levels 1-6 for the
affected region:

For each level L (1 through 6):
1. Determine which chunk-rows at level L are affected by this zone
2. Process in row strips: read 2 chunk-rows from level L-1,
   coarsen 2x2 -> 1 pixel using mean, write 1 chunk-row to level L
3. Parallelize strips with `dask.delayed` (independent rows)

Memory per strip: 2 x 512 x zone_width x 4 bytes. For a 6-degree zone
(60K pixels): ~234MB peak. Acceptable for a single strip.

At each successive level, the zone's width halves, so coarsening gets
progressively cheaper.

### Incremental Updates

The store is created once via `_ensure_global_store()` which:
1. Checks if the store exists and has the correct shape
2. If not, creates it with the fixed global dimensions at all levels
3. Writes multiscales metadata and band coordinate arrays

Subsequent runs for additional zones just write into the existing arrays.
Re-processing a zone overwrites its region (idempotent).

The metadata (`spatial.bounds`) is always the fixed global extent, so
it never needs recomputation.

## CLI Interface

```bash
# Process one zone (creates store if needed)
geotessera-registry global-preview /data/zarr/v0/ \
    --year 2025 --zones 30

# Add more zones incrementally
geotessera-registry global-preview /data/zarr/v0/ \
    --year 2025 --zones 31,32

# Process all available zones for a year
geotessera-registry global-preview /data/zarr/v0/ --year 2025

# Control parallelism
geotessera-registry global-preview /data/zarr/v0/ \
    --year 2025 --zones 30 --workers 8
```

The `--output` flag is removed. Output is always
`{zarr_dir}/global_rgb_{year}.zarr`.

## Function Signatures

```python
# Constants
GLOBAL_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
BASE_RES = 0.0001
LEVEL0_W = 3_600_000
LEVEL0_H = 1_800_000
CHUNK_SIZE = 512
NUM_BANDS = 4
NUM_LEVELS = 7
BATCH_CHUNK_ROWS = 64  # chunk-rows per dask batch


def build_global_preview(
    zarr_dir: Path,
    year: int,
    zones: Optional[List[int]] = None,
    num_levels: int = NUM_LEVELS,
    workers: int = 4,
    console: Optional["rich.console.Console"] = None,
) -> Path:
    """Create or update the global preview store for a given year."""


def _ensure_global_store(store_path: Path, num_levels: int) -> None:
    """Create the global store with fixed dimensions if it doesn't exist."""


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
    """Reproject one zone's RGB into level 0. Returns affected region
    as (row_start, row_end, col_start, col_end) in chunk coords."""


def _reproject_chunk(
    global_arr,  # zarr array handle
    chunk_row: int,
    chunk_col: int,
    src_arr,     # zarr array handle (source UTM store)
    src_epsg: int,
    src_pixel: float,
    src_origin_e: float,
    src_origin_n: float,
    src_h: int,
    src_w: int,
) -> bool:
    """Reproject one 512x512 output chunk. Writes directly to global_arr.
    Returns True if any data was written."""


def _coarsen_zone_pyramid(
    store_path: Path,
    row_start: int, row_end: int,
    col_start: int, col_end: int,
    num_levels: int,
    workers: int,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Update pyramid levels 1-N for the region affected by a zone."""
```

## Edge Cases

**Zone overlaps**: Adjacent UTM zones overlap by ~0.5 deg in longitude.
Both zones will write to the overlapping chunk-columns. Since zones are
processed sequentially, the later zone's data wins. This is acceptable
for preview imagery.

**Partial chunks at zone edges**: The zone region is snapped to chunk
boundaries, so every task writes a complete 512x512 chunk. Source data
that falls outside the zone produces zeros (transparent), which is
correct fill.

**Empty chunks**: Chunks over ocean or outside the zone's data coverage
contain all zeros. rasterio.warp.reproject produces NaN for out-of-bounds
pixels, which we convert to 0. zarr v3 skips writing all-zero chunks.

**Anti-meridian**: UTM zone 1 starts at 180W and zone 60 ends at 180E.
The fixed grid handles this naturally since -180 and +180 are the
grid edges.

## Memory Budget (worst case, 4 workers)

| Phase | Per-task | Peak total |
|-------|----------|------------|
| Reproject | ~6.5 MB | 26 MB |
| Coarsen strip | ~234 MB | 234 MB |
| Dask graph | ~1 KB/node | ~7.5 MB/batch |
| **Total peak** | | **~270 MB** |

## What Changes

### Deleted
- `_reproject_tile()` — replaced by chunk-aligned `_reproject_chunk()`
- `_write_global_store()` — replaced by `_reproject_zone()` + `_coarsen_zone_pyramid()`
- Temporary store + shutil.move logic — writes directly to output store
- Bounds union computation — replaced by fixed `GLOBAL_BOUNDS`

### New
- `_ensure_global_store()` — idempotent store creation
- `_reproject_chunk()` — chunk-aligned, self-contained reprojection
- `_coarsen_zone_pyramid()` — incremental pyramid update
- `BATCH_CHUNK_ROWS` constant — controls dask batch size

### Modified
- `build_global_preview()` — simplified API (no `--output`), incremental semantics
- `global_preview_command()` — updated CLI args
- CLI parser — remove `--output`, keep `--year`, `--zones`, `--workers`, `--levels`
