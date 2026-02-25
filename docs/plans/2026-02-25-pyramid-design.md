# UTM-native Pyramids for RGB/PCA Previews

## Problem

Current Zarr stores have single-resolution RGB and PCA preview arrays at 10m/px.
Browser clients must fetch full-resolution chunks even when zoomed out, wasting
bandwidth and making overview rendering slow for large UTM zones (~60k × 600k px).

## Design

Build multi-resolution pyramids of the existing `rgb` and `pca_rgb` RGBA arrays
using iterative 2× mean coarsening. Pyramids stay in UTM projection (no
reprojection to Web Mercator). The client selects the appropriate level based on
zoom and reprojects per-chunk as it does today.

### Store Layout

```
utm30_2025.zarr/
├── embeddings/          # unchanged
├── scales/              # unchanged
├── rgb/                 # full-res RGBA uint8 (level 0)
├── pca_rgb/             # full-res RGBA uint8 (level 0)
├── rgb_pyramid/
│   ├── 1/              # 2× coarsened  (20m)
│   ├── 2/              # 4× coarsened  (40m)
│   ├── 3/              # 8× coarsened  (80m)
│   ├── 4/              # 16× coarsened (160m)
│   ├── 5/              # 32× coarsened (320m)
│   ├── 6/              # 64× coarsened (640m)
│   └── 7/              # 128× coarsened (1280m)
├── pca_rgb_pyramid/
│   ├── 1/ … 7/
├── easting/
├── northing/
└── band/
```

Level 0 is the existing `rgb`/`pca_rgb` array — no duplication. Levels 1–7 are
stored under `{name}_pyramid/{level}/` as RGBA uint8 arrays with the same
1024×1024×4 chunking.

### Coarsening Method

Each level halves both spatial dimensions by averaging 2×2 blocks using
`xarray.DataArray.coarsen(northing=2, easting=2).mean()`. Alpha channel is
averaged too, correctly handling partial-data boundaries where some pixels in a
block are transparent.

8 fixed levels total (level 0 = full-res, levels 1–7 stored as pyramid groups).

### Per-Level Metadata

Each pyramid group stores attrs:

```python
{
    "level": int,              # pyramid level (1–7)
    "pixel_size_m": float,     # 10 * 2^level
    "transform": [6 floats],   # adjusted affine for this resolution
    "shape": [height, width],  # dimensions at this level
}
```

### Root Store Metadata

Added to existing store attrs:

```python
{
    "has_rgb_pyramid": True,
    "rgb_pyramid_levels": 8,       # total including level 0
    "has_pca_pyramid": True,
    "pca_pyramid_levels": 8,
}
```

### CLI Integration

Two new flags on `geotessera-registry zarr-build`:

- `--pyramid` — build pyramids after writing previews (requires rgb/pca to exist)
- `--pyramid-only` — scan existing stores, build/rebuild pyramids from full-res
  preview arrays without regenerating the full-res data

### Dependencies

No new dependencies. Uses `xarray.coarsen` (xarray is already a dependency).
ndpyramid's value is mainly for reprojection which we don't need since pyramids
stay in UTM.

### Implementation Notes

- Read full-res array as xarray DataArray, iteratively coarsen and write each level
- Chunk size stays 1024×1024×4 at all levels (coarser levels have fewer chunks)
- Uncompressed storage (consistent with existing preview arrays)
- Progress reporting via Rich (one progress bar per store per preview type)
- `--pyramid-only` scans for existing `.zarr` stores and checks `has_rgb_preview`
  / `has_pca_preview` attrs to decide what to build
