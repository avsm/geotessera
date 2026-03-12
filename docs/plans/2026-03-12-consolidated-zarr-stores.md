# Consolidated Per-Year Zarr Stores

**Date**: 2026-03-12
**Status**: Approved

## Summary

Consolidate the per-zone standalone Zarr stores (`utm29_2024.zarr`, `utm30_2024.zarr`, ...)
and the global preview store (`global_rgb_2024.zarr`) into a single Zarr v3 store per year
(`2024.zarr`), with each zone and the global preview as nested groups. Add a
`tessera:dataset_version` attribute to the store root for version tracking via STAC.

## Motivation

- **Single entry point**: One store per year instead of N+1 stores. One consolidated metadata
  fetch gives clients the full zone catalog.
- **Version in path**: `zarr/v1/2024.zarr` makes version explicit. The root attributes are
  self-describing (`tessera:dataset_version: "v1"`) even if the store is moved.
- **Simpler cloud hosting**: One base URL per year-store. Zone access is just a group path.
- **No performance impact**: Sharding is per-array, so each zone's data is still independently
  chunked. Zarr v3 writes are per-chunk, so zone builds and global preview builds don't contend.

## Store Layout

### Before

```
zarr/v1/
  utm29_2024.zarr/          # standalone store per zone
    zarr.json               # root group attrs: proj:code, spatial:transform, ...
    embeddings/
    scales/
    rgb/
  utm30_2024.zarr/
  utm31_2024.zarr/
  global_rgb_2024.zarr/     # standalone global preview store
    zarr.json               # root attrs: multiscales, spatial:bbox, proj:code
    0/rgb/
    1/rgb/
    ...
```

### After

```
zarr/v1/
  2024.zarr/                # one store per year
    zarr.json               # root group: tessera:dataset_version, year, zarr_conventions
    utm29/
      zarr.json             # zone group: proj:code, spatial:transform, spatial:shape, ...
      embeddings/
      scales/
      rgb/
    utm30/
      zarr.json
      embeddings/
      scales/
      rgb/
    global_rgb/
      zarr.json             # global group: multiscales, spatial:bbox, proj:code
      0/rgb/
      1/rgb/
      ...
```

### Root Group Attributes

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "tessera:dataset_version": "v1",
    "year": 2024,
    "zarr_conventions": [
      {"uuid": "f17cb550-...", "name": "proj:"},
      {"uuid": "689b58e2-...", "name": "spatial:"},
      {"uuid": "d35379db-...", "name": "multiscales"}
    ]
  }
}
```

Zone group and global_rgb group attributes remain unchanged from the current format.

## Migration Script

A one-off CLI command `zarr-consolidate` added to `registry_cli.py`:

```
geotessera-registry zarr-consolidate /path/to/zarr/v1 --year 2024 [--dry-run] [--version v1]
```

### Algorithm

1. Scan directory for `utm\d{2}_{year}.zarr` and `global_rgb_{year}.zarr`
2. Validate all expected stores exist
3. Create `{year}.zarr/` directory
4. Write root `zarr.json` with `tessera:dataset_version` and `year` attributes
5. `os.rename()` each `utm{NN}_{year}.zarr/` → `{year}.zarr/utm{NN}/`
6. `os.rename()` `global_rgb_{year}.zarr/` → `{year}.zarr/global_rgb/`
7. Remove stale zone completion markers (`.zone_*_done`)
8. `zarr.consolidate_metadata()` on the new store
9. Print summary

### Constraints

- `--dry-run` prints all operations without executing them
- `--version` defaults to `"v1"`
- Must be same filesystem (uses `os.rename`, not copy)
- No data is rewritten — only directory moves and new `zarr.json` files
- **Partial migration**: If `{year}.zarr/` already exists, the script checks each zone
  individually. A zone is skipped if `utm{NN}_{year}.zarr/` no longer exists AND
  `{year}.zarr/utm{NN}/` does exist. If both old and new paths exist for the same zone,
  the script errors (ambiguous state — manual resolution required). The root `zarr.json`
  is always (re)written to ensure attributes are up to date.

## Code Changes

### zarr_zone.py

| Function | Change |
|----------|--------|
| `_store_name()` | Replace with two helpers: `_year_store_name(year)` → `"{year}.zarr"` and `_zone_group_name(zone)` → `"utm{NN}"` |
| `create_zone_store()` | Open/create `{year}.zarr` root group, create `utm{NN}` subgroup, write zone attrs to subgroup |
| `build_global_preview()` | Open `{year}.zarr`, discover zones by listing `utm\d+` groups instead of scanning for dirs |
| `_ensure_global_store()` | Create `global_rgb` group within `{year}.zarr` instead of `global_rgb_{year}.zarr`. **Safety**: the shape-mismatch `shutil.rmtree` must target only the `global_rgb` group, not the parent store |
| `add_rgb_to_existing_store()` | Receives year store path + zone group name instead of standalone store path |
| `open_zone_store()` | Accept `(store_path, zone_group)` or a combined path |
| `read_region_from_zone()` | Open year store, access zone group |
| `_init_reproj_worker()` | Open year store, resolve zone group for source RGB |
| Zone completion markers | Move to `{year}.zarr/.zone_{NN}_done` |
| Module docstring (line 8) | Update `utm{zone:02d}_{year}.zarr/` layout diagram |

### tiles.py

| Function | Change |
|----------|--------|
| `Tile.from_zone_zarr()` | Accept `store_path` + `zone_group` instead of standalone store path |
| `_load_from_zone_zarr()` | `zarr.open_group(store_path, path=zone_group)` |

### registry_cli.py

| Function | Change |
|----------|--------|
| `stac_index_command()` | Scan for `\d{4}.zarr` stores, iterate zone groups within each |
| `_zarr_store_to_stac_item()` | Receives zone group (not standalone store). Constructs item ID as `utm{NN}_{year}` (preserving existing format). Reads `tessera:dataset_version` from store root attrs; warns if mismatched with directory version; includes version and `year` in STAC item properties |
| `_store_bbox_wgs84()` | Receives zone group reference instead of standalone store path |
| `_store_matches()` | Update `--rgb-only` zone discovery to enumerate groups within year store |
| `zarr_build_command()` | Adapt output path logic |
| `global_preview_command()` | No change (delegates to `build_global_preview`) |
| CLI help strings | Update references to `utm*_YYYY.zarr` patterns |
| New: `zarr_consolidate_command()` | Migration CLI |

### scripts/patch_global_bounds.py

| Function | Change |
|----------|--------|
| `compute_zone_bounds_full_precision()` | Open year store, iterate zone groups |
| Global store discovery | `global_rgb` group within year store |

## STAC Catalog Changes

### Item IDs

STAC item IDs are preserved as `utm{NN}_{year}` (e.g. `utm29_2024`). The ID is
constructed synthetically from zone group name + year attribute, not derived from
the store path. This preserves backwards compatibility with tze viewer code that
extracts the year from the item ID via regex `_(\d{4})$`.

### Asset URLs

Zone store assets change from relative paths to zone stores:

**Before**: `"href": "utm29_2024.zarr"`
**After**: `"href": "2024.zarr/utm29"`

### New Properties

- `tessera:dataset_version`: Read from store root attrs, included in each STAC item
- `year`: Already present in zone attrs, included in STAC item properties
- Version mismatch warning: If store's `tessera:dataset_version` differs from directory
  name (e.g., store says `v2` but lives under `v1/`), emit a warning during STAC generation

### Rebuild

A STAC catalog rebuild (`geotessera-registry stac-index`) is required after migration
to update all asset hrefs.

## tze Viewer Impact

### Zone Stores

The viewer discovers zone stores via STAC catalog (`loadCatalog()` in `stac.ts`). The
`zarrUrl` for each zone is constructed from the STAC item's asset href resolved against
the catalog base URL. **A STAC rebuild is sufficient** — no TypeScript code changes needed
for zone store access.

The zarr-reader opens the store at whatever URL the STAC catalog provides and resolves
`embeddings`, `scales`, `rgb` relative to that root. After consolidation, the STAC href
`2024.zarr/utm29` resolves to the zone group, which has the same internal layout.

### Global Preview

The global preview discovery in `stac.ts` (line 92) hardcodes the pattern
`global_rgb_{year}.zarr`. This single line must change:

**Before**: `const candidateUrl = \`\${baseUrl}global_rgb_\${year}.zarr\`;`
**After**: `const candidateUrl = \`\${baseUrl}\${year}.zarr/global_rgb\`;`

### Summary of tze Changes

- One line change in `apps/viewer/src/lib/stac.ts` for global preview URL
- STAC catalog rebuild for zone store URLs
- No changes to zarr-reader, zarr-source, zarr-tile-protocol, or source-manager
