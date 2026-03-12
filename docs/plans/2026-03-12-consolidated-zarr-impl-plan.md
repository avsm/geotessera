# Consolidated Per-Year Zarr Store Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate per-zone standalone Zarr stores and global preview into a single per-year store with nested groups, and provide a migration script for existing data.

**Architecture:** Replace the flat `utm{NN}_{year}.zarr` layout with `{year}.zarr/utm{NN}/` groups. A one-off `zarr-consolidate` CLI command uses `os.rename` to restructure existing stores without rewriting data. All code that discovers/opens zone stores is updated to work with the nested layout.

**Tech Stack:** Python, zarr-python (v3), click/argparse CLI, cram tests

**Spec:** `docs/plans/2026-03-12-consolidated-zarr-stores.md`

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `geotessera/zarr_zone.py` | Store creation, reading, global preview pipeline | Modify |
| `geotessera/tiles.py` | Per-tile Zarr reading | Modify |
| `geotessera/registry_cli.py` | CLI commands, STAC generation, migration script | Modify |
| `scripts/patch_global_bounds.py` | Post-hoc bounds patching | Modify |
| `tests/zarr.t` | Zarr store cram tests | Modify |

## Key Notes

- **Variable naming**: Use `year_store_path` (Path) consistently for the year store directory.
- **Attribute naming**: Root group uses `tessera:dataset_version` (colon-namespaced). Zone groups keep existing `tessera_dataset_version` (underscore). These are intentionally different.
- **Year store creation**: The year store must be created once in `build_zone_stores` before the per-zone loop, not inside `create_zone_store`, to avoid TOCTOU races when zones are built concurrently.
- **rmtree safety**: Any `shutil.rmtree` call must target the zone group directory or `global_rgb` group, never the year store root.
- **Filesystem path access**: Functions like `_nonempty_shard_indices` and `_ensure_rgb_array` traverse zone directories via filesystem paths. After consolidation, the zone directory is `{year}.zarr/utm{NN}/` which is still a valid filesystem path, so these work without changes. Document this assumption.

---

## Chunk 1: Core Store Layout Changes

### Task 1: Replace `_store_name` with two helpers

**Files:**
- Modify: `geotessera/zarr_zone.py:519-520` (function definition)
- Modify: `geotessera/zarr_zone.py:533,943,975` (call sites)

- [ ] **Step 1: Replace `_store_name` with `_year_store_name` and `_zone_group_name`**

In `geotessera/zarr_zone.py`, replace lines 519-520:

```python
def _store_name(zone: int, year: int) -> str:
    return f"utm{zone:02d}_{year}.zarr"
```

With:

```python
def _year_store_name(year: int) -> str:
    """Return the Zarr store directory name for a given year."""
    return f"{year}.zarr"


def _zone_group_name(zone: int) -> str:
    """Return the group name for a UTM zone within a year store."""
    return f"utm{zone:02d}"
```

- [ ] **Step 2: Update call sites**

Find all uses of `_store_name(zone, year)` and replace with the two helpers. There are 3 call sites at lines 533, 943, and 975. Each needs context-specific changes — the store path uses `_year_store_name(year)` and the group path uses `_zone_group_name(zone)`.

At line 533 in `create_zone_store`:
```python
# Before
store_path = str(output_dir / _store_name(zone_grid.zone, year))
# After
year_store = output_dir / _year_store_name(year)
zone_group = _zone_group_name(zone_grid.zone)
```

At lines 943 and 975 in `build_zone_stores`, apply the same pattern — construct the year store path and zone group name separately.

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "refactor: replace _store_name with _year_store_name + _zone_group_name"
```

---

### Task 2: Update `create_zone_store` for nested groups

**Files:**
- Modify: `geotessera/zarr_zone.py:523-612` (`create_zone_store` function)

The current function creates a standalone store at `utm{NN}_{year}.zarr/`. It needs to:
1. Open or create the year store (`{year}.zarr/`)
2. Write root group attributes (`tessera:dataset_version`, `year`, `zarr_conventions`)
3. Create the zone group (`utm{NN}/`) within it
4. Write zone attributes to the zone group (not the root)

- [ ] **Step 1: Move year store creation to `build_zone_stores`**

The year store must be created once before the per-zone loop (not inside `create_zone_store`) to avoid TOCTOU races when zones are built concurrently. In `build_zone_stores` (around line 943), add:

```python
year_store_path = output_dir / _year_store_name(year)
_ensure_year_store(year_store_path, year, dataset_version="v1")
```

Add the helper function near `_year_store_name`:

```python
def _ensure_year_store(year_store_path: Path, year: int, dataset_version: str = "v1") -> None:
    """Create the year store root if it doesn't exist."""
    import json as _json
    os.makedirs(str(year_store_path), exist_ok=True)
    root_meta = year_store_path / "zarr.json"
    if not root_meta.exists():
        meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "tessera:dataset_version": dataset_version,
                "year": year,
                "zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
            },
        }
        with open(str(root_meta), "w") as f:
            _json.dump(meta, f, indent=2)
```

- [ ] **Step 2: Update `create_zone_store` to create zone group within year store**

At line 539, the current code is:
```python
store = zarr.open_group(store_path, mode="w", zarr_format=3)
```

Replace with zone group creation within the pre-existing year store:

```python
import json as _json

# Create zone group directory within year store
zone_dir = year_store_path / zone_group
os.makedirs(str(zone_dir), exist_ok=True)
zone_meta = zone_dir / "zarr.json"
with open(str(zone_meta), "w") as f:
    _json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)
# Open year store and get zone group handle
root = zarr.open_group(str(year_store_path), mode="r+", zarr_format=3)
store = root[zone_group]
```

Update the function signature to receive `year_store_path: Path` and `zone_group: str` instead of constructing them internally.

- [ ] **Step 3: Fix rmtree safety**

At line 535-537, the existing code does `shutil.rmtree(str(store_path))` to replace an existing store. After consolidation, this MUST target the zone group directory, not the year store:

```python
# Before (line 536) — DANGEROUS after consolidation
if Path(store_path).exists():
    shutil.rmtree(store_path)

# After — safe: only removes the zone group directory
zone_dir = year_store_path / zone_group
if zone_dir.exists():
    shutil.rmtree(str(zone_dir))
```

- [ ] **Step 2: Move zone attributes from root to zone group**

The attrs block at lines 590-612 currently writes to `store.attrs` (which was the root). After the change, `store` is the zone group, so the attrs write to the correct place without changes. Verify this is the case.

- [ ] **Step 3: Update array creation paths**

The array creation calls (lines 541-575) use paths like `"embeddings"`, `"scales"`, etc. Since `store` now points to the zone group, these become `{zone_group}/embeddings` etc. in the actual Zarr hierarchy. Verify the `store.create_array("embeddings", ...)` calls work correctly when `store` is a group object (they should — zarr creates arrays relative to the group).

- [ ] **Step 5: Update the return value and type annotation**

The function currently returns `Path(store_path)` with annotation `-> "zarr.Group"`. Update to return the year store path with correct annotation:

```python
def create_zone_store(...) -> Path:
    ...
    return year_store_path
```

Check all callers of `create_zone_store` — the return value is currently ignored at line 956, so this is safe.

- [ ] **Step 5: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: create_zone_store writes zones as nested groups in year store"
```

---

### Task 3: Update `_ensure_global_store` for nested layout

**Files:**
- Modify: `geotessera/zarr_zone.py:1364-1454` (`_ensure_global_store`)

The function currently creates a standalone `global_rgb_{year}.zarr` store. It needs to create a `global_rgb` group within the existing year store.

- [ ] **Step 1: Change function signature**

The function currently takes `store_path: Path` (path to the global store) and `num_levels: int`. Change to accept the year store path and create the `global_rgb` group within it:

```python
def _ensure_global_store(year_store_path: Path, num_levels: int) -> None:
```

- [ ] **Step 2: Fix the rmtree safety issue**

At line 1389, the current code does `shutil.rmtree(str(store_path))` when dimensions don't match. This would delete the entire year store. Change to only remove the `global_rgb` group:

```python
# Before (DANGEROUS after consolidation)
shutil.rmtree(str(store_path))

# After (safe — only removes global_rgb group)
global_rgb_dir = year_store_path / "global_rgb"
if global_rgb_dir.exists():
    shutil.rmtree(str(global_rgb_dir))
```

- [ ] **Step 3: Update store opening to use group path**

Replace standalone store creation with group creation within the year store:

```python
# Before
root = zarr.open_group(str(store_path), mode="w", zarr_format=3)

# After — create global_rgb group within existing year store
root = zarr.open_group(str(year_store_path), mode="r+", zarr_format=3)
```

Array creation calls like `root.create_array(f"{lvl}/rgb", ...)` need to be prefixed with `global_rgb/`:

```python
root.create_array(f"global_rgb/{lvl}/rgb", ...)
```

Or alternatively, get the group handle first:

```python
global_grp = root.require_group("global_rgb")
# Then create arrays relative to global_grp
```

- [ ] **Step 4: Update metadata consolidation call**

At line 1454: `zarr.consolidate_metadata(str(store_path))` — change to `zarr.consolidate_metadata(str(year_store_path))`.

- [ ] **Step 5: Update the caller in `build_global_preview`**

At line ~2308 (`output_path = zarr_dir / f"global_rgb_{year}.zarr"`), change to:

```python
year_store_path = zarr_dir / _year_store_name(year)
```

And pass `year_store_path` to `_ensure_global_store`.

- [ ] **Step 6: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('OK')"`

- [ ] **Step 7: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: _ensure_global_store creates global_rgb group within year store"
```

---

### Task 4: Update `build_global_preview` zone discovery

**Files:**
- Modify: `geotessera/zarr_zone.py:2243-2302` (zone discovery and metadata collection)

The function currently scans the directory for `utm\d{2}_{year}.zarr` directories. It needs to open the year store and list `utm\d+` groups.

- [ ] **Step 1: Replace filesystem scanning with group enumeration**

Replace lines 2243-2255 (the `for entry in sorted(zarr_dir.iterdir())` loop) with:

```python
year_store_path = zarr_dir / _year_store_name(year)
if not year_store_path.exists():
    if console is not None:
        console.print(f"  [red]Year store not found: {year_store_path}[/red]")
    return year_store_path

root = zarr.open_group(str(year_store_path), mode="r", zarr_format=3)
zone_pattern = re.compile(r"^utm(\d{2})$")
zone_stores: Dict[int, str] = {}  # zone_num -> group name

for name in sorted(root.keys()):
    m = zone_pattern.match(name)
    if m is None:
        continue
    zone_num = int(m.group(1))
    if zones is not None and zone_num not in zones:
        continue
    zone_stores[zone_num] = name
```

- [ ] **Step 2: Update zone metadata collection**

The section at lines ~2283-2302 currently opens each zone as a standalone store:
```python
store = zarr.open_group(str(store_path), mode="r")
```

Change to open the zone group within the year store:
```python
zone_grp = root[zone_name]  # zone_name is e.g. "utm29"
attrs = dict(zone_grp.attrs)
```

- [ ] **Step 3: Update RGB generation check**

Lines ~2269-2281 check `store.attrs.get("has_rgb_preview", False)`. Update to use the zone group handle.

Note: `_ensure_rgb_array`, `_run_rgb_generation_parallel`, and `_nonempty_shard_indices` traverse zone directories via filesystem paths (e.g., `store_path / "scales" / "c"`). After consolidation, the zone directory path `{year}.zarr/utm{NN}/` is still a valid filesystem path with the same internal structure, so these functions work without changes. Pass `year_store_path / zone_group_name` as their `store_path` argument.

- [ ] **Step 4: Update `_reproject_zone` call**

The call at line ~2344 passes `zone_store_path=zinfo["store_path"]`. Change to pass the year store path and zone group name:

```python
_reproject_zone(
    store_path=year_store_path,
    zone_num=zone_num,
    zone_group=zinfo["group_name"],
    ...
)
```

This requires updating `_reproject_zone`'s signature and `_init_reproj_worker` to open the zone group within the year store.

- [ ] **Step 5: Update zone completion markers**

The markers are at `store_path / f".zone_{zone_num}_done"`. Keep them at `year_store_path / f".zone_{zone_num}_done"`.

- [ ] **Step 6: Update `_coarsen_zone_pyramid` call**

The call passes `store_path=output_path`. Change to `store_path=year_store_path`. The coarsening function opens `{lvl}/rgb` arrays — these need to become `global_rgb/{lvl}/rgb`.

- [ ] **Step 7: Update final metadata consolidation**

Line ~2377: `zarr.consolidate_metadata(str(output_path))` — change to use `year_store_path`.

- [ ] **Step 8: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('OK')"`

- [ ] **Step 9: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: build_global_preview discovers zones as groups within year store"
```

---

### Task 5: Update reprojection and coarsening for nested layout

**Files:**
- Modify: `geotessera/zarr_zone.py:1463-1474` (`_init_reproj_worker`)
- Modify: `geotessera/zarr_zone.py:1686-1820` (`_reproject_zone`)
- Modify: `geotessera/zarr_zone.py:1823-1906` (`_coarsen_zone_pyramid`)

- [ ] **Step 1: Update `_init_reproj_worker` to open zone group**

At line 1470, the worker opens the zone store directly. Change to open the year store and resolve the zone group:

```python
def _init_reproj_worker(global_store_path: str, zone_group_path: str, zone_epsg: int):
    global _reproj_worker_global_arr, _reproj_worker_src_arr, _reproj_worker_to_utm
    import zarr
    from pyproj import Transformer

    global_store = zarr.open_group(global_store_path, mode="r+")
    _reproj_worker_global_arr = global_store["global_rgb/0/rgb"]
    zone_grp = zarr.open_group(global_store_path, mode="r", path=zone_group_path)
    _reproj_worker_src_arr = zone_grp["rgb"]
    _reproj_worker_to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{zone_epsg}", always_xy=True,
    )
```

Note: the global array path changes from `"0/rgb"` to `"global_rgb/0/rgb"`.

- [ ] **Step 2: Update `_reproject_zone` signature and initargs**

Add `zone_group: str` parameter. Update the `ProcessPoolExecutor` initargs:

```python
# Before
initargs=(str(store_path), str(zone_store_path), zone_epsg)

# After
initargs=(str(store_path), zone_group, zone_epsg)
```

- [ ] **Step 3: Update `_coarsen_zone_pyramid` array paths**

The function opens arrays at `f"{lvl - 1}/rgb"` and `f"{lvl}/rgb"`. Prefix with `global_rgb/`:

```python
prev_arr_path = f"global_rgb/{lvl - 1}/rgb"
cur_arr_path = f"global_rgb/{lvl}/rgb"
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: update reprojection and coarsening for nested store layout"
```

---

### Task 6: Update readers (`open_zone_store`, `read_region_from_zone`, `add_rgb_to_existing_store`)

**Files:**
- Modify: `geotessera/zarr_zone.py:1193-1198` (`add_rgb_to_existing_store`)
- Modify: `geotessera/zarr_zone.py:1316-1361` (`open_zone_store`, `read_region_from_zone`)

- [ ] **Step 1: Update `open_zone_store`**

```python
# Before (line 1316-1319)
def open_zone_store(path) -> "xarray.Dataset":
    import xarray as xr
    return xr.open_zarr(str(path))

# After
def open_zone_store(year_store_path, zone_group: str) -> "xarray.Dataset":
    import xarray as xr
    return xr.open_zarr(str(year_store_path), group=zone_group)
```

- [ ] **Step 2: Update `read_region_from_zone`**

Update the signature to accept year store path + zone group. Change the zarr.open_group call:

```python
# Before
store = zarr.open_group(str(path), mode="r")

# After
store = zarr.open_group(str(year_store_path), mode="r", path=zone_group)
```

- [ ] **Step 3: Update `add_rgb_to_existing_store`**

Update signature to accept year store path + zone group. Change the store opening:

```python
# Before
store = zarr.open_group(str(store_path), mode="r+")

# After
store = zarr.open_group(str(year_store_path), mode="r+", path=zone_group)
```

- [ ] **Step 4: Update all callers of these functions**

Search for call sites and update them to pass year store path + zone group.

- [ ] **Step 5: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: update zone store readers for nested group layout"
```

---

### Task 7: Update `tiles.py`

**Files:**
- Modify: `geotessera/tiles.py:240-307` (`from_zone_zarr`, `_load_from_zone_zarr`)

- [ ] **Step 1: Update `Tile.from_zone_zarr`**

Change the method to accept year store path + zone group:

```python
# Before
@classmethod
def from_zone_zarr(cls, zone_store_path, lon, lat, year):
    ...
    tile._zone_store_path = zone_store_path

# After
@classmethod
def from_zone_zarr(cls, year_store_path, zone_group, lon, lat, year):
    ...
    tile._year_store_path = year_store_path
    tile._zone_group = zone_group
```

- [ ] **Step 2: Update `_load_from_zone_zarr`**

Change the store opening:

```python
# Before
store = zarr.open_group(str(self._zone_store_path), mode="r")

# After
store = zarr.open_group(str(self._year_store_path), mode="r", path=self._zone_group)
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/tiles.py').read()); print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add geotessera/tiles.py
git commit -m "feat: update tile reader for nested zone group layout"
```

---

### Task 8: Update module docstring

**Files:**
- Modify: `geotessera/zarr_zone.py:1-16` (module docstring)

- [ ] **Step 1: Update store layout diagram**

Replace the current layout in the docstring:

```python
"""Zone-wide Zarr format for consolidated Tessera embeddings.

This module provides tools for building and reading Zarr v3 stores that
consolidate all tiles within a UTM zone into a single store per year.
This enables efficient spatial subsetting and cloud-native access.

Store layout (one store per year, zone groups within):
    {year}.zarr/
        zarr.json             # root: tessera:dataset_version, year
        utm{zone:02d}/        # one group per UTM zone
            embeddings        # int8    (H, W, 128)  chunks=(4,4,128)   shards=(256,256,128)
            scales            # float32 (H, W)       chunks=(4,4)       shards=(256,256)
            rgb               # uint8   (H, W, 4)    chunks=(4,4,4)     shards=(256,256,4)   [optional]
        global_rgb/           # global EPSG:4326 preview
            {level}/rgb       # uint8   (H, W, 4)    chunks=(512,512,4)

NaN in scales indicates no-data (water or no coverage).
Per-pixel inner chunks enable O(2KB) single-pixel lookups via HTTP range
requests.  Tile-aligned shards (256x256) keep file counts reasonable.
"""
```

- [ ] **Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "docs: update module docstring for consolidated store layout"
```

---

## Chunk 2: CLI and STAC Changes

### Task 9: Update `stac_index_command` and `_zarr_store_to_stac_item`

**Files:**
- Modify: `geotessera/registry_cli.py:2704-2851`

- [ ] **Step 1: Update `_zarr_store_to_stac_item` signature**

Change from receiving a standalone store path to receiving a zone group handle, zone name, year, and the store root attrs:

```python
def _zarr_store_to_stac_item(
    zone_grp,
    zone_name: str,
    year: int,
    root_attrs: dict,
    store_path: Path,
) -> "pystac.Item":
```

- [ ] **Step 2: Construct item ID synthetically**

Replace line 2772:
```python
# Before
item_id = store_path.name.removesuffix(".zarr")

# After
item_id = f"{zone_name}_{year}"
```

- [ ] **Step 3: Add version to STAC item properties**

Read `tessera:dataset_version` from `root_attrs` and include in STAC properties:

```python
props["tessera:dataset_version"] = root_attrs.get("tessera:dataset_version", "unknown")
```

- [ ] **Step 4: Add version mismatch warning**

After reading the version, compare with the directory path:

```python
store_version = root_attrs.get("tessera:dataset_version", "")
# Extract version from parent directory name (e.g., "v1" from "/path/to/v1/2024.zarr")
dir_version = store_path.parent.name if store_path.parent.name.startswith("v") else ""
if store_version and dir_version and store_version != dir_version:
    logger.warning(
        "Store version '%s' differs from directory '%s' for %s",
        store_version, dir_version, store_path,
    )
```

- [ ] **Step 5: Update asset href**

```python
# Before
assets={"zarr": pystac.Asset(href=store_path.name, ...)}

# After
assets={"zarr": pystac.Asset(href=f"{store_path.name}/{zone_name}", ...)}
```

- [ ] **Step 6: Update `stac_index_command` to scan year stores**

Replace the directory scanning at lines 2841-2846:

```python
# Before
store_pattern = re.compile(r"^utm(\d{2})_(\d{4})\.zarr$")
store_paths = sorted(
    p for p in zarr_dir.iterdir()
    if p.is_dir() and store_pattern.match(p.name)
)

# After
year_pattern = re.compile(r"^(\d{4})\.zarr$")
zone_pattern = re.compile(r"^utm(\d{2})$")

for entry in sorted(zarr_dir.iterdir()):
    if not entry.is_dir():
        continue
    m = year_pattern.match(entry.name)
    if m is None:
        continue
    year = int(m.group(1))
    root = zarr.open_group(str(entry), mode="r")
    root_attrs = dict(root.attrs)
    for zone_name in sorted(root.keys()):
        if not zone_pattern.match(zone_name):
            continue
        zone_grp = root[zone_name]
        item = _zarr_store_to_stac_item(
            zone_grp, zone_name, year, root_attrs, entry,
        )
        catalog.add_item(item)
```

- [ ] **Step 7: Verify `_store_bbox_wgs84` needs no changes**

Check the current signature at line 2704. It already receives attrs as a dict (not a store path), so no changes are needed. The spec's change table entry for this function is incorrect — skip this step and note in the spec.

- [ ] **Step 8: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/registry_cli.py').read()); print('OK')"`

- [ ] **Step 9: Commit**

```bash
git add geotessera/registry_cli.py
git commit -m "feat: update STAC generation for consolidated year stores"
```

---

### Task 10: Update `zarr_build_command` and `_store_matches`

**Files:**
- Modify: `geotessera/registry_cli.py:2526-2565`

- [ ] **Step 1: Update output path logic**

The command constructs output paths for zone stores. Update to create zones as groups within a year store:

```python
# The output_dir should now contain year stores
# build_zone_stores already updated in Task 2 to create nested groups
```

- [ ] **Step 2: Update `_store_matches` for `--rgb-only` mode**

The `_store_matches` function at lines 2551-2560 parses zone names from `utm30_2025.zarr`. Update to enumerate zone groups within year stores:

```python
# Scan for year stores and iterate their zone groups
year_pattern = re.compile(r"^(\d{4})\.zarr$")
zone_pattern = re.compile(r"^utm(\d{2})$")
for entry in sorted(Path(output_dir).iterdir()):
    m = year_pattern.match(entry.name)
    if m is None:
        continue
    root = zarr.open_group(str(entry), mode="r")
    for zone_name in sorted(root.keys()):
        if zone_pattern.match(zone_name):
            # Process this zone group
            ...
```

- [ ] **Step 3: Update CLI help strings**

Update help text for `zarr_dir` argument and any references to `utm*_YYYY.zarr` patterns.

- [ ] **Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/registry_cli.py').read()); print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add geotessera/registry_cli.py
git commit -m "feat: update zarr_build_command for consolidated store layout"
```

---

### Task 11: Update `scripts/patch_global_bounds.py`

**Files:**
- Modify: `scripts/patch_global_bounds.py:42-51,150`

- [ ] **Step 1: Update zone store discovery**

Replace the `utm\d{2}_{year}.zarr` directory scanning with year store group enumeration:

```python
year_store_path = base / f"{year}.zarr"
root = zarr.open_group(str(year_store_path), mode="r")
zone_pattern = re.compile(r"^utm(\d{2})$")
zone_stores = {}
for name in sorted(root.keys()):
    m = zone_pattern.match(name)
    if m:
        zone_num = int(m.group(1))
        if zone_filter is None or zone_num in zone_filter:
            zone_stores[zone_num] = name
```

- [ ] **Step 2: Update global store discovery**

Replace line 150 (`global_stores = sorted(base.glob("global_rgb_*.zarr"))`):

```python
# Global preview is now a group within the year store
global_grp = root["global_rgb"]
```

- [ ] **Step 3: Update attribute reads to use group handles**

Replace `zarr.open_group(str(store_path), mode="r")` calls with group handle access.

- [ ] **Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('scripts/patch_global_bounds.py').read()); print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add scripts/patch_global_bounds.py
git commit -m "feat: update patch_global_bounds for consolidated store layout"
```

---

## Chunk 3: Migration Script

### Task 12: Implement `zarr_consolidate_command`

**Files:**
- Modify: `geotessera/registry_cli.py` (add new command function and CLI subparser)

- [ ] **Step 1: Write the migration function**

Add `zarr_consolidate_command` function:

```python
def zarr_consolidate_command(args):
    """Consolidate per-zone Zarr stores into a single per-year store."""
    import json as _json
    import zarr

    base_dir = Path(args.zarr_dir)
    year = args.year
    version = args.version
    dry_run = args.dry_run

    year_store = base_dir / f"{year}.zarr"

    # 1. Discover zone stores
    zone_pattern = re.compile(rf"^utm(\d{{2}})_{year}\.zarr$")
    zone_dirs = {}
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = zone_pattern.match(entry.name)
        if m:
            zone_num = int(m.group(1))
            zone_dirs[zone_num] = entry

    # Discover global preview store
    global_dir = base_dir / f"global_rgb_{year}.zarr"
    has_global = global_dir.exists()

    if not zone_dirs and not has_global:
        console.print(f"[red]No stores found for year {year}[/red]")
        return 1

    console.print(f"[bold]Consolidating stores for year {year}[/bold]")
    console.print(f"  Version:  {version}")
    console.print(f"  Zones:    {sorted(zone_dirs.keys())}")
    console.print(f"  Global:   {'yes' if has_global else 'no'}")
    console.print(f"  Target:   {year_store}")
    if dry_run:
        console.print("  [yellow]DRY RUN — no changes will be made[/yellow]")

    # 2. Check for conflicts (zones and global preview)
    for zone_num, zone_dir in sorted(zone_dirs.items()):
        target = year_store / f"utm{zone_num:02d}"
        if zone_dir.exists() and target.exists():
            console.print(
                f"[red]Error: both {zone_dir.name} and "
                f"{year_store.name}/utm{zone_num:02d} exist — "
                f"ambiguous state, manual resolution required[/red]"
            )
            return 1
    if has_global and (year_store / "global_rgb").exists():
        console.print(
            f"[red]Error: both {global_dir.name} and "
            f"{year_store.name}/global_rgb exist — "
            f"ambiguous state, manual resolution required[/red]"
        )
        return 1

    # 3. Create year store root
    if not year_store.exists():
        console.print(f"  mkdir {year_store.name}/")
        if not dry_run:
            year_store.mkdir()

    # Write root zarr.json
    root_meta = year_store / "zarr.json"
    console.print(f"  write {year_store.name}/zarr.json")
    if not dry_run:
        from .zarr_zone import PROJ_CONVENTION, SPATIAL_CONVENTION, MULTISCALES_CONVENTION
        meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "tessera:dataset_version": version,
                "year": year,
                "zarr_conventions": [
                    PROJ_CONVENTION, SPATIAL_CONVENTION, MULTISCALES_CONVENTION,
                ],
            },
        }
        with open(str(root_meta), "w") as f:
            _json.dump(meta, f, indent=2)

    # 4. Move zone stores
    for zone_num, zone_dir in sorted(zone_dirs.items()):
        target = year_store / f"utm{zone_num:02d}"
        console.print(f"  rename {zone_dir.name}/ → {year_store.name}/{target.name}/")
        if not dry_run:
            os.rename(str(zone_dir), str(target))

    # 5. Move global preview
    if has_global:
        target = year_store / "global_rgb"
        console.print(f"  rename {global_dir.name}/ → {year_store.name}/global_rgb/")
        if not dry_run:
            os.rename(str(global_dir), str(target))

    # 6. Clean up zone completion markers (they live inside store dirs,
    #    moved along with the rename, now at year_store/.zone_*_done)
    if not dry_run:
        for marker in year_store.glob(".zone_*_done"):
            console.print(f"  remove {marker.name}")
            marker.unlink()

    # 7. Consolidate metadata
    if not dry_run:
        console.print("  consolidating metadata...")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata")
            zarr.consolidate_metadata(str(year_store))

    console.print(
        f"\n[bold green]Done: {len(zone_dirs)} zone(s)"
        f"{' + global preview' if has_global else ''}"
        f" → {year_store.name}[/bold green]"
    )
    return 0
```

- [ ] **Step 2: Add CLI subparser**

Add after the existing subparsers:

```python
consolidate_parser = subparsers.add_parser(
    "zarr-consolidate",
    help="Consolidate per-zone Zarr stores into a single per-year store",
)
consolidate_parser.add_argument(
    "zarr_dir",
    type=Path,
    help="Directory containing utm*_YYYY.zarr stores",
)
consolidate_parser.add_argument(
    "--year",
    type=int,
    required=True,
    help="Year to consolidate",
)
consolidate_parser.add_argument(
    "--version",
    type=str,
    default="v1",
    help="Dataset version string (default: v1)",
)
consolidate_parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print operations without executing them",
)
consolidate_parser.set_defaults(func=zarr_consolidate_command)
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('geotessera/registry_cli.py').read()); print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add geotessera/registry_cli.py
git commit -m "feat: add zarr-consolidate CLI command for store migration"
```

---

## Chunk 4: Tests and Verification

### Task 13: Update cram tests

**Files:**
- Modify: `tests/zarr.t`

- [ ] **Step 1: Read existing zarr.t test**

Read `tests/zarr.t` to understand the current test patterns. The tests likely create zone stores and verify their structure.

- [ ] **Step 2: Update store path expectations**

Any assertions about `utm{NN}_{year}.zarr` directory names need to change to `{year}.zarr/utm{NN}` group paths. Update expected output patterns.

- [ ] **Step 3: Add migration test**

Add a cram test that:
1. Creates a few zone stores in the old flat layout
2. Runs `zarr-consolidate --dry-run` and verifies output
3. Runs `zarr-consolidate` for real
4. Verifies the new layout exists
5. Verifies the old layout is gone

```
  $ geotessera-registry zarr-consolidate $TMPDIR --year 2024 --dry-run
  *Consolidating stores for year 2024* (glob)
  *DRY RUN* (glob)
  *rename utm29_2024.zarr* (glob)
```

- [ ] **Step 4: Run tests**

Run: `cd tests && uv run cram zarr.t`

- [ ] **Step 5: Commit**

```bash
git add tests/zarr.t
git commit -m "test: update zarr cram tests for consolidated store layout"
```

---

### Task 14: End-to-end verification

- [ ] **Step 1: Run full test suite**

Run: `cd tests && uv run cram *.t`

Fix any remaining failures.

- [ ] **Step 2: Verify syntax of all modified files**

```bash
python3 -c "import ast; ast.parse(open('geotessera/zarr_zone.py').read()); print('zarr_zone OK')"
python3 -c "import ast; ast.parse(open('geotessera/tiles.py').read()); print('tiles OK')"
python3 -c "import ast; ast.parse(open('geotessera/registry_cli.py').read()); print('cli OK')"
python3 -c "import ast; ast.parse(open('scripts/patch_global_bounds.py').read()); print('patch OK')"
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: fix any remaining issues from consolidated store migration"
```
