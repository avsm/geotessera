# GeoZarr Convention Compliance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all Zarr stores fully compliant with the three GeoZarr conventions (proj:, spatial:, multiscales) from ~/src/git/zarr-conventions, update all readers to use the new keys, and add a one-off migration command for existing stores.

**Architecture:** Three-phase approach: (1) update store creation to write convention-compliant metadata, (2) update all readers to use the new namespaced keys, (3) add a `zarr-migrate-attrs` CLI command that rewrites existing store metadata in-place. No backwards-compat shims — the migration command is the upgrade path.

**Tech Stack:** Python, zarr v3, pyproj, topozarr, numpy

---

## Constants & Convention Definitions

These constants will be used across all tasks:

```python
# Convention registration entries (from zarr-conventions specs)
PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}
SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate transformations and mappings",
}
MULTISCALES_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}
```

---

### Task 1: Add convention constants and helper to zarr_zone.py

**Files:**
- Modify: `geotessera/zarr_zone.py:40-55` (add after existing constants)

**Step 1: Add convention registration constants**

Add after `SHARD_SIZE`/`INNER_CHUNK` constants (line ~55):

```python
# GeoZarr convention registration entries
PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}
SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate transformations and mappings",
}
MULTISCALES_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}
```

**Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: add GeoZarr convention registration constants"
```

---

### Task 2: Update per-zone store creation to write convention-compliant attrs

**Files:**
- Modify: `geotessera/zarr_zone.py:485-498` (the `store.attrs.update(...)` block in `_create_zone_store`)

**Step 1: Replace the attrs block**

Replace lines 485-498:

```python
    store.attrs.update({
        "utm_zone": zone_grid.zone,
        "year": zone_grid.year,
        "crs_epsg": zone_grid.canonical_epsg,
        "crs_wkt": crs_wkt,
        "transform": [
            zone_grid.pixel_size, 0.0, zone_grid.origin_easting,
            0.0, -zone_grid.pixel_size, zone_grid.origin_northing,
        ],
        "pixel_size_m": zone_grid.pixel_size,
        "geotessera_version": geotessera_version,
        "tessera_dataset_version": dataset_version,
        "n_tiles": len(zone_grid.tiles),
    })
```

With:

```python
    # Compute spatial:bbox from the grid extent
    easting_min = zone_grid.origin_easting
    easting_max = zone_grid.origin_easting + zone_grid.width_px * zone_grid.pixel_size
    northing_max = zone_grid.origin_northing
    northing_min = zone_grid.origin_northing - zone_grid.height_px * zone_grid.pixel_size

    store.attrs.update({
        # Convention registration
        "zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
        # proj: convention
        "proj:code": f"EPSG:{zone_grid.canonical_epsg}",
        "proj:wkt2": crs_wkt,
        # spatial: convention
        "spatial:dimensions": ["northing", "easting"],
        "spatial:transform": [
            zone_grid.pixel_size, 0.0, zone_grid.origin_easting,
            0.0, -zone_grid.pixel_size, zone_grid.origin_northing,
        ],
        "spatial:shape": [zone_grid.height_px, zone_grid.width_px],
        "spatial:bbox": [easting_min, northing_min, easting_max, northing_max],
        "spatial:registration": "pixel",
        # Application-specific metadata (not part of conventions)
        "utm_zone": zone_grid.zone,
        "year": zone_grid.year,
        "pixel_size_m": zone_grid.pixel_size,
        "geotessera_version": geotessera_version,
        "tessera_dataset_version": dataset_version,
        "n_tiles": len(zone_grid.tiles),
    })
```

**Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: write GeoZarr proj: and spatial: convention attrs in zone stores"
```

---

### Task 3: Update global preview store to use proper convention attrs

**Files:**
- Modify: `geotessera/zarr_zone.py:1312-1321` (the metadata block in `_ensure_global_store`)

**Step 1: Replace the global store metadata block**

Replace lines 1312-1321:

```python
    from topozarr.metadata import create_multiscale_metadata
    actual_levels = len([k for k in root.keys() if k.isdigit()])
    ms_attrs = create_multiscale_metadata(actual_levels, "EPSG:4326", "mean")
    ms_attrs["multiscales"]["crs"] = "EPSG:4326"
    west, south, east, north = GLOBAL_BOUNDS
    ms_attrs["spatial"] = {
        "bounds": [west, south, east, north],
        "resolution": GLOBAL_BASE_RES,
    }
    root.attrs.update(ms_attrs)
```

With:

```python
    from topozarr.metadata import create_multiscale_metadata
    actual_levels = len([k for k in root.keys() if k.isdigit()])
    ms_attrs = create_multiscale_metadata(actual_levels, "EPSG:4326", "mean")
    # topozarr already sets zarr_conventions with multiscales + proj:, and sets proj:code
    # Add spatial: convention to the registration list
    if "zarr_conventions" in ms_attrs:
        ms_attrs["zarr_conventions"].append(SPATIAL_CONVENTION)
    else:
        ms_attrs["zarr_conventions"] = [
            MULTISCALES_CONVENTION, PROJ_CONVENTION, SPATIAL_CONVENTION,
        ]
    # Add spatial: convention attributes
    west, south, east, north = GLOBAL_BOUNDS
    ms_attrs["spatial:dimensions"] = ["lat", "lon"]
    ms_attrs["spatial:bbox"] = [west, south, east, north]
    # Inject spatial:shape and spatial:transform into each multiscale layout item
    h_lvl, w_lvl = GLOBAL_LEVEL0_H, GLOBAL_LEVEL0_W
    res = GLOBAL_BASE_RES
    for item in ms_attrs.get("multiscales", {}).get("layout", []):
        item["spatial:shape"] = [h_lvl, w_lvl]
        item["spatial:transform"] = [res, 0.0, west, 0.0, -res, north]
        h_lvl //= 2
        w_lvl //= 2
        res *= 2.0
    root.attrs.update(ms_attrs)
```

**Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "feat: write GeoZarr spatial: convention attrs in global preview store"
```

---

### Task 4: Update all readers in zarr_zone.py to use new keys

**Files:**
- Modify: `geotessera/zarr_zone.py` — five locations that read `attrs["transform"]` or `attrs["crs_epsg"]`

**Step 1: Update `_utm_array_to_xarray` (lines 1165-1167)**

Replace:
```python
    store_attrs = dict(store.attrs)
    transform = list(store_attrs["transform"])
    epsg = int(store_attrs["crs_epsg"])
```

With:
```python
    store_attrs = dict(store.attrs)
    transform = list(store_attrs["spatial:transform"])
    epsg = int(store_attrs["proj:code"].split(":")[1])
```

**Step 2: Update `read_region_from_zone` (lines 1224-1229)**

Replace:
```python
    store = zarr.open_group(str(path), mode="r")
    attrs = dict(store.attrs)

    transform = attrs["transform"]
    pixel_size = transform[0]
    origin_easting = transform[2]
    origin_northing = transform[5]
```

With:
```python
    store = zarr.open_group(str(path), mode="r")
    attrs = dict(store.attrs)

    transform = attrs["spatial:transform"]
    pixel_size = transform[0]
    origin_easting = transform[2]
    origin_northing = transform[5]
```

**Step 3: Update `build_global_preview` zone info reading (lines 2083-2088)**

Replace:
```python
        zone_infos[zone_num] = {
            "store_path": store_path,
            "epsg": int(attrs["crs_epsg"]),
            "transform": list(attrs["transform"]),
            "shape": store["rgb"].shape,
        }
```

With:
```python
        zone_infos[zone_num] = {
            "store_path": store_path,
            "epsg": int(attrs["proj:code"].split(":")[1]),
            "transform": list(attrs["spatial:transform"]),
            "shape": store["rgb"].shape,
        }
```

**Step 4: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "refactor: update zarr_zone readers to use convention-namespaced attribute keys"
```

---

### Task 5: Update readers in tiles.py

**Files:**
- Modify: `geotessera/tiles.py:279-284` (`_load_from_zone_zarr`)

**Step 1: Update attribute reads**

Replace:
```python
        store = zarr.open_group(str(self._zone_store_path), mode="r")
        attrs = dict(store.attrs)

        transform_list = attrs["transform"]
        pixel_size = transform_list[0]
        origin_easting = transform_list[2]
        origin_northing = transform_list[5]
```

With:
```python
        store = zarr.open_group(str(self._zone_store_path), mode="r")
        attrs = dict(store.attrs)

        transform_list = attrs["spatial:transform"]
        pixel_size = transform_list[0]
        origin_easting = transform_list[2]
        origin_northing = transform_list[5]
```

**Step 2: Commit**

```bash
git add geotessera/tiles.py
git commit -m "refactor: update tiles.py zone zarr reader to use spatial:transform key"
```

---

### Task 6: Update readers in registry_cli.py (STAC index)

**Files:**
- Modify: `geotessera/registry_cli.py:2711-2719` (`_store_bbox_wgs84`)
- Modify: `geotessera/registry_cli.py:2747` (`_zarr_store_to_stac_item` required attrs list)
- Modify: `geotessera/registry_cli.py:2792-2800` (STAC item properties)

**Step 1: Update `_store_bbox_wgs84`**

Replace:
```python
    utm_zone = attrs["utm_zone"]
    west = (utm_zone - 1) * 6 - 180
    east = utm_zone * 6 - 180

    transform = attrs["transform"]
    pixel_size = transform[0]
    origin_northing = transform[5]
    height_px = attrs["grid_height"]
    epsg = attrs["crs_epsg"]
```

With:
```python
    utm_zone = attrs["utm_zone"]
    west = (utm_zone - 1) * 6 - 180
    east = utm_zone * 6 - 180

    transform = attrs["spatial:transform"]
    pixel_size = transform[0]
    origin_northing = transform[5]
    height_px = attrs["grid_height"]
    epsg = int(attrs["proj:code"].split(":")[1])
```

**Step 2: Update required attrs check**

Replace:
```python
    required = ["utm_zone", "crs_epsg", "transform", "year"]
```

With:
```python
    required = ["utm_zone", "proj:code", "spatial:transform", "year"]
```

**Step 3: Update STAC item properties**

Replace:
```python
    properties = {
        "utm_zone": attrs["utm_zone"],
        "crs_epsg": attrs["crs_epsg"],
        "pixel_size_m": attrs.get("pixel_size_m", attrs["transform"][0]),
        "grid_width": attrs["grid_width"],
        "grid_height": attrs["grid_height"],
        "n_bands": n_bands,
        "has_rgb_preview": attrs.get("has_rgb_preview", False),
        "geotessera_version": attrs.get("geotessera_version", "unknown"),
    }
```

With:
```python
    properties = {
        "utm_zone": attrs["utm_zone"],
        "proj:code": attrs["proj:code"],
        "pixel_size_m": attrs.get("pixel_size_m", attrs["spatial:transform"][0]),
        "grid_width": attrs["grid_width"],
        "grid_height": attrs["grid_height"],
        "n_bands": n_bands,
        "has_rgb_preview": attrs.get("has_rgb_preview", False),
        "geotessera_version": attrs.get("geotessera_version", "unknown"),
    }
```

**Step 4: Commit**

```bash
git add geotessera/registry_cli.py
git commit -m "refactor: update STAC index to use convention-namespaced attribute keys"
```

---

### Task 7: Update scripts/patch_global_bounds.py

**Files:**
- Modify: `scripts/patch_global_bounds.py:66-67` (zone attr reads)
- Modify: `scripts/patch_global_bounds.py:248-254` (spatial bounds patching)

**Step 1: Update zone attr reads**

Replace:
```python
        epsg = int(attrs["crs_epsg"])
        transform = list(attrs["transform"])
```

With:
```python
        epsg = int(attrs["proj:code"].split(":")[1])
        transform = list(attrs["spatial:transform"])
```

**Step 2: Update spatial bounds patching**

Replace:
```python
        attrs = meta.setdefault("attributes", {})
        attrs["spatial"] = {
            "bounds": bounds,
            "resolution": base_res,
        }
```

With:
```python
        attrs = meta.setdefault("attributes", {})
        attrs["spatial:bbox"] = bounds
        # Remove legacy key if present
        attrs.pop("spatial", None)
```

**Step 3: Commit**

```bash
git add scripts/patch_global_bounds.py
git commit -m "refactor: update patch_global_bounds to use convention-namespaced keys"
```

---

### Task 8: Add `zarr-migrate-attrs` CLI command

This is the one-off migration command that scans a directory of existing .zarr stores and rewrites their metadata to be convention-compliant.

**Files:**
- Modify: `geotessera/zarr_zone.py` (add `migrate_store_attrs` function after the convention constants)
- Modify: `geotessera/registry_cli.py` (add `zarr-migrate-attrs` subcommand)

**Step 1: Add the migration function to zarr_zone.py**

Add after the convention constants (after Task 1's additions):

```python
def migrate_store_attrs(store_path: Path, dry_run: bool = False) -> dict:
    """Migrate a zone store's attrs from legacy keys to GeoZarr convention keys.

    Rewrites in-place. Idempotent — skips stores that already have convention keys.

    Returns a dict summarising what was changed (or would be changed if dry_run).
    """
    import zarr

    store = zarr.open_group(str(store_path), mode="r" if dry_run else "r+")
    attrs = dict(store.attrs)
    changes = {}

    # Skip if already migrated
    if "proj:code" in attrs and "spatial:transform" in attrs:
        return {"status": "already_migrated"}

    # --- proj: convention ---
    epsg = attrs.get("crs_epsg")
    crs_wkt = attrs.get("crs_wkt", "")
    if epsg is not None:
        changes["proj:code"] = f"EPSG:{epsg}"
    if crs_wkt:
        changes["proj:wkt2"] = crs_wkt
    elif epsg is not None:
        # Generate WKT2 from EPSG code
        try:
            from pyproj import CRS
            changes["proj:wkt2"] = CRS.from_epsg(epsg).to_wkt()
        except Exception:
            pass

    # --- spatial: convention ---
    transform = attrs.get("transform")
    if transform is not None:
        changes["spatial:transform"] = list(transform)
        changes["spatial:dimensions"] = ["northing", "easting"]
        changes["spatial:registration"] = "pixel"

        pixel_size = transform[0]
        origin_easting = transform[2]
        origin_northing = transform[5]

        # Compute shape from the scales array
        try:
            scales_shape = store["scales"].shape
            height_px, width_px = scales_shape[0], scales_shape[1]
            changes["spatial:shape"] = [height_px, width_px]
            # Compute bbox
            easting_min = origin_easting
            easting_max = origin_easting + width_px * pixel_size
            northing_max = origin_northing
            northing_min = origin_northing - height_px * pixel_size
            changes["spatial:bbox"] = [
                easting_min, northing_min, easting_max, northing_max,
            ]
        except Exception:
            pass

    # --- Convention registration ---
    changes["zarr_conventions"] = [PROJ_CONVENTION, SPATIAL_CONVENTION]

    # --- Remove legacy keys ---
    legacy_keys = ["crs_epsg", "crs_wkt", "transform"]

    if not dry_run:
        store.attrs.update(changes)
        for key in legacy_keys:
            if key in attrs:
                del store.attrs[key]

    changes["_removed"] = [k for k in legacy_keys if k in attrs]
    changes["status"] = "migrated"
    return changes
```

**Step 2: Add the CLI subcommand to registry_cli.py**

Add in `main()` alongside the other subparsers (after the stac-index parser block):

```python
    # zarr-migrate-attrs command
    migrate_parser = subparsers.add_parser(
        "zarr-migrate-attrs",
        help="Migrate existing Zarr store attributes to GeoZarr convention format",
    )
    migrate_parser.add_argument(
        "zarr_dir",
        help="Directory containing .zarr stores to migrate",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying stores",
    )
    migrate_parser.set_defaults(func=zarr_migrate_attrs_command)
```

Add the command function (before `main()`):

```python
def zarr_migrate_attrs_command(args):
    """Migrate existing Zarr store attrs to GeoZarr convention format."""
    from .zarr_zone import migrate_store_attrs

    zarr_dir = Path(args.zarr_dir)
    if not zarr_dir.is_dir():
        console.print(f"[red]Error: {zarr_dir} is not a directory[/red]")
        return 1

    # Find all .zarr stores (zone stores only, not global)
    stores = sorted(
        p for p in zarr_dir.iterdir()
        if p.is_dir() and p.name.endswith(".zarr") and not p.name.startswith("global_")
    )

    if not stores:
        console.print(f"[yellow]No .zarr stores found in {zarr_dir}[/yellow]")
        return 1

    action = "Would migrate" if args.dry_run else "Migrating"
    console.print(
        f"[bold]{action} {len(stores)} store(s) in {zarr_dir}[/bold]"
    )
    if args.dry_run:
        console.print("[dim](dry run — no changes will be made)[/dim]")

    migrated = 0
    skipped = 0
    for store_path in stores:
        result = migrate_store_attrs(store_path, dry_run=args.dry_run)
        status = result.pop("status")
        if status == "already_migrated":
            console.print(f"  {store_path.name}: [dim]already migrated[/dim]")
            skipped += 1
        else:
            removed = result.pop("_removed", [])
            console.print(f"  {store_path.name}: [green]{status}[/green]")
            for key, value in sorted(result.items()):
                if key == "zarr_conventions":
                    names = [c["name"] for c in value]
                    console.print(f"    + zarr_conventions: {names}")
                elif key == "proj:wkt2":
                    console.print(f"    + proj:wkt2: [dim](WKT2 string)[/dim]")
                else:
                    console.print(f"    + {key}: {value}")
            for key in removed:
                console.print(f"    - {key}")
            migrated += 1

    console.print(
        f"\n[bold]{'Would migrate' if args.dry_run else 'Migrated'}: "
        f"{migrated}, skipped: {skipped}[/bold]"
    )
    return 0
```

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py geotessera/registry_cli.py
git commit -m "feat: add zarr-migrate-attrs command for one-off legacy store migration"
```

---

### Task 9: Update the RGB preview attr writes

**Files:**
- Modify: `geotessera/zarr_zone.py:1126-1130` (single-zone RGB)
- Modify: `geotessera/zarr_zone.py:1982-1986` (parallel RGB)

These write `has_rgb_preview`, `rgb_bands`, `rgb_stretch` — these are application-specific, not part of any convention. No changes needed to these keys. But verify that the migration function doesn't accidentally remove them.

**Step 1: Verify — no code changes needed**

The migration function only removes `crs_epsg`, `crs_wkt`, `transform`. The RGB attrs are preserved. This task is a verification-only step.

**Step 2: Commit** — N/A (no changes)

---

### Task 10: Run tests and verify

**Step 1: Run existing cram tests**

```bash
cd tests && uv run cram *.t
```

Expected: all tests pass (cram tests don't exercise zarr store creation directly).

**Step 2: Verify import works**

```bash
uv run python -c "from geotessera.zarr_zone import PROJ_CONVENTION, SPATIAL_CONVENTION, MULTISCALES_CONVENTION, migrate_store_attrs; print('OK')"
```

**Step 3: Commit** — N/A (verification only)

---

## Attribute Key Mapping Reference

| Legacy Key | Convention Key | Notes |
|------------|---------------|-------|
| `crs_epsg` | `proj:code` | Format changes: `32630` → `"EPSG:32630"` |
| `crs_wkt` | `proj:wkt2` | Same value, just renamed |
| `transform` | `spatial:transform` | Same `[a, b, c, d, e, f]` format |
| _(new)_ | `spatial:dimensions` | `["northing", "easting"]` for zone stores |
| _(new)_ | `spatial:shape` | `[height_px, width_px]` |
| _(new)_ | `spatial:bbox` | `[easting_min, northing_min, easting_max, northing_max]` |
| _(new)_ | `spatial:registration` | `"pixel"` |
| _(new)_ | `zarr_conventions` | Array of convention registration objects |
| `utm_zone` | `utm_zone` | Kept as app-specific |
| `year` | `year` | Kept as app-specific |
| `pixel_size_m` | `pixel_size_m` | Kept as app-specific |
| `has_rgb_preview` | `has_rgb_preview` | Kept as app-specific |
| `spatial.bounds` | `spatial:bbox` | Global store: dict → flat key |
| `multiscales.crs` | _(removed)_ | Redundant with `proj:code` |

## Files Modified Summary

| File | What changes |
|------|-------------|
| `geotessera/zarr_zone.py` | Convention constants, store creation attrs, reader updates, migration function |
| `geotessera/tiles.py` | `_load_from_zone_zarr` reads `spatial:transform` |
| `geotessera/registry_cli.py` | STAC index reads new keys, new `zarr-migrate-attrs` command |
| `scripts/patch_global_bounds.py` | Reads/writes new keys |
