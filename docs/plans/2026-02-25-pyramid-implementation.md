# Pyramid Preview Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-resolution pyramid generation for RGB and PCA preview arrays in UTM-native Zarr stores, with `--pyramid` and `--pyramid-only` CLI flags.

**Architecture:** A new `build_pyramids_for_store()` function in `zarr_zone.py` reads the full-res `rgb` or `pca_rgb` array from an existing store, iteratively coarsens it 7 times (2× mean each), and writes each level as a sub-group. The CLI wires `--pyramid` and `--pyramid-only` flags following the same pattern as `--rgb-only`/`--pca-only`.

**Tech Stack:** xarray (coarsen), zarr (group creation), numpy (array ops), Rich (progress)

---

### Task 1: Core pyramid builder function

**Files:**
- Modify: `geotessera/zarr_zone.py` (add after `add_pca_to_existing_store`, ~line 1279)

**Step 1: Write the function `build_preview_pyramid`**

Add this function to `zarr_zone.py` after the `add_pca_to_existing_store` function (around line 1279):

```python
PYRAMID_LEVELS = 8  # total levels: 0 (full-res) + 1..7 (coarsened)


def build_preview_pyramid(
    store: "zarr.Group",
    preview_name: str,
    console: Optional["rich.console.Console"] = None,
) -> int:
    """Build a multi-resolution pyramid from an existing preview array.

    Reads store[preview_name] (full-res RGBA uint8) and creates
    store[f"{preview_name}_pyramid/{level}"] for levels 1..7,
    each halving both spatial dimensions via 2x2 mean coarsening.

    Args:
        store: Zarr group opened in r+ mode.
        preview_name: "rgb" or "pca_rgb".
        console: Optional Rich Console for progress.

    Returns:
        Number of pyramid levels written (7 on success).
    """
    import xarray as xr

    try:
        src_arr = store[preview_name]
    except KeyError:
        if console is not None:
            console.print(f"  [yellow]No {preview_name} array found, skipping pyramid[/yellow]")
        return 0

    h, w = src_arr.shape[:2]
    if h == 0 or w == 0:
        return 0

    # Read full-res into xarray for coarsening
    data = np.asarray(src_arr[:])
    da = xr.DataArray(
        data,
        dims=["northing", "easting", "rgba"],
    )

    # Get transform from store attrs for per-level metadata
    transform = store.attrs.get("transform", [10.0, 0.0, 0.0, 0.0, -10.0, 0.0])
    base_pixel_size = transform[0]
    origin_easting = transform[2]
    origin_northing = transform[5]

    group_name = f"{preview_name}_pyramid"

    # Delete existing pyramid group if present
    if group_name in store:
        import shutil
        group_path = Path(store.store.path) / group_name
        if group_path.exists():
            shutil.rmtree(group_path)

    pyramid_group = store.create_group(group_name)

    levels_written = 0
    current = da

    if console is not None:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(),
            console=console,
        )
    else:
        from contextlib import nullcontext
        progress_ctx = nullcontext()

    with progress_ctx as progress:
        if progress is not None:
            task = progress.add_task(
                f"Building {preview_name} pyramid", total=PYRAMID_LEVELS - 1,
            )

        for level in range(1, PYRAMID_LEVELS):
            # Coarsen 2x in both spatial dims; boundary="trim" drops remainder
            nh, nw = current.sizes["northing"], current.sizes["easting"]
            if nh < 2 or nw < 2:
                break  # too small to coarsen further

            coarsened = (
                current
                .coarsen(northing=2, easting=2, boundary="trim")
                .mean()
                .astype(np.uint8)
            )

            level_h, level_w = coarsened.sizes["northing"], coarsened.sizes["easting"]
            pixel_size = base_pixel_size * (2 ** level)

            # Create array in pyramid group
            level_arr = pyramid_group.create_array(
                str(level),
                shape=(level_h, level_w, 4),
                chunks=(min(1024, level_h), min(1024, level_w), 4),
                dtype=np.uint8,
                fill_value=np.uint8(0),
                compressors=None,
            )
            level_arr[:] = coarsened.values

            # Per-level metadata
            level_arr.attrs.update({
                "level": level,
                "pixel_size_m": pixel_size,
                "transform": [
                    pixel_size, 0.0, origin_easting,
                    0.0, -pixel_size, origin_northing,
                ],
                "shape": [level_h, level_w],
            })

            current = coarsened
            levels_written += 1

            if progress is not None:
                progress.advance(task)

    return levels_written
```

**Step 2: Verify module still imports cleanly**

Run: `uv run python -c "from geotessera.zarr_zone import build_preview_pyramid, PYRAMID_LEVELS; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "zarr: add build_preview_pyramid for multi-resolution previews"
```

---

### Task 2: Store-level pyramid builder

**Files:**
- Modify: `geotessera/zarr_zone.py` (add after `build_preview_pyramid`)

**Step 1: Write `add_pyramids_to_existing_store`**

Add this function immediately after `build_preview_pyramid`:

```python
def add_pyramids_to_existing_store(
    store_path: Path,
    console: Optional["rich.console.Console"] = None,
) -> None:
    """Build pyramids for all preview arrays in an existing Zarr store."""
    import zarr

    store = zarr.open_group(str(store_path), mode="r+")
    attrs = dict(store.attrs)

    for preview_name, attr_key in [("rgb", "has_rgb_preview"), ("pca_rgb", "has_pca_preview")]:
        if not attrs.get(attr_key, False):
            continue

        pyramid_attr = f"has_{preview_name}_pyramid"
        if attrs.get(pyramid_attr, False):
            if console is not None:
                console.print(f"  [dim]{preview_name} pyramid already exists, rebuilding[/dim]")

        if console is not None:
            console.print(f"  Building {preview_name} pyramid...")

        levels = build_preview_pyramid(store, preview_name, console=console)

        if levels > 0:
            store.attrs.update({
                pyramid_attr: True,
                f"{preview_name}_pyramid_levels": levels + 1,  # includes level 0
            })
            if console is not None:
                console.print(f"  [green]{preview_name} pyramid: {levels} levels written[/green]")
```

**Step 2: Verify import**

Run: `uv run python -c "from geotessera.zarr_zone import add_pyramids_to_existing_store; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "zarr: add add_pyramids_to_existing_store wrapper"
```

---

### Task 3: CLI flags `--pyramid` and `--pyramid-only`

**Files:**
- Modify: `geotessera/registry_cli.py` (~line 3395 in zarr-build argparse, ~line 2526 in zarr_build_command)

**Step 1: Add argparse flags**

In `main()`, after the `--pca-only` argument (around line 3408), add:

```python
    zarr_build_parser.add_argument(
        "--pyramid",
        action="store_true",
        help="Generate multi-resolution pyramids for preview arrays during build",
    )
    zarr_build_parser.add_argument(
        "--pyramid-only",
        action="store_true",
        help="Add pyramids to existing stores without rebuilding "
        "(scans existing .zarr stores in output dir)",
    )
```

**Step 2: Add `--pyramid-only` handler in `zarr_build_command`**

In `zarr_build_command()`, after the `--pca-only` handler block (around line 2635), add a new block following the same pattern. Add `add_pyramids_to_existing_store` to the import at the top of the function (line 2528):

Update the import line:
```python
    from .zarr_zone import build_zone_stores, add_rgb_to_existing_store, add_pca_to_existing_store, add_pyramids_to_existing_store
```

Then add the handler block:
```python
    # Handle --pyramid-only mode: add pyramids to existing stores
    if args.pyramid_only:
        zarr_dir = Path(output_dir)
        if not zarr_dir.is_dir():
            console.print(f"[red]Error: directory not found: {zarr_dir}[/red]")
            return 1

        zone_filter = None
        if args.zones:
            try:
                zone_filter = {int(z.strip()) for z in args.zones.split(",")}
            except ValueError:
                console.print("[red]Error: --zones must be comma-separated integers[/red]")
                return 1

        def _store_matches_pyr(p):
            if zone_filter is None:
                return True
            try:
                zone_num = int(p.name.split("_")[0].replace("utm", ""))
                return zone_num in zone_filter
            except (ValueError, IndexError):
                return True

        zarr_stores = sorted(
            p for p in zarr_dir.iterdir()
            if p.is_dir() and p.name.endswith(".zarr") and _store_matches_pyr(p)
        )

        if not zarr_stores:
            console.print(f"[yellow]No .zarr stores found in {zarr_dir}[/yellow]")
            return 1

        console.print(
            f"[bold]Adding pyramids to existing stores[/bold]\n"
            f"  Directory: {zarr_dir}\n"
            f"  Stores: {len(zarr_stores)}"
        )
        if zone_filter:
            console.print(f"  Zones: {', '.join(str(z) for z in sorted(zone_filter))}")

        for store_path in zarr_stores:
            console.print(f"\n  [cyan]{store_path.name}[/cyan]")
            add_pyramids_to_existing_store(store_path, console=console)

        console.print(f"\n[bold green]Pyramids added to {len(zarr_stores)} store(s)[/bold green]")
        return 0
```

**Step 3: Wire `--pyramid` into `build_zone_stores` call**

In `zarr_build_command()`, find the call to `build_zone_stores()` (around line 2680). After the RGB/PCA preview generation completes in `build_zone_stores` (around line 769 in zarr_zone.py), add pyramid building.

Add `pyramid: bool = False` parameter to `build_zone_stores` signature and wire it. At the end of the per-zone loop (after `pca` block, before `created_stores.append`), add:

```python
        if pyramid:
            for preview_name in ["rgb", "pca_rgb"]:
                if preview_name in store:
                    if console is not None:
                        console.print(f"  Building {preview_name} pyramid...")
                    levels = build_preview_pyramid(store, preview_name, console=console)
                    if levels > 0:
                        store.attrs.update({
                            f"has_{preview_name}_pyramid": True,
                            f"{preview_name}_pyramid_levels": levels + 1,
                        })
                        if console is not None:
                            console.print(f"  [green]{preview_name} pyramid: {levels} levels[/green]")
```

Then in `zarr_build_command()`, pass `pyramid=args.pyramid` to `build_zone_stores()`.

**Step 4: Verify CLI help**

Run: `uv run geotessera-registry zarr-build --help | grep -A1 pyramid`
Expected: Shows both `--pyramid` and `--pyramid-only` with descriptions.

**Step 5: Commit**

```bash
git add geotessera/registry_cli.py geotessera/zarr_zone.py
git commit -m "zarr-build: add --pyramid and --pyramid-only CLI flags"
```

---

### Task 4: Integration test

**Files:**
- Modify: `tests/zarr.t` (append test at end)

**Step 1: Add cram test for pyramid building**

Append to `tests/zarr.t`:

```
Test: Pyramid Building on Preview Arrays
-----------------------------------------

Build a small zone store with RGB, then add pyramids.
Use the Cambridge tiles downloaded earlier:

  $ geotessera-registry zarr-build \
  >   "$TESTDIR/cb_tiles_zarr" \
  >   --output-dir "$TESTDIR/zarr_pyramid_test" \
  >   --year 2024 \
  >   --rgb 2>&1 | grep -E '(RGB preview|Zone)' | head -3 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)

Add pyramids to the store:

  $ geotessera-registry zarr-build \
  >   "$TESTDIR/cb_tiles_zarr" \
  >   --output-dir "$TESTDIR/zarr_pyramid_test" \
  >   --pyramid-only 2>&1 | grep -E '(pyramid|Pyramids)' | head -3 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)

Verify pyramid structure exists in the zarr store:

  $ ZARR_STORE=$(find "$TESTDIR/zarr_pyramid_test" -name "*.zarr" -type d | head -1)
  $ uv run python -c "
  > import zarr
  > store = zarr.open_group('$ZARR_STORE', mode='r')
  > attrs = dict(store.attrs)
  > print(f'has_rgb_pyramid: {attrs.get(\"has_rgb_pyramid\", False)}')
  > pyramid = store['rgb_pyramid']
  > levels = sorted(k for k in pyramid.keys())
  > print(f'levels: {levels}')
  > level1 = pyramid['1']
  > full_h, full_w = store['rgb'].shape[:2]
  > l1_h, l1_w = level1.shape[:2]
  > print(f'full: {full_h}x{full_w}, level1: {l1_h}x{l1_w}')
  > print(f'halved: {l1_h == full_h // 2 and l1_w == full_w // 2}')
  > "
  has_rgb_pyramid: True
  levels: * (glob)
  full: *x*, level1: *x* (glob)
  halved: True
```

**Step 2: Run the test**

Run: `cd tests && uv run cram zarr.t`
Expected: All tests pass (existing + new).

**Step 3: Commit**

```bash
git add tests/zarr.t
git commit -m "tests: add pyramid building integration test"
```

---

### Task 5: Update module docstring

**Files:**
- Modify: `geotessera/zarr_zone.py:1-15`

**Step 1: Update the module docstring**

Update the store layout in the module docstring at the top of `zarr_zone.py` to reflect pyramids:

```python
"""Zone-wide Zarr format for consolidated Tessera embeddings.

This module provides tools for building and reading Zarr v3 stores that
consolidate all tiles within a UTM zone into a single store per year.
This enables efficient spatial subsetting and cloud-native access.

Store layout (uncompressed):
    utm{zone:02d}_{year}.zarr/
        embeddings        # int8    (northing, easting, band)  chunks=(1024, 1024, 128)
        scales            # float32 (northing, easting)        chunks=(1024, 1024)
        rgb               # uint8   (northing, easting, rgba)  chunks=(1024, 1024, 4)  [optional]
        pca_rgb           # uint8   (northing, easting, rgba)  chunks=(1024, 1024, 4)  [optional]
        rgb_pyramid/      # multi-resolution pyramid of rgb    [optional]
            1/ .. 7/      # uint8, each level 2x coarsened
        pca_rgb_pyramid/  # multi-resolution pyramid of pca_rgb [optional]
            1/ .. 7/      # uint8, each level 2x coarsened

NaN in scales indicates no-data (water or no coverage).
Embeddings are high-entropy quantised values; compression gives negligible
benefit so we store uncompressed.
"""
```

**Step 2: Commit**

```bash
git add geotessera/zarr_zone.py
git commit -m "docs: update zarr_zone module docstring with pyramid layout"
```
