#!/usr/bin/env python3
"""
Patch the global preview store with band coordinate arrays and spatial bounds.

Usage:
    python scripts/patch_global_bounds.py /path/to/zarr/v0/

For each global_rgb_*.zarr store found:
  1. Computes spatial.bounds using the EXACT same algorithm as
     build_global_preview (UTM corner reprojection at full precision).
  2. Creates band coordinate arrays at each pyramid level (for zarr-layer).
  3. Re-consolidates metadata.

The year is extracted from the store filename (global_rgb_YYYY.zarr) to
determine which zone stores (utmZZ_YYYY.zarr) contributed to the build.
Bounds are computed from the zone union, matching build_global_preview exactly.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import zarr
from pyproj import Transformer


def compute_zone_bounds_full_precision(base: Path, year: str, preview_names: list,
                                       zone_filter=None):
    """Compute WGS84 bounds from zone stores, matching build_global_preview exactly.

    Uses the same UTM corner + midpoint reprojection algorithm as
    build_global_preview in zarr_zone.py.

    Args:
        zone_filter: Optional set of zone numbers to include. If None, uses all.

    Returns (west, south, east, north) at full float64 precision,
    or None if no zone stores found.
    """
    pattern = re.compile(rf"^utm(\d{{2}})_{year}\.zarr$")
    zone_stores = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m:
            zone_num = int(m.group(1))
            if zone_filter is None or zone_num in zone_filter:
                zone_stores[zone_num] = entry

    if not zone_stores:
        return None

    print(f"  Found {len(zone_stores)} zone stores for year {year}: {sorted(zone_stores.keys())}")

    global_lon_min = float("inf")
    global_lon_max = float("-inf")
    global_lat_min = float("inf")
    global_lat_max = float("-inf")

    for zone_num, store_path in sorted(zone_stores.items()):
        store = zarr.open_group(str(store_path), mode="r")
        attrs = dict(store.attrs)
        epsg = int(attrs["proj:code"].split(":")[1])
        transform = list(attrs["spatial:transform"])
        pixel_size = transform[0]
        origin_e = transform[2]
        origin_n = transform[5]

        preview_shape = None
        for pname in preview_names:
            if pname in store:
                preview_shape = store[pname].shape
                break
        if preview_shape is None:
            print(f"    Zone {zone_num}: no preview arrays, skipping")
            continue

        h, w = preview_shape[:2]

        # Same corner + midpoint algorithm as build_global_preview
        corners_utm = [
            (origin_e, origin_n),
            (origin_e + w * pixel_size, origin_n),
            (origin_e, origin_n - h * pixel_size),
            (origin_e + w * pixel_size, origin_n - h * pixel_size),
        ]
        mid_e = origin_e + w * pixel_size / 2
        mid_n = origin_n - h * pixel_size / 2
        corners_utm += [
            (mid_e, origin_n),
            (mid_e, origin_n - h * pixel_size),
            (origin_e, mid_n),
            (origin_e + w * pixel_size, mid_n),
        ]

        transformer = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )
        corners_4326 = [transformer.transform(e, n) for e, n in corners_utm]
        lons = [c[0] for c in corners_4326]
        lats = [c[1] for c in corners_4326]

        zone_lon_min, zone_lon_max = min(lons), max(lons)
        zone_lat_min, zone_lat_max = min(lats), max(lats)

        global_lon_min = min(global_lon_min, zone_lon_min)
        global_lon_max = max(global_lon_max, zone_lon_max)
        global_lat_min = min(global_lat_min, zone_lat_min)
        global_lat_max = max(global_lat_max, zone_lat_max)

        print(
            f"    Zone {zone_num:02d} (EPSG:{epsg}): {h}x{w} px, "
            f"lon [{zone_lon_min:.6f}, {zone_lon_max:.6f}], "
            f"lat [{zone_lat_min:.6f}, {zone_lat_max:.6f}]"
        )

    if global_lon_min == float("inf"):
        return None

    return (global_lon_min, global_lat_min, global_lon_max, global_lat_max)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Patch global preview store with band arrays and spatial bounds."
    )
    parser.add_argument("zarr_dir", type=Path, help="Path to zarr/v0/ directory")
    parser.add_argument(
        "--zones", type=str, default=None,
        help="Comma-separated UTM zone numbers used to build the store "
             "(e.g. '29,30,31,32'). If omitted, uses all zones for the year."
    )
    args = parser.parse_args()

    base = args.zarr_dir
    zone_filter = None
    if args.zones:
        zone_filter = set(int(z.strip()) for z in args.zones.split(","))

    preview_names = ["rgb"]
    base_res = 0.0001
    num_bands = 4

    # Find global stores
    global_stores = sorted(base.glob("global_rgb_*.zarr"))
    if not global_stores:
        print("No global_rgb_*.zarr store found.", file=sys.stderr)
        sys.exit(1)

    for gs in global_stores:
        print(f"\nProcessing {gs.name}...")

        # Extract year from filename
        m = re.match(r"global_rgb_(\d{4})\.zarr", gs.name)
        if not m:
            print(f"  Cannot extract year from {gs.name}, skipping")
            continue
        year = m.group(1)

        zarr_json_path = gs / "zarr.json"
        if not zarr_json_path.exists():
            print(f"  No zarr.json, skipping")
            continue

        with open(zarr_json_path) as f:
            meta = json.load(f)

        # Read actual array shape from consolidated metadata
        cm = meta.get("consolidated_metadata", {}).get("metadata", {})
        level0_shape = None
        for key in ["0/rgb", "0/pca_rgb"]:
            arr_meta = cm.get(key)
            if arr_meta and "shape" in arr_meta:
                level0_shape = arr_meta["shape"]
                break

        if level0_shape:
            actual_h, actual_w = level0_shape[0], level0_shape[1]
            print(f"  Array shape: {actual_h} x {actual_w} (lat x lon)")

        # Compute bounds from zone stores for this year
        zone_bounds = compute_zone_bounds_full_precision(base, year, preview_names,
                                                          zone_filter=zone_filter)

        if zone_bounds:
            west, south, east, north = zone_bounds
            computed_w = int(math.ceil((east - west) / base_res))
            computed_h = int(math.ceil((north - south) / base_res))

            print(f"  Computed bounds: [{west:.10f}, {south:.10f}, {east:.10f}, {north:.10f}]")
            print(f"  Computed dims: {computed_w} x {computed_h}")

            if level0_shape:
                if computed_w == actual_w and computed_h == actual_h:
                    print(f"  Dimensions match exactly!")
                elif abs(computed_w - actual_w) <= 1 and abs(computed_h - actual_h) <= 1:
                    print(f"  Dimensions match within ±1 pixel (rounding)")
                else:
                    print(
                        f"  WARNING: Computed dims {computed_w}x{computed_h} != "
                        f"array {actual_w}x{actual_h}"
                    )
                    print(f"  The store may have been built with a --zones filter.")
                    print(f"  Cannot determine correct bounds — skipping bounds patch")
                    zone_bounds = None

        if zone_bounds:
            bounds = list(zone_bounds)
        else:
            # Fall back to existing bounds
            attrs = meta.get("attributes", {})
            existing = attrs.get("spatial:bbox") or attrs.get("spatial", {}).get("bounds")
            if existing:
                print(f"  Keeping existing bounds: {existing}")
                bounds = existing
            else:
                print(f"  ERROR: No bounds available — skipping")
                continue

        # Create band coordinate arrays at each pyramid level
        store_root = zarr.open_group(str(gs), mode="r+", zarr_format=3)
        level_keys = sorted(k for k in store_root.keys() if k.isdigit())
        band_data = np.arange(num_bands, dtype=np.int32)

        for lvl in level_keys:
            band_path = f"{lvl}/band"
            if band_path in store_root:
                print(f"  {band_path} already exists, skipping")
                continue
            store_root.create_array(
                band_path,
                data=band_data,
                chunks=(num_bands,),
            )
            print(f"  Created {band_path} coordinate array")

        # Re-consolidate metadata to include band arrays
        zarr.consolidate_metadata(str(gs))

        # Re-read and patch spatial bounds
        with open(zarr_json_path) as f:
            meta = json.load(f)
        attrs = meta.setdefault("attributes", {})
        attrs["spatial:bbox"] = bounds
        # Remove legacy key if present
        attrs.pop("spatial", None)
        with open(zarr_json_path, "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

        print(f"  Patched {gs.name} with spatial:bbox = {bounds}")


if __name__ == "__main__":
    main()
