"""Read-only Icechunk input for the one-off geoembeddings transcoder.

Only logical initialized chunks in a pinned snapshot are inventoried. The
physical Icechunk chunks/ prefix is neither a Zarr hierarchy nor an inventory.
"""

from __future__ import annotations

import math
from datetime import UTC
from urllib.parse import urlparse

import numpy as np

from ._migration_state import canonical_url

SPATIAL_NAMES = {"northing": "y", "easting": "x", "y": "y", "x": "x"}


def open_source(url, *, snapshot_id=None, branch="main", options=None):
    import icechunk

    # Registers the source's numcodecs.pcodec Zarr-v3 codec.
    import zarr
    import zarr.codecs.numcodecs

    url = canonical_url(url)
    options = options or {}
    p = urlparse(url)
    if p.scheme == "s3":
        kwargs = {
            "bucket": p.netloc,
            "prefix": p.path.strip("/"),
            "region": options.get("client_kwargs", {}).get("region_name"),
            "endpoint_url": options.get("endpoint_url"),
            "anonymous": bool(options.get("anon")),
            "requester_pays": bool(options.get("requester_pays")),
            "force_path_style": options.get("config_kwargs", {})
            .get("s3", {})
            .get("addressing_style")
            == "path",
        }
        if options.get("profile") or options.get("key"):
            from datetime import datetime, timedelta

            import boto3

            session = boto3.Session(
                profile_name=options.get("profile"),
                aws_access_key_id=options.get("key"),
                aws_secret_access_key=options.get("secret"),
                aws_session_token=options.get("token"),
            )

            def credentials():
                frozen = session.get_credentials().get_frozen_credentials()
                # Re-enter botocore's refreshing provider chain periodically.
                return icechunk.S3StaticCredentials(
                    frozen.access_key,
                    frozen.secret_key,
                    frozen.token,
                    expires_after=datetime.now(UTC) + timedelta(minutes=5),
                )

            kwargs["get_credentials"] = credentials
        storage = icechunk.s3_storage(**kwargs)
    else:
        storage = icechunk.local_filesystem_storage(url)
    repo = icechunk.Repository.open(storage)
    session = (
        repo.readonly_session(snapshot_id=snapshot_id)
        if snapshot_id
        else repo.readonly_session(branch=branch)
    )
    return session, zarr.open_group(session.store, mode="r", use_consolidated=False)


def dims(array):
    names = array.metadata.dimension_names
    if names is None or any(n is None for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{array.path}: explicit, unique dimension_names are required")
    return [SPATIAL_NAMES.get(n, n) for n in names]


def fill_value(value):
    return (
        {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": -float("inf")}.get(
            value, value
        )
        if isinstance(value, str)
        else value
    )


def _years(array):
    values = np.asarray(array[:])
    units = array.attrs.get("units", "")
    if units == "nanoseconds since 1970-01-01":
        dates = values.astype("datetime64[ns]")
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        if not np.array_equal(
            dates, (years - 1970).astype("datetime64[Y]").astype("datetime64[ns]")
        ):
            raise ValueError(f"{array.path}: expected annual January 1 timestamps")
    elif np.issubdtype(values.dtype, np.integer) and np.all(
        (values >= 1900) & (values <= 2200)
    ):
        years = values
    else:
        raise ValueError(f"{array.path}: unsupported annual time encoding {units!r}")
    years = [int(y) for y in years]
    if years != sorted(set(years)):
        raise ValueError(f"{array.path}: years must be unique and increasing")
    return years


def _axis(array, step):
    # Coordinates are a few MB, unlike the embedding arrays. Validate every
    # coordinate to reject an irregular grid hidden between its endpoints.
    values = np.asarray(array[:], dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{array.path}: invalid coordinate axis")
    if len(values) > 1 and not np.allclose(np.diff(values), step, atol=1e-7, rtol=0):
        raise ValueError(f"{array.path}: expected regular {step} metre steps")
    return float(values[0]), len(values)


def inspect_group(group, name, arrays=None):
    """Validate one hemisphere and return a small JSON-serializable schema."""
    zone, hemisphere = int(name[:2]), name[2:]
    expected_epsg = (32600 if hemisphere == "N" else 32700) + zone
    attrs = dict(group.attrs)
    if attrs.get("proj:code", attrs.get("crs")) != f"EPSG:{expected_epsg}":
        raise ValueError(f"{name}: CRS does not match the group name")
    members = dict(group.arrays())
    if not {"embeddings", "scales", "time", "band"} <= members.keys():
        raise ValueError(f"{name}: missing embeddings, scales, time, or band")
    coordinate_names = {SPATIAL_NAMES[n]: n for n in members if n in SPATIAL_NAMES}
    if set(coordinate_names) != {"x", "y"}:
        raise ValueError(f"{name}: missing spatial coordinate arrays")
    x, width = _axis(members[coordinate_names["x"]], 10.0)
    y, height = _axis(members[coordinate_names["y"]], -10.0)
    if hemisphere == "S":
        y -= 10_000_000.0
    years = _years(members["time"])
    if not np.array_equal(members["band"][:], np.arange(128)):
        raise ValueError(f"{name}: expected band coordinates 0..127")
    if set(dims(members["embeddings"])) != {"time", "band", "y", "x"} or members[
        "embeddings"
    ].dtype != np.dtype("int8"):
        raise ValueError(
            f"{name}: expected int8 embeddings with time/band/y/x dimensions"
        )
    if set(dims(members["scales"])) != {"time", "y", "x"} or members[
        "scales"
    ].dtype != np.dtype("float32"):
        raise ValueError(f"{name}: expected float32 scales with time/y/x dimensions")
    if (
        not np.isnan(members["scales"].fill_value)
        or members["embeddings"].fill_value != 0
    ):
        raise ValueError(f"{name}: expected NaN scale fill and zero embedding fill")
    selected = (
        set(members)
        if arrays is None
        else set(arrays) | {"embeddings", "scales", "time", "band"}
    )
    missing = selected - members.keys()
    if missing:
        raise ValueError(f"{name}: unknown selected arrays {sorted(missing)}")
    # Retain dimension coordinates and annual bounds when selecting arrays.
    for n in list(selected):
        selected.update(d for d in members[n].metadata.dimension_names if d in members)
        if members[n].attrs.get("bounds") in members:
            selected.add(members[n].attrs["bounds"])
    selected -= set(coordinate_names.values()) | {"time", "band"}
    schemas, static = {}, {}
    for n in sorted(selected | {"embeddings", "scales"}):
        a = members[n]
        ds = dims(a)
        if ("x" in ds) != ("y" in ds):
            raise ValueError(f"{name}/{n}: spatial arrays must have both x and y")
        for d, size in zip(ds, a.shape):
            expected = {"time": len(years), "band": 128, "x": width, "y": height}.get(d)
            if expected is not None and size != expected:
                raise ValueError(f"{name}/{n}: incompatible {d} dimension")
        if "x" in ds:
            if "time" not in ds:
                raise ValueError(f"{name}/{n}: spatial arrays require time")
            if a.dtype.kind not in "biuf":
                raise ValueError(f"{name}/{n}: unsupported dtype {a.dtype}")
            output_dims = (
                ["time", "band", "y", "x"]
                if n == "embeddings"
                else ["time", "y", "x"] + [d for d in ds if d not in ("time", "y", "x")]
            )
            md = a.metadata.to_dict()
            schemas[n] = {
                "dims": ds,
                "output_dims": output_dims,
                "shape": list(a.shape),
                "outer_chunks": list(a.metadata.chunk_grid.chunk_shape),
                "dtype": a.dtype.str,
                "fill": md["fill_value"],
                "attrs": dict(a.attrs),
                "metadata": md,
            }
        else:
            if a.size > 1_000_000:
                raise ValueError(
                    f"{name}/{n}: nonspatial array too large to include in metadata"
                )
            data = np.asarray(a[:])
            if data.dtype.kind == "f" and not np.isfinite(data).all():
                raise ValueError(
                    f"{name}/{n}: nonfinite static coordinates are unsupported"
                )
            static[n] = {
                "dims": ds,
                "dtype": a.dtype.str,
                "fill": a.metadata.to_dict()["fill_value"],
                "attrs": dict(a.attrs),
                "data": data.tolist(),
                "shape": list(a.shape),
            }
    return {
        "name": name,
        "zone": zone,
        "hemisphere": hemisphere,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "years": years,
        "arrays": schemas,
        "static": static,
        "attrs": attrs,
    }


def zone_grid(groups, years, shard_size):
    """Integer placement on a common 10m grid; do not resample anything."""
    x = min(g["x"] for g in groups)
    y = max(g["y"] for g in groups)
    for g in groups:
        ox, oy = (g["x"] - x) / 10, (y - g["y"]) / 10
        if abs(ox - round(ox)) > 1e-6 or abs(oy - round(oy)) > 1e-6:
            raise ValueError("Hemisphere coordinate grids have different pixel phases")
        g["col_offset"], g["row_offset"] = round(ox), round(oy)
    width = max(g["col_offset"] + g["width"] for g in groups)
    height = max(g["row_offset"] + g["height"] for g in groups)
    return {
        "x": x,
        "y": y,
        "width": math.ceil(width / shard_size) * shard_size,
        "height": math.ceil(height / shard_size) * shard_size,
        "years": years,
    }


async def inventory(session, root, groups, grid, shard_size, log):
    """Map sparse initialized outer chunks to destination work units.

    Memory is proportional to destination units for one zone, not source
    chunks or the world's bounding rectangles. Count every selected array.
    """
    units, counts = set(), {}
    for g in groups:
        for name, spec in g["arrays"].items():
            count = 0
            ds, shape, chunks = spec["dims"], spec["shape"], spec["outer_chunks"]
            yi, xi, ti = (ds.index(d) for d in ("y", "x", "time"))
            async for coord in session.chunk_coordinates(f"/{g['name']}/{name}"):
                if len(coord) != len(shape) or any(
                    c < 0 or c * k >= n for c, k, n in zip(coord, chunks, shape)
                ):
                    raise ValueError(
                        f"Invalid initialized chunk in {g['name']}/{name}: {coord}"
                    )
                count += 1
                # A real codec read before declaring this array usable. The
                # first initialized chunk may not be at the array origin.
                if count == 1:
                    selection = tuple(
                        slice(c * k, c * k + 1) for c, k in zip(coord, chunks)
                    )
                    np.asarray(root[g["name"]][name][selection])
                r0 = g["row_offset"] + coord[yi] * chunks[yi]
                c0 = g["col_offset"] + coord[xi] * chunks[xi]
                r1 = g["row_offset"] + min(shape[yi], (coord[yi] + 1) * chunks[yi])
                c1 = g["col_offset"] + min(shape[xi], (coord[xi] + 1) * chunks[xi])
                for t in range(
                    coord[ti] * chunks[ti], min(shape[ti], (coord[ti] + 1) * chunks[ti])
                ):
                    year = g["years"][t]
                    if year not in grid["years"]:
                        continue
                    for r in range(r0 // shard_size, (r1 - 1) // shard_size + 1):
                        for c in range(c0 // shard_size, (c1 - 1) // shard_size + 1):
                            units.add((year, r, c))
            counts[f"{g['name']}/{name}"] = count
            log(
                "inventory_array",
                array=f"{g['name']}/{name}",
                source_chunks=count,
                destination_units=len(units),
            )
    return sorted(units), counts
