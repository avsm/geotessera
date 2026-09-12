"""Shared geographic input normalization for the API and command line."""

import geopandas as gpd
import numpy as np


def parse_points(points):
    """Return finite WGS84 pairs, preserving input order and cardinality."""
    if isinstance(points, dict):
        if points.get("type") != "FeatureCollection":
            raise ValueError("Expected a GeoJSON FeatureCollection")
        features = points.get("features", [])
        if not features:
            return []
        points = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if isinstance(points, gpd.GeoDataFrame):
        if points.crs is None:
            raise ValueError("Point GeoDataFrame must declare its CRS")
        if (
            points.geometry.isna().any()
            or points.geometry.is_empty.any()
            or not points.geom_type.eq("Point").all()
        ):
            raise ValueError("Every geometry must be a nonempty Point")
        points = points.to_crs(4326)
        points = list(zip(points.geometry.x, points.geometry.y))
    values = np.asarray(list(points), dtype=float)
    if values.size == 0:
        return []
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Points must be (longitude, latitude) pairs")
    if (
        not np.isfinite(values).all()
        or (np.abs(values[:, 0]) > 180).any()
        or (np.abs(values[:, 1]) > 90).any()
    ):
        raise ValueError(
            "Points must contain finite WGS84 longitude/latitude coordinates"
        )
    return [tuple(row) for row in values]


def parse_bbox(value):
    """Parse CLI lon,lat or west,south,east,north; expand points to one tile."""
    from .registry import tile_from_world

    coords = (
        tuple(map(float, value.split(","))) if isinstance(value, str) else tuple(value)
    )
    if len(coords) == 2:
        lon, lat = parse_points([coords])[0]
        lon, lat = tile_from_world(lon, lat)
        coords = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
    if len(coords) != 4 or not np.isfinite(coords).all():
        raise ValueError(
            "bbox must be 'lon,lat' (single tile) or 'min_lon,min_lat,max_lon,max_lat'"
        )
    west, south, east, north = coords
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Bounding box must have positive area within WGS84 bounds")
    return coords


def read_region_file(location):
    """Read a supported local/remote vector dataset in WGS84."""
    frame = (
        location.copy()
        if isinstance(location, gpd.GeoDataFrame)
        else gpd.read_file(location)
    )
    if frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError("Region must contain nonempty geometries")
    if frame.crs is None:
        raise ValueError("Region file must declare its CRS")
    return frame.to_crs(4326)


def resolve_region(*, bbox=None, tile=None, region_file=None, country=None):
    """Resolve exactly one selector to WGS84 bounds and optional geometry."""
    count = sum(value is not None for value in (bbox, tile, region_file, country))
    if count != 1:
        reason = (
            "Cannot specify multiple region options."
            if count > 1
            else "Must specify a region."
        )
        raise ValueError(
            reason + " Choose one of: --bbox, --tile, --region-file, --country"
        )
    if tile is not None and len(tile.split(",")) != 2:
        raise ValueError("--tile must be 'lon,lat'")
    if bbox is not None or tile is not None:
        return parse_bbox(bbox if bbox is not None else tile), None
    if country is not None:
        from .country import get_country_lookup

        frame = get_country_lookup().get_geometry(country).to_crs(4326)
    else:
        frame = read_region_file(region_file)
    return parse_bbox(frame.total_bounds), frame
