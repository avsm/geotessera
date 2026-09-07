"""Offline regressions for public sampling, raster, and streaming workflows."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point
from typer.testing import CliRunner

from geotessera.cli import app
from geotessera.core import GeoTessera
from geotessera.inputs import parse_points
from geotessera.raster import merge_geotiffs
from geotessera.tiles import Tile, discover_tiles, tile_for_coord
from test_store import _fake_store, _fake_zone, _seam_zone


def write_tile(path, value=1, nodata=-9999):
    e, n = Transformer.from_crs(4326, 32630, always_xy=True).transform(-2.95, 52.05)
    data = np.full((4, 10, 10), value, np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=4,
        dtype="float32",
        nodata=nodata,
        crs=32630,
        transform=from_origin(e - 50, n + 50, 10, 10),
    ) as dst:
        dst.write(data)
    return path


def fake_region():
    e, n = Transformer.from_crs(4326, 32630, always_xy=True).transform(-2.95, 52.05)
    ds = _fake_zone(np.ones((10, 10), np.float32), epsg=32630, ox=e - 50, oy=n + 50)
    return _fake_store({30: ds}), (-2.9504, 52.0497, -2.9496, 52.0503)


def test_projected_point_inputs_and_sparse_tiff_sampling(tmp_path, monkeypatch):
    path = write_tile(tmp_path / "grid_-2.95_52.05_2024.tif")
    tile = Tile.from_geotiff(path)
    monkeypatch.setattr(
        tile, "load_embedding", lambda: pytest.fail("must not load entire tile")
    )
    np.testing.assert_array_equal(tile.sample_at_point(-2.95, 52.05), np.ones(4))
    points = gpd.GeoDataFrame(geometry=[Point(-2.95, 52.05)], crs=4326).to_crs(32630)
    np.testing.assert_allclose(parse_points(points), [[-2.95, 52.05]])
    assert len(discover_tiles(tmp_path)) == 1
    assert tile_for_coord(tmp_path, -2.95, 52.05, 2024) is not None


def test_geojson_rejects_nonpoints_without_dropping_rows():
    with pytest.raises(ValueError, match="Point"):
        parse_points(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0, 0], [1, 1]],
                        },
                    }
                ],
            }
        )


def test_merge_preserves_valid_zero_and_nodata(tmp_path):
    zero = write_tile(tmp_path / "zero.tif", 0)
    one = write_tile(tmp_path / "one.tif", 1)
    with rasterio.open(zero, "r+") as dst:
        dst.write(
            np.full((4, 1, 1), -9999, np.float32),
            window=rasterio.windows.Window(0, 0, 1, 1),
        )
    output = tmp_path / "merged.tif"
    merge_geotiffs([zero, one], output, "EPSG:32630")
    with rasterio.open(output) as src:
        assert np.isnan(src.nodata)
        data = src.read()
    assert np.all(data[:, 0, 0] == 1)
    assert np.all(data[:, 1:, 1:] == 0)


def test_generator_with_progress(monkeypatch):
    gt = GeoTessera.__new__(GeoTessera)
    gt.logger = logging.getLogger("test")
    monkeypatch.setattr(
        gt, "fetch_embedding", lambda *args: (np.ones((1, 1, 4)), None, None)
    )
    result = list(gt.fetch_embeddings(iter([(2024, 0.05, 52.05)]), lambda *args: None))
    assert len(result) == 1


def test_fetch_failure_is_not_silently_skipped(monkeypatch):
    gt = GeoTessera.__new__(GeoTessera)

    def fail(*args):
        raise OSError("download interrupted")

    monkeypatch.setattr(gt, "fetch_embedding", fail)
    with pytest.raises(RuntimeError, match="Failed to fetch tile"):
        list(gt.fetch_embeddings([(2024, 0.05, 52.05)]))


def test_sampling_has_explicit_failure_policy(monkeypatch):
    gt = GeoTessera.__new__(GeoTessera)
    gt.logger = logging.getLogger("test")

    class BrokenTile:
        def sample_points(self, points):
            raise OSError("corrupt tile")

    monkeypatch.setattr(gt, "_group_points_by_tile", lambda *args: {(0.05, 52.05): [0]})
    monkeypatch.setattr(
        gt, "_ensure_tiles_available", lambda **kwargs: {(0.05, 52.05): BrokenTile()}
    )
    with pytest.raises(RuntimeError, match="corrupt tile"):
        gt.sample_embeddings_at_points([(0.05, 52.05)])
    values, metadata = gt.sample_embeddings_at_points(
        [(0.05, 52.05)], errors="coerce", include_metadata=True
    )
    assert np.isnan(values).all()
    assert "corrupt tile" in metadata[0]["error"]


def test_folium_viewer_uses_relative_tile_path(tmp_path):
    from geotessera.web import create_simple_web_viewer

    output = tmp_path / "viewer.html"
    create_simple_web_viewer(str(tmp_path / "custom tiles"), str(output))
    assert "custom%20tiles/{z}/{x}/{y}.png" in output.read_text()


def test_pca_shared_basis_and_nan_mask(tmp_path):
    from geotessera.projection import write_pca_tiles

    rng = np.random.default_rng(42)
    data = rng.normal(size=(10, 10, 4)).astype(np.float32)
    data[0, 0] = np.nan
    source = dict(
        data=data,
        width=10,
        height=10,
        crs=32630,
        transform=from_origin(500000, 5800000, 10, 10),
    )
    files = write_pca_tiles([source, source], tmp_path)
    with rasterio.open(files[0]) as a, rasterio.open(files[1]) as b:
        np.testing.assert_array_equal(a.read(), b.read())
        assert a.dataset_mask()[0, 0] == 0
        assert a.dataset_mask()[1, 1] == 255
