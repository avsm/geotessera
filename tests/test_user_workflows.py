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


@pytest.mark.parametrize("per_band", [False, True])
def test_sparse_npy_sampling_matches_dense_nodata(tmp_path, per_band):
    tile = Tile.from_geotiff(write_tile(tmp_path / "grid_-2.95_52.05_2024.tif"))
    tile._format = "npy"
    tile._embedding_path = tmp_path / "embedding.npy"
    tile._scales_path = tmp_path / "scales.npy"
    values = np.arange(400, dtype=np.int8).reshape(10, 10, 4)
    scales = np.ones((10, 10, 4) if per_band else (10, 10), np.float32)
    scales[1, 2], scales[3, 4], scales[5, 6] = np.inf, -np.inf, np.nan
    np.save(tile._embedding_path, values)
    np.save(tile._scales_path, scales)
    rows, cols = [1, 3, 5, 7], [2, 4, 6, 8]
    xs, ys = rasterio.transform.xy(tile.transform, rows, cols)
    lon, lat = Transformer.from_crs(tile.crs, 4326, always_xy=True).transform(xs, ys)
    samples = tile.sample_points(list(zip(lon, lat)))
    np.testing.assert_allclose(
        samples, tile.load_embedding()[rows, cols], equal_nan=True
    )
    assert np.isnan(samples[:3]).all()


def test_wide_geotiffs_keep_visualization_strips_small(tmp_path):
    path = tmp_path / "grid_0.05_52.05_2024.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=12000,
        height=128,
        count=32,
        dtype="float32",
        crs=32631,
        transform=from_origin(500000, 5800000, 10, 10),
        nodata=np.nan,
        tiled=True,
        compress="lzw",
        SPARSE_OK=True,
    ):
        pass
    blocks = Tile.from_geotiff(path).iter_blocks()
    try:
        top, data = next(blocks)
        assert top == 0 and data.shape[1:] == (12000, 32)
        assert data.nbytes <= 64 * 1024**2
        assert np.isnan(data).all()
    finally:
        blocks.close()


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


def test_far_outside_patch_keeps_requested_location():
    gt, _ = fake_region()
    lon, lat = -2.8, 52.05
    patch, transform, crs = gt.read_patch(lon, lat, 2024, 4)
    assert np.isnan(patch).all()
    e, n = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    actual_e, actual_n = transform * (2.5, 2.5)
    assert abs(actual_e - e) <= 5.01 and abs(actual_n - n) <= 5.01


def test_generator_with_progress(monkeypatch):
    gt = GeoTessera.__new__(GeoTessera)
    gt.logger = logging.getLogger("test")
    monkeypatch.setattr(
        gt, "fetch_embedding", lambda *args: (np.ones((1, 1, 4)), None, None)
    )
    result = list(gt.fetch_embeddings(iter([(2024, 0.05, 52.05)]), lambda *args: None))
    assert len(result) == 1


def test_streamed_export_selected_bands_and_provenance(tmp_path):
    gt, bbox = fake_region()
    files = gt.export_geotiffs(bbox, 2024, tmp_path, bands=[3, 1], strip_rows=2)
    assert len(files) == 1
    with rasterio.open(files[0]) as src:
        assert src.count == 2 and src.crs.to_epsg() == 32630
        assert np.isnan(src.nodata)
        assert src.tags()["TESSERA_SOURCE"] == "fake://"
        np.testing.assert_array_equal(src.read()[:, 0, 0], [4, 2])
    assert len(discover_tiles(tmp_path)) == 1


def test_stream_export_dry_run_does_not_read_chunks(tmp_path):
    gt, bbox = fake_region()
    gt._root = object()  # Any attempt to read a Zarr chunk fails.
    estimate = gt.export_geotiffs(bbox, 2024, tmp_path / "absent", dry_run=True)
    assert estimate[0]["uncompressed_bytes"] > 0
    assert not (tmp_path / "absent").exists()


def test_stream_export_crosses_zones(tmp_path):
    gt = _fake_store({30: _seam_zone(32630, True), 31: _seam_zone(32631, False)})
    files = gt.export_geotiffs(
        (-0.001, 51.999, 0.001, 52.001), 2024, tmp_path, bands=[0]
    )
    assert len(files) == 2
    zones = set()
    for path in files:
        with rasterio.open(path) as src:
            zones.add(src.crs.to_epsg())
    assert zones == {32630, 32631}


def test_stream_failure_keeps_previous_export(tmp_path):
    gt, bbox = fake_region()
    output = tmp_path / "tessera_2024_utm30.tif"
    output.write_bytes(b"previous export")

    class BrokenScales:
        def __getitem__(self, key):
            raise OSError("interrupted read")

    gt._root = {
        "utm30": {
            "embeddings": gt._root["utm30"]["embeddings"],
            "scales": BrokenScales(),
        }
    }
    with pytest.raises(OSError, match="interrupted"):
        gt.export_geotiffs(bbox, 2024, tmp_path)
    assert output.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("operation", ["stream", "merge"])
def test_large_compressed_outputs_use_bigtiff(tmp_path, monkeypatch, operation):
    from types import SimpleNamespace
    import geotessera.streaming as streaming

    original_open = rasterio.open
    transform = from_origin(500000, 5800000, 10, 10)
    source = tmp_path / "large-source.tif"
    with original_open(
        source,
        "w",
        driver="GTiff",
        width=3300,
        height=3300,
        count=128,
        dtype="float32",
        crs=32630,
        transform=transform,
        tiled=True,
        compress="lzw",
        SPARSE_OK=True,
        BIGTIFF="YES",
    ):
        pass
    headers = []

    class StopBeforePixelReads(Exception):
        pass

    def inspect_writer(path, mode="r", **kwargs):
        if mode == "w":
            # Create the actual large-layout header without allocating its pixels.
            with original_open(path, mode, SPARSE_OK=True, **kwargs):
                pass
            with open(path, "rb") as file:
                headers.append(file.read(4))
            raise StopBeforePixelReads
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(rasterio, "open", inspect_writer)
    with pytest.raises(StopBeforePixelReads):
        if operation == "stream":
            gt, bbox = fake_region()
            gt.n_bands = 128
            window = SimpleNamespace(x0=0, x1=3300, y0=0, y1=3300, transform=transform)
            monkeypatch.setattr(
                streaming, "region_windows", lambda *args: [(30, gt._cache[30], window)]
            )
            gt.export_geotiffs(bbox, 2024, tmp_path / "output")
        else:
            merge_geotiffs([source], tmp_path / "merged.tif", 32630)
    assert headers == [b"II+\x00"] or headers == [b"MM\x00+"]


def test_stream_export_reads_selected_depth(tmp_path):
    gt, bbox = fake_region()
    group = gt._root["utm30"]
    data = group["embeddings"][:, :2, :, :]
    group.create_array("embeddings_d2", data=data)
    gt._cache[30]["embeddings_d2"] = (
        gt._cache[30]["embeddings"].isel(band=slice(0, 2)).rename(band="band_d2")
    )
    gt.depths[2] = "embeddings_d2"
    files = gt.export_geotiffs(bbox, 2024, tmp_path, depth=2)
    with rasterio.open(files[0]) as src:
        assert src.count == 2
        np.testing.assert_array_equal(src.read()[:, 0, 0], [1, 2])


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


def test_patch_rejects_geographic_resolution_units():
    gt, _ = fake_region()
    with pytest.raises(ValueError, match="metre"):
        gt.read_patch(-2.95, 52.05, 2024, 4, dst_crs="EPSG:4326")


@pytest.mark.parametrize("kwargs", [{"bands": [-1]}, {"bands": [4]}, {"strip_rows": 0}])
def test_stream_export_validates_before_writing(tmp_path, kwargs):
    gt, bbox = fake_region()
    with pytest.raises(ValueError):
        gt.export_geotiffs(bbox, 2024, tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_cli_download_streams_without_registry(tmp_path, monkeypatch):
    import geotessera.workflows as workflows
    import geotessera.cli as cli

    gt, bbox = fake_region()
    monkeypatch.setattr(workflows, "open_stream", lambda *args: gt)
    monkeypatch.setattr(
        cli,
        "GeoTessera",
        lambda **kwargs: pytest.fail("Zarr must not construct Registry"),
    )
    result = CliRunner().invoke(
        app,
        [
            "download",
            "--bbox",
            ",".join(map(str, bbox)),
            "--output",
            str(tmp_path),
            "--bands",
            "0,2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tessera_2024_utm30.tif").exists()


def test_cli_download_from_real_local_zarr(tmp_path):
    import zarr

    gt, bbox = fake_region()
    store = tmp_path / "local.zarr"
    gt._cache[30].to_zarr(store, group="utm30", zarr_format=3, consolidated=False)
    root = zarr.open_group(store, mode="a")
    root.attrs["geoemb:dimensions"] = 4
    zarr.consolidate_metadata(root.store)
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "download",
            "--store-url",
            str(store),
            "--bbox",
            ",".join(map(str, bbox)),
            "--output",
            str(output),
            "--bands",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    with rasterio.open(output / "tessera_2024_utm30.tif") as src:
        assert np.all(src.read() == 3)


def test_cli_webmap_streams_region(tmp_path, monkeypatch):
    import geotessera.workflows as workflows
    import geotessera.cli as cli

    gt, bbox = fake_region()
    monkeypatch.setattr(workflows, "open_stream", lambda *args: gt)

    def fake_tiles(**kwargs):
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    monkeypatch.setattr(cli, "geotiff_to_web_tiles", fake_tiles)
    result = CliRunner().invoke(
        app, ["webmap", "--bbox", ",".join(map(str, bbox)), "--output", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "viewer.html").exists()
    with rasterio.open(tmp_path / "rgb_mosaic.tif") as src:
        assert src.count == 3 and src.crs.to_epsg() == 3857
        assert src.dtypes == ("uint8",) * 3


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


def test_four_pca_components_do_not_mark_data_as_alpha(tmp_path):
    from geotessera.projection import write_pca_tiles

    source = dict(
        data=np.random.default_rng(42).normal(size=(10, 10, 4)).astype(np.float32),
        width=10,
        height=10,
        crs=32630,
        transform=from_origin(500000, 5800000, 10, 10),
    )
    paths = write_pca_tiles([source], tmp_path, n_components=4)
    with rasterio.open(paths[0]) as src:
        assert src.count == 4
        assert rasterio.enums.ColorInterp.alpha not in src.colorinterp
        assert np.all(src.dataset_mask() == 255)


def test_webmap_reuses_completed_stages(tmp_path, monkeypatch):
    import geotessera.workflows as workflows
    import geotessera.cli as cli

    gt, bbox = fake_region()
    reads, renders = [], []

    def open_stream(*args):
        reads.append(args)
        return gt

    def tiles(**kwargs):
        renders.append(kwargs)
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True)
        (directory / "tile.png").touch()
        return str(directory)

    monkeypatch.setattr(workflows, "open_stream", open_stream)
    monkeypatch.setattr(cli, "geotiff_to_web_tiles", tiles)
    args = ["webmap", "--bbox", ",".join(map(str, bbox)), "--output", str(tmp_path)]
    for extra, counts in [
        ([], (1, 1)),
        ([], (1, 1)),
        (["--max-zoom", "14"], (1, 2)),
        (["--bands", "2,1,0"], (2, 3)),
        (["--bands", "2,1,0", "--force"], (3, 4)),
    ]:
        result = CliRunner().invoke(app, args + extra)
        assert result.exit_code == 0, result.output
        assert (len(reads), len(renders)) == counts


def test_webmap_retries_failed_tiles_without_streaming(tmp_path, monkeypatch):
    import geotessera.workflows as workflows
    import geotessera.cli as cli

    gt, bbox = fake_region()
    monkeypatch.setattr(workflows, "open_stream", lambda *args: gt)

    def failed_tiles(**kwargs):
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True)
        (directory / "partial.png").touch()
        raise RuntimeError("interrupted")

    monkeypatch.setattr(cli, "geotiff_to_web_tiles", failed_tiles)
    args = ["webmap", "--bbox", ",".join(map(str, bbox)), "--output", str(tmp_path)]
    assert CliRunner().invoke(app, args).exit_code == 1
    assert not (tmp_path / "tiles.json").exists()
    monkeypatch.setattr(
        workflows, "open_stream", lambda *args: pytest.fail("streamed again")
    )

    def tiles(**kwargs):
        directory = Path(kwargs["output_dir"])
        assert not directory.exists()
        directory.mkdir()
        (directory / "finished.png").touch()
        return str(directory)

    monkeypatch.setattr(cli, "geotiff_to_web_tiles", tiles)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("ipv6", [False, True])
def test_occupied_port_fails_before_streaming(tmp_path, monkeypatch, ipv6):
    import http.server
    import socket
    import geotessera.workflows as workflows

    if ipv6 and not socket.has_dualstack_ipv6():
        pytest.skip("IPv6 unavailable")
    # A broken bind check must fail this regression, never start an endless
    # server loop inside CliRunner (which previously hung Windows CI).
    monkeypatch.setattr(
        http.server.ThreadingHTTPServer,
        "serve_forever",
        lambda *args, **kwargs: pytest.fail("Started a server on an occupied port"),
    )
    monkeypatch.setattr(workflows, "open_stream", lambda *args: pytest.fail("streamed"))
    with socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET) as listener:
        listener.bind(("::1" if ipv6 else "127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        for args in [
            ["serve", str(tmp_path), "--no-open"],
            [
                "webmap",
                "--bbox",
                "0,52,0.01,52.01",
                "--output",
                str(tmp_path),
                "--serve",
            ],
        ]:
            result = CliRunner().invoke(app, args + ["--port", str(port)])
            assert result.exit_code != 0
            assert "Cannot bind web server" in result.output
            assert not (tmp_path / "rgb_mosaic.tif").exists()


def test_web_server_accepts_connections_on_a_free_port(tmp_path):
    import socket
    from geotessera.cli import _bind_web_server

    with _bind_web_server(tmp_path, 0) as server:
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=2):
            pass
