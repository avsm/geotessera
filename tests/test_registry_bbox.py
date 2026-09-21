"""Tests for bbox-scoped manifest loading (Registry(bbox=...))."""

from pathlib import Path

import pandas as pd
import pytest

from geotessera.registry import Registry


def make_manifest(path: Path, rows):
    """Write a minimal file-scan-inventory-schema manifest.parquet."""
    df = pd.DataFrame(
        rows, columns=["year", "lon", "lat", "grid_size", "scales_size"]
    )
    df.to_parquet(path)


def make_empty_landmasks(path: Path):
    """Write a minimal landmasks.parquet so Registry() makes no network call."""
    pd.DataFrame(
        {"lon": [], "lat": [], "file_size": []}
    ).astype({"lon": "float64", "lat": "float64", "file_size": "int64"}).to_parquet(
        path
    )


@pytest.fixture
def make_registry(tmp_path: Path):
    """Build a Registry over the given manifest rows."""

    def make(rows, **kwargs):
        manifest_path = tmp_path / "manifest.parquet"
        landmasks_path = tmp_path / "landmasks.parquet"
        make_manifest(manifest_path, rows)
        make_empty_landmasks(landmasks_path)
        return Registry(
            version="v1",
            registry_path=manifest_path,
            landmasks_registry_path=landmasks_path,
            **kwargs,
        )

    return make


ANDORRA = [(2024, 1.55, 42.55, 100, 10), (2024, 1.65, 42.55, 100, 10)]
ELSEWHERE = [(2024, 50.05, -10.05, 100, 10), (2024, -120.05, 35.05, 100, 10)]


def test_bbox_scopes_loaded_rows(make_registry):
    """Only rows within bbox (+ margin) end up in the registry."""
    registry = make_registry(ANDORRA + ELSEWHERE, bbox=(1.5, 42.5, 1.7, 42.6))
    assert sorted(registry.get_available_embeddings()) == [
        (2024, 1.55, 42.55),
        (2024, 1.65, 42.55),
    ]


def test_bbox_margin_keeps_boundary_tiles(make_registry):
    """A tile just outside bbox, but within the margin, is kept."""
    rows = [
        (2024, 1.45, 42.55, 100, 10),  # centre 0.05 deg west of bbox
        (2024, 5.05, 42.55, 100, 10),  # far outside, dropped
    ]
    registry = make_registry(rows, bbox=(1.5, 42.5, 1.7, 42.6))
    assert registry.get_available_embeddings() == [(2024, 1.45, 42.55)]
    assert registry.load_blocks_for_region((1.5, 42.5, 1.6, 42.6), 2024) == [
        (2024, 1.45, 42.55)
    ]


def test_no_bbox_loads_everything(make_registry):
    """Without bbox, every row is loaded."""
    registry = make_registry(ANDORRA + ELSEWHERE)
    assert len(registry.get_available_embeddings()) == len(ANDORRA + ELSEWHERE)
    assert len(registry.load_blocks_for_region((-180, -90, 180, 90), 2024)) == 4


def test_region_query_outside_bbox_raises(make_registry):
    """A region query that leaves the loaded area raises instead of truncating."""
    registry = make_registry(ANDORRA + ELSEWHERE, bbox=(1.5, 42.5, 1.6, 42.6))
    assert registry.load_blocks_for_region((1.52, 42.52, 1.58, 42.58), 2024) == [
        (2024, 1.55, 42.55)
    ]
    with pytest.raises(ValueError, match="outside the registry bbox"):
        registry.load_blocks_for_region((1.5, 42.5, 2.0, 42.6), 2024)


def test_tile_lookup_outside_bbox_raises(make_registry):
    """Looking up a tile outside the loaded area names the bbox."""
    registry = make_registry(ANDORRA + ELSEWHERE, bbox=(1.5, 42.5, 1.6, 42.6))
    assert registry.get_tile_file_size(2024, 1.55, 42.55) == 100
    with pytest.raises(ValueError, match="outside the registry bbox"):
        registry.get_tile_file_size(2024, 50.05, -10.05)


def test_empty_bbox_raises_bbox_error(make_registry):
    """A bbox with no tiles reports the bbox, not the dataset version."""
    with pytest.raises(ValueError, match="no tiles within bbox"):
        make_registry(ANDORRA, bbox=(-30.0, -30.0, -29.0, -29.0))


@pytest.mark.parametrize(
    "bbox",
    [
        (179.5, -1.0, -179.5, 1.0),  # crosses the antimeridian
        (1.7, 42.5, 1.5, 42.6),  # swapped longitudes
        (1.5, 42.6, 1.7, 42.5),  # swapped latitudes
    ],
)
def test_invalid_bbox_raises(make_registry, bbox):
    with pytest.raises(ValueError, match="min <= max"):
        make_registry(ANDORRA, bbox=bbox)


def test_negative_margin_raises(make_registry):
    with pytest.raises(ValueError, match="non-negative"):
        make_registry(ANDORRA, bbox=(1.5, 42.5, 1.7, 42.6), bbox_margin_deg=-0.1)
