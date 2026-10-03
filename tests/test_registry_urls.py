"""Dataset resolution and identity across public mirror routes."""

import pytest

from geotessera.registry import (
    DATASETS,
    TESSERA_MIRROR_S3_HTTP_URL,
    TESSERA_MIRROR_URL,
    dataset_for_location,
    dataset_from_tags,
    zarr_store_url,
)


@pytest.mark.parametrize(
    "version,variant,path",
    [
        ("v1", None, "v1"),
        ("v1.1", None, "v1.1-dclimate"),
        ("v1.1", "cambridge", "v1.1"),
        ("v2", None, "v2-2B-L~beta1"),
        ("v2-2B-L~beta1", None, "v2-2B-L~beta1"),
        ("v1", "custom", "v1-custom"),
        ("v1.1", "custom", "v1.1-custom"),
        ("v3", "custom", "v3-custom"),
        ("3.0", "custom", "v3-custom"),
        ("v3", None, "v3"),
    ],
)
def test_store_resolution(version, variant, path):
    assert zarr_store_url(version, variant) == f"{TESSERA_MIRROR_URL}/zarr/{path}"


def test_explicit_path_rejects_variant():
    with pytest.raises(ValueError, match="explicit store path"):
        zarr_store_url("v2-2B-L~beta1", "2B-L~beta2")


@pytest.mark.parametrize("dataset", [ds for ds in DATASETS if ds.zarr])
@pytest.mark.parametrize("base", [TESSERA_MIRROR_URL, TESSERA_MIRROR_S3_HTTP_URL])
def test_public_store_identity(dataset, base):
    location = f"{base}/zarr/{dataset.zarr}/"
    assert dataset_for_location(location) == dataset
    assert dataset_from_tags({"TESSERA_SOURCE": location}) == (
        dataset.version,
        dataset.variant,
    )


def test_other_mirrors_are_not_identified_as_public_stores():
    assert dataset_for_location("https://example.org/zarr/v1") is None
    assert dataset_for_location(f"{TESSERA_MIRROR_S3_HTTP_URL}-other/zarr/v1") is None
