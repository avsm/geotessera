"""Real Icechunk fixtures and failures at the migration's commit boundaries."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zarr

icechunk = pytest.importorskip("icechunk")
pytest.importorskip("pcodec")
from zarr.codecs.numcodecs import PCodec

from geotessera._migration_state import (
    BusyError,
    State,
    digest,
)
from geotessera.transcode import RESUMABLE, Migration, create_plan


@pytest.fixture
def migration(tmp_path):
    source = tmp_path / "input.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(source)))
    session = repo.writable_session("main")
    root = zarr.group(session.store)
    root.attrs.update(
        {
            "geoemb:dimensions": 128,
            "geoemb:data_type": "int8",
            "geoemb:model": "https://geotessera.org/model/1.1",
            "geoemb:quantization": {
                "method": "per_pixel_scale",
                "scale": {"type": "array", "array_name": "scales", "nodata": "NaN"},
            },
        }
    )
    for hemisphere, offset in (("N", 0), ("S", 3)):
        group = root.create_group(
            "31" + hemisphere,
            attributes={
                "proj:code": "EPSG:32631" if hemisphere == "N" else "EPSG:32731",
                "years_complete": [2024] if hemisphere == "S" else [2023, 2024],
            },
        )
        for n, data in (
            ("time", np.array([2023, 2024], dtype="i4")),
            ("band", np.arange(128, dtype="i4")),
            ("month", np.arange(1, 13, dtype="i2")),
            ("easting", np.arange(8) * 10.0 + 500005),
            (
                "northing",
                45.0
                - (np.arange(4) + offset) * 10
                + (10_000_000 if hemisphere == "S" else 0),
            ),
        ):
            group.create_array(n, data=data, dimension_names=[n])
        group.create_array(
            "time_bnds",
            data=np.array([[20230101, 20231231], [20240101, 20241231]], dtype="i8"),
            dimension_names=["time", "bnds"],
            attributes={"units": "YYYYMMDD"},
        )
        group["time"].attrs["bounds"] = "time_bnds"
        emb = group.create_array(
            "embeddings",
            shape=(2, 4, 8, 128),
            dtype="i1",
            fill_value=0,
            dimension_names=["time", "northing", "easting", "band"],
            chunks=(1, 1, 1, 128),
            shards=(1, 2, 2, 128),
        )
        scale = group.create_array(
            "scales",
            shape=(4, 2, 8),
            dtype="f4",
            fill_value=np.nan,
            dimension_names=["northing", "time", "easting"],
            chunks=(1, 1, 1),
            shards=(2, 1, 2),
            serializer=PCodec(),
            compressors=None,
        )
        counts = group.create_array(
            "s2_obs_count",
            shape=(2, 4, 8),
            dtype="u2",
            fill_value=0,
            dimension_names=["time", "northing", "easting"],
            chunks=(1, 1, 1),
            shards=(1, 2, 2),
        )
        monthly = group.create_array(
            "s2_month_covered",
            shape=(2, 4, 8, 12),
            dtype="i1",
            fill_value=0,
            dimension_names=["time", "northing", "easting", "month"],
            chunks=(1, 1, 1, 12),
            shards=(1, 2, 2, 12),
        )
        # Only the western half has embeddings; an eastern pixel has QA only.
        for t in range(2):
            for row in range(4):
                emb[t, row, :4, :] = t * 10 + row + offset
                scale[row, t, :4] = 0.25 + t * 0.25 + (row + offset) * 0.0625
                counts[t, row, :4] = 20 + row + offset
                monthly[t, row, :4, :] = np.tile(np.arange(12) % 2, (4, 1))
        counts[0, 0, 6] = 7
    snapshot = session.commit("fixture")
    state, destination = tmp_path / "state", tmp_path / "output.zarr"
    plan = create_plan(
        str(source),
        str(state),
        str(destination),
        snapshot_id=snapshot,
        shard_size=4,
        inner_chunk=2,
    )
    m = Migration(str(state))
    m.repo = repo
    m.fixture_plan = plan
    return m


def test_roundtrip_and_noop_resume(migration, tmp_path):
    m = migration
    m.initialize()
    assert m.transcode([31], workers=2, spill_dir=str(tmp_path / "spill")) == 0
    assert {r["status"] for r in m.scan(checksum=True)} == {"complete"}
    report = m.finalize(full=True)
    assert report["checked_shards"] > 0
    group = m.store.open_group(mode="r", path="utm31")
    np.testing.assert_array_equal(group["y"][:], [45, 35, 25, 15, 5, -5, -15, -25])
    np.testing.assert_array_equal(group["x"][:], np.arange(8) * 10 + 500005)
    assert group.attrs["spatial:transform"] == [10, 0, 500000.0, 0, -10, 50.0]
    from zarr_cm import geo_proj, spatial

    spatial.validate(dict(group.attrs))
    geo_proj.validate(dict(group.attrs))
    assert group["embeddings"][0, :, 0, 0].sum() == 0  # valid zero vector
    assert group["scales"][0, 0, 0] == 0.25
    for t in range(2):
        expected = np.broadcast_to((t * 10 + np.arange(7))[None, :, None], (128, 7, 4))
        np.testing.assert_array_equal(group["embeddings"][t, :, :7, :4], expected)
        expected_scales = np.broadcast_to(
            (0.25 + t * 0.25 + np.arange(7) * 0.0625)[:, None], (7, 4)
        )
        np.testing.assert_array_equal(group["scales"][t, :7, :4], expected_scales)
    assert np.isnan(group["scales"][0, 0, 6])
    assert group["s2_obs_count"][0, 0, 6] == 7  # ancillary-only shard
    assert group["time_bnds"].attrs["units"] == "YYYYMMDD"
    assert group.attrs["geotessera:source_groups"]["31S"]["years_complete"] == [2024]
    before = {
        str(p): p.stat().st_mtime_ns
        for p in Path(m.store.url).rglob("*")
        if p.is_file()
    }
    assert m.transcode([31]) == 0
    after = {
        str(p): p.stat().st_mtime_ns
        for p in Path(m.store.url).rglob("*")
        if p.is_file()
    }
    assert before == after


def test_public_reader_verifies_data(migration, tmp_path):
    m = migration
    m.initialize()
    assert m.transcode([31]) == 0
    m.finalize(full=True)
    public = tmp_path / "public.zarr"
    shutil.copytree(m.store.url, public)
    report = m.verify_public(str(public))
    assert report["checked_public_patches"] == 2
    group = zarr.open_group(str(public), mode="r+", use_consolidated=False)["utm31"]
    group["embeddings"][0, :, 0, 0] = 99
    with pytest.raises(ValueError, match="Public data mismatch"):
        m.verify_public(str(public))


def test_uploaded_object_disappears_before_receipt(migration, monkeypatch):
    m = migration
    m.initialize()
    doc = m.zone(31)
    unit = next(m.units(doc))
    key = m.object_key(doc, "scales", unit)
    original = m.object_info

    def lose_object(name):
        if name == key:
            (Path(m.store.url) / name).unlink()
        return original(name)

    monkeypatch.setattr(m, "object_info", lose_object)
    with pytest.raises(RuntimeError, match="did not publish"):
        m.write_unit(doc, unit)
    assert m.unit_status(doc, unit) == "partial"


@pytest.mark.parametrize(
    "failure", ["scales", "s2_obs_count", "s2_month_covered", "embeddings", "receipt"]
)
def test_interrupted_unit_repaired(migration, monkeypatch, failure):
    m = migration
    m.initialize()
    with monkeypatch.context() as patch:
        if failure == "receipt":
            original = m.state.write

            def fail(key, *args, **kwargs):
                if key.startswith("commits/"):
                    raise OSError("interrupted receipt")
                return original(key, *args, **kwargs)

            patch.setattr(m.state, "write", fail)
        else:
            original = zarr.Array.__setitem__

            def fail(array, selection, value):
                if array.path.endswith("/" + failure):
                    raise OSError("interrupted array")
                return original(array, selection, value)

            patch.setattr(zarr.Array, "__setitem__", fail)
        with pytest.raises(OSError, match="interrupted"):
            m.transcode([31])
    assert m.transcode([31]) == 0
    m.finalize(full=True)


def test_bounded_attempt_and_missing_object_recovery(migration):
    m = migration
    m.initialize()
    assert m.transcode([31], max_shards=1) == RESUMABLE
    rows = list(m.scan())
    assert sum(r["status"] == "complete" for r in rows) == 1
    assert m.transcode([31]) == 0
    doc = m.zone(31)
    unit = next(m.units(doc))
    m.store.remove(m.object_key(doc, "scales", unit))
    assert m.unit_status(doc, unit) == "partial"
    assert m.transcode([31]) == 0
    m.finalize(full=True)


def test_initialize_recovers_coordinate_failure(migration, monkeypatch):
    m = migration
    original = zarr.Array.__setitem__

    def fail(array, selection, value):
        if array.path == "utm31/y":
            raise OSError("coordinates")
        return original(array, selection, value)

    with monkeypatch.context() as patch:
        patch.setattr(zarr.Array, "__setitem__", fail)
        with pytest.raises(OSError, match="coordinates"):
            m.initialize()
    with pytest.raises(FileNotFoundError):
        m.transcode([31])
    m.initialize()
    assert m.transcode([31]) == 0


def test_finalization_requires_complete_and_repairs_consolidation(
    migration, monkeypatch
):
    m = migration
    m.initialize()
    with pytest.raises(ValueError, match="incomplete"):
        m.finalize()
    assert m.transcode([31]) == 0
    with monkeypatch.context() as patch:
        patch.setattr(
            zarr,
            "consolidate_metadata",
            lambda *a, **k: (_ for _ in ()).throw(OSError("consolidation")),
        )
        with pytest.raises(OSError, match="consolidation"):
            m.finalize()
    assert m.state.get("complete.json")[0] is None
    m.finalize()


def test_reject_changed_schema_and_destination(migration):
    m = migration
    with pytest.raises(ValueError, match="Destination"):
        Migration(m.state.url, destination=m.store.url + "-other")
    m.initialize()
    root = m.store.open_group(mode="r+")
    root["utm31"].attrs["spatial:transform"] = [10, 0, 0, 0, -10, 0]
    with pytest.raises(ValueError, match="metadata"):
        m.transcode([31])


def test_pinned_snapshot_ignores_branch_updates(migration):
    m = migration
    session = m.repo.writable_session("main")
    root = zarr.open_group(session.store)
    root["31N/embeddings"][0, 0, 0, :] = 99
    session.commit("changed after plan")
    m.initialize()
    assert m.transcode([31]) == 0
    assert not m.store.open_group(mode="r")["utm31/embeddings"][0, :, 0, 0].any()
    with pytest.raises(ValueError, match="changed"):
        create_plan(
            m.request["source"], m.state.url, m.store.url, shard_size=8, inner_chunk=2
        )


def test_duplicate_zone_and_controller_rejected(migration):
    m = migration
    m.initialize()
    with m.state.owner("utm31-2023"):
        with pytest.raises(BusyError):
            m.transcode([31])
        with pytest.raises(BusyError):
            m.finalize()
    with m.state.owner("controller"), pytest.raises(BusyError):
        m.transcode([31])


def test_separate_processes_own_different_years(migration, tmp_path):
    m = migration
    m.initialize()
    metadata = {p: p.read_bytes() for p in Path(m.store.url).rglob("zarr.json")}
    code = """
import sys, time
from pathlib import Path
from geotessera.transcode import Migration
m = Migration(sys.argv[1])
year, barrier = int(sys.argv[2]), Path(sys.argv[3])
write = m.write_unit
def synchronized_write(*args, **kwargs):
    (barrier / str(year)).touch()
    deadline = time.monotonic() + 30
    while not (barrier / "go").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("test synchronization timed out")
        time.sleep(0.02)
    return write(*args, **kwargs)
m.write_unit = synchronized_write
sys.exit(m.transcode([31], years=[year], max_shards=1 if year == 2023 else None))
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, m.state.url, str(year), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for year in (2023, 2024)
    ]
    try:
        deadline = time.monotonic() + 20
        while not all((tmp_path / str(y)).exists() for y in (2023, 2024)):
            assert time.monotonic() < deadline, (
                "workers did not acquire independent year locks"
            )
            assert all(child.poll() is None for child in children)
            time.sleep(0.02)
        for year in (2023, 2024):
            with pytest.raises(BusyError):
                m.transcode([31], years=[year])
        with pytest.raises(BusyError):
            m.finalize()
        (tmp_path / "go").touch()
        for child, expected in zip(children, (RESUMABLE, 0)):
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == expected, stdout + stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    assert all(p.read_bytes() == value for p, value in metadata.items())
    assert {r["status"] for r in m.scan([31], [2024], checksum=True)} == {"complete"}
    assert "missing" in {r["status"] for r in m.scan([31], [2023])}
    with m.object_inventory(31, 2024):
        before = dict(m._object_index)
        assert before and all(k.split("/")[3] == "1" for k in before)
    assert m.transcode([31], years=[2023]) == 0
    with m.object_inventory(31, 2024):
        assert before == m._object_index
    m.finalize(full=True)


def test_worker_year_selection_is_validated_before_writes(migration):
    m = migration
    m.initialize()
    with pytest.raises(ValueError, match="requested years"):
        m.transcode([31], years=[2023, 2050])
    assert {r["status"] for r in m.scan()} == {"missing"}


def test_local_sigkill_owner_recovered(tmp_path):
    state = State(str(tmp_path / "state"))
    code = f"from geotessera._migration_state import State; import time; s=State({state.url!r});\nwith s.owner('utm31'):\n print('ready', flush=True)\n time.sleep(60)"
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(BusyError), state.owner("utm31"):
            pass
        child.kill()
        child.wait(timeout=10)
        with state.owner("utm31"):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_state_conditional_and_immutable(tmp_path):
    state = State(str(tmp_path))
    state.immutable("a.json", b"1")
    state.immutable("a.json", b"1")
    with pytest.raises(ValueError):
        state.immutable("a.json", b"2")
    with pytest.raises(BusyError):
        state.put("a.json", b"2", match="wrong")
    state.put("a.json", b"2", match=digest(b"1"))
    assert state.get("a.json")[0] == b"2"


def test_corrupt_inventory_rejected(migration):
    m = migration
    doc = m.zone(31)
    m.state.put(doc["inventory"], b"broken")
    with pytest.raises(ValueError, match="checksum"):
        list(m.units(doc))


def test_cli_help_and_plan_mode(migration):
    m = migration
    command = [sys.executable, "-m", "geotessera.registry_cli"]
    result = subprocess.run(
        command + ["zarr-init", "--state", m.state.url, "--output", m.store.url],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        command
        + [
            "zarr-transcode",
            "--state",
            m.state.url,
            "--zones",
            "31",
            "--years",
            "2023",
            "--max-shards",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == RESUMABLE, result.stdout + result.stderr


def test_cli_plan_initializes_destination(migration, tmp_path):
    state = tmp_path / "cli-state"
    destination = tmp_path / "cli-output.zarr"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "geotessera.registry_cli",
            "zarr-transcode-plan",
            migration.request["source"],
            "--state",
            str(state),
            "--destination",
            str(destination),
            "--snapshot-id",
            migration.plan["snapshot_id"],
            "--shard-size",
            "4",
            "--inner-chunk",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["initialized"] is True
    assert (destination / "zarr.json").is_file()
    assert State(str(state)).read("initialized.json") == {
        "migration_id": Migration(str(state)).id
    }


def replan(migration, tmp_path, mutate, arrays=None):
    session = migration.repo.writable_session("main")
    root = zarr.open_group(session.store)
    mutate(root)
    snapshot = session.commit("modified fixture")
    create_plan(
        migration.request["source"],
        str(tmp_path / "state2"),
        str(tmp_path / "out2"),
        snapshot_id=snapshot,
        shard_size=4,
        inner_chunk=2,
        arrays=arrays,
    )
    return Migration(str(tmp_path / "state2"))


def test_conflicting_overlap_rejected(migration, tmp_path):
    def mutate(root):
        root["31S/embeddings"][0, 0, 0, :] = 99

    m = replan(migration, tmp_path, mutate)
    m.initialize()
    with pytest.raises(ValueError, match="Conflicting hemisphere embeddings"):
        m.transcode([31])


def test_half_pixel_mismatch_rejected(migration, tmp_path):
    def mutate(root):
        a = root["31S/easting"]
        a[:] = a[:] + 5

    with pytest.raises(ValueError, match="pixel phases"):
        replan(migration, tmp_path, mutate)


def test_all_fill_unit_has_durable_receipt(migration, tmp_path):
    def mutate(root):
        root["31N/embeddings"].with_config({"write_empty_chunks": True})[
            0:1, 2:4, 6:8, :
        ] = np.zeros((1, 2, 2, 128), dtype="i1")
        root["31N/scales"].with_config({"write_empty_chunks": True})[2:4, 0:1, 6:8] = (
            np.full((2, 1, 2), np.nan, dtype="f4")
        )

    m = replan(migration, tmp_path, mutate, ["embeddings", "scales"])
    m.initialize()
    assert m.transcode([31]) == 0
    doc, unit = m.zone(31), (2023, 0, 1)
    receipt = m.state.read(m.receipt_key(31, unit))
    assert all(r["info"] is None for r in receipt["objects"].values())
    assert m.unit_status(doc, unit) == "complete"
    m.finalize(full=True)


def test_source_nodata_reader_status(migration):
    from geotessera import GeoTesseraZarr
    from geotessera.store import NODATA, VALID

    m = migration
    m.initialize()
    m.transcode([31])
    m.finalize()
    ds = GeoTesseraZarr(m.store.url).open_zone(zone=31)
    vector, status = ds.tessera.probe(500005, 45, 2023, search_px=0)
    assert status == VALID
    np.testing.assert_array_equal(vector, np.zeros(128))
    assert ds.tessera.probe(500065, 45, 2023, search_px=0)[1] == NODATA


def test_zarr_variant_does_not_publish_npy():
    from geotessera.registry import (
        dataset_path,
        published_datasets,
        published_zarr_datasets,
        zarr_store_url,
    )

    assert zarr_store_url("v1.1").endswith("/zarr/v1.1")
    assert zarr_store_url("v1.1", variant="dclimate").endswith("/zarr/v1.1-dclimate")
    assert not any(v == "dclimate" for _, v, _ in published_datasets())
    assert not any(v == "dclimate" for _, v, _ in published_zarr_datasets())
    with pytest.raises(ValueError, match="not yet published"):
        dataset_path("1.1", "dclimate")


def test_plan_resume_reuses_zone_inventory(migration, monkeypatch):
    from geotessera import icechunk_source, transcode

    m = migration
    m.state.delete("plan.json")

    async def unexpected(*args):
        raise AssertionError("Should reuse completed zone inventory")

    monkeypatch.setattr(icechunk_source, "inventory", unexpected)
    monkeypatch.setattr(transcode, "_implementation", lambda: "compatible-fix")
    plan = create_plan(
        m.request["source"],
        m.state.url,
        m.store.url,
        snapshot_id=m.plan["snapshot_id"],
        shard_size=4,
        inner_chunk=2,
    )
    assert plan["migration_id"] == m.id
    assert Migration(m.state.url).id == m.id
