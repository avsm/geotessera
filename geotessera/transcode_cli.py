"""CLI integration, kept separate from the historical NPY registry commands."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter

from ._migration_state import BusyError


def add_commands(subparsers):
    from .registry_cli import _add_storage_args

    plan = subparsers.add_parser(
        "zarr-transcode-plan",
        help="Plan and initialize a snapshot-bound Icechunk migration",
        description="Pin a read-only Icechunk snapshot, inventory sparse shards, and initialize a fresh GeoTessera Zarr v3 destination. All arrays, including Sentinel counts and monthly coverage, are retained by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run this command once, then run zarr-transcode --state STATE --zones 31 --years 2024 on each worker. Source, destination and state prefixes must be separate. Repeating the identical planning command repairs an interrupted initialization; changing selection or layout needs fresh prefixes. Keep the pinned snapshot retained until verification finishes.",
    )
    plan.add_argument(
        "source", help="Icechunk repository (local directory or s3://bucket/prefix)"
    )
    plan.add_argument(
        "--state",
        dest="output",
        metavar="STATE",
        required=True,
        help="Durable migration-state directory (local or S3)",
    )
    plan.add_argument(
        "--destination", required=True, help="New Zarr destination (local or S3)"
    )
    selection = plan.add_mutually_exclusive_group()
    selection.add_argument("--snapshot-id", help="Exact immutable snapshot to read")
    selection.add_argument(
        "--branch", default="main", help="Resolve this branch once when first planning"
    )
    plan.add_argument(
        "--zones", help="Comma-separated UTM numbers/ranges, e.g. 1-3,31; omit for all"
    )
    plan.add_argument(
        "--years", help="Comma-separated years/ranges, e.g. 2023-2024; omit for all"
    )
    plan.add_argument(
        "--arrays",
        help="Comma-separated arrays; default all. embeddings/scales always included",
    )
    plan.add_argument(
        "--shard-size",
        type=int,
        default=4096,
        help="Spatial pixels per output shard edge",
    )
    plan.add_argument(
        "--inner-chunk",
        type=int,
        default=32,
        help="Spatial pixels per inner chunk edge; must divide shard size",
    )
    _add_storage_args(plan, "source", "Icechunk source")
    _add_storage_args(plan, "store", "Output store", writable=True)
    _add_storage_args(plan, "state", "Migration state", writable=True)
    plan.set_defaults(func=run)

    worker = subparsers.add_parser(
        "zarr-transcode",
        help="Resume missing shards of a planned migration",
        description="Read the state's pinned Icechunk snapshot and repair missing or incomplete shards. Each shard commits only after all selected arrays are written. Different UTM/year pairs can run in separate processes; duplicate owners are refused.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: zarr-transcode --state STATE --zones 31 --years 2024 --workers 1 --spill-dir /scratch. Exit 0 means the selected years are complete, 75 means incomplete/busy/retryable, and 2 means a configuration or data error. SIGTERM stops new work; completed shard receipts survive SIGKILL. Finish with zarr-consolidate STORE --state STATE --require-complete. Source/store/state credentials may be supplied through GEOTESSERA_{SOURCE,STORE,STATE}_{ACCESS_KEY_ID,SECRET_ACCESS_KEY,SESSION_TOKEN} environment variables.",
    )
    worker.add_argument(
        "--state",
        dest="plan",
        metavar="STATE",
        required=True,
        help="Migration-state directory",
    )
    worker.add_argument(
        "--zones",
        required=True,
        help="UTM numbers/ranges within the plan, e.g. 31 or 30-32",
    )
    worker.add_argument(
        "--years",
        help="Years within the plan; separate processes may own different years of one zone",
    )
    worker.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent shard buffers; size against RAM",
    )
    worker.add_argument(
        "--spill-dir",
        help="Disposable directory for shard buffers; progress stays in STATE",
    )
    worker.add_argument(
        "--max-shards",
        type=int,
        help="Maximum new shards per attempt; unfinished selection exits 75",
    )
    worker.add_argument(
        "--max-seconds",
        type=float,
        help="Soft admission deadline in seconds; in-flight writes may finish later",
    )
    worker.add_argument(
        "--read-rows", type=int, default=256, help="Source strip height in pixels"
    )
    worker.add_argument(
        "--io-concurrency",
        type=int,
        default=4,
        help="Bound Zarr asynchronous I/O and codec threads",
    )
    for prefix in ("source", "store", "state"):
        _add_storage_args(worker, prefix, prefix.title(), writable=prefix != "source")
    worker.set_defaults(func=run)

    for name in ("zarr-init", "zarr-scan", "zarr-verify", "zarr-consolidate"):
        parser = subparsers.choices[name]
        parser.epilog = {
            "zarr-init": "Icechunk mode: zarr-init --state STATE --output STORE. Omit base_dir and --years: source, years, grids, arrays and nodata are fixed by the state. Repeating repairs matching partial initialization; conflicting metadata is refused.",
            "zarr-scan": "Icechunk mode: zarr-scan STORE --state STATE --output scan.parquet. Reports missing/partial/complete shard receipts for every retained array; --checksums reads and hashes encoded objects. --output exports Parquet; missing or partial work exits 75.",
            "zarr-verify": "Icechunk mode: zarr-verify STORE --state STATE --samples 8. Omit base_dir. Samples are complete shards per zone/year, plus boundary/equatorial shards; --full compares every shard and encoded checksum. --output writes JSON. Use --spill-dir to limit in-memory buffers.",
            "zarr-consolidate": "Icechunk mode: zarr-consolidate STORE --state STATE --require-complete. Requires complete receipts and source verification, then consolidates under exclusive ownership. --full verifies all shards; --samples is per zone/year. Active workers prevent finalization.",
        }[name]
        parser.add_argument(
            "--state",
            dest="plan",
            metavar="STATE",
            help="Use Icechunk migration state instead of NPY registries",
        )
        _add_storage_args(parser, "state", "Migration state", writable=True)
        if name == "zarr-verify":
            parser.add_argument(
                "--full", action="store_true", help="Verify every migration shard"
            )
            parser.add_argument("--spill-dir")
        if name == "zarr-scan":
            parser.add_argument(
                "--checksums",
                action="store_true",
                help="Read/hash encoded objects as well as checking their identities",
            )
        if name == "zarr-consolidate":
            parser.add_argument(
                "--require-complete",
                action="store_true",
                help="Require migration coverage and source verification",
            )
            parser.add_argument(
                "--samples", type=int, default=8, help="Migration samples per zone/year"
            )
            parser.add_argument("--full", action="store_true")
            parser.add_argument("--spill-dir")
            _add_storage_args(parser, "source", "Icechunk source")
        original = parser.get_default("func")

        def dispatch(args, original=original, parser=parser):
            if args.command == "zarr-verify" and args.samples is None:
                args.samples = 8 if args.plan else 1000
            if args.plan:
                return run(args)
            if args.command == "zarr-init" and (not args.base_dir or not args.years):
                parser.error("base_dir and --years are required without --state")
            if args.command == "zarr-verify" and not args.base_dir:
                parser.error("base_dir is required without --state")
            if getattr(args, "require_complete", False):
                parser.error("--require-complete requires --state")
            return original(args)

        parser.set_defaults(func=dispatch)


def run(args):
    from . import transcode
    from .registry_cli import _parse_int_range, _storage_options_for

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def options(prefix, url):
        result = _storage_options_for(args, prefix, url)
        if str(url).startswith("s3://"):
            result = dict(result or {})
            for env, key in (
                ("ACCESS_KEY_ID", "key"),
                ("SECRET_ACCESS_KEY", "secret"),
                ("SESSION_TOKEN", "token"),
            ):
                value = os.environ.get(f"GEOTESSERA_{prefix.upper()}_{env}")
                if value:
                    result[key] = value
        return result

    try:
        zones = _parse_int_range(args.zones) if getattr(args, "zones", None) else None
        years = _parse_int_range(args.years) if getattr(args, "years", None) else None
        if args.command == "zarr-transcode-plan":
            plan = transcode.create_plan(
                args.source,
                args.output,
                args.destination,
                snapshot_id=args.snapshot_id,
                branch=args.branch,
                zones=zones,
                years=years,
                arrays=args.arrays.split(",") if args.arrays else None,
                shard_size=args.shard_size,
                inner_chunk=args.inner_chunk,
                source_options=options("source", args.source),
                state_options=options("state", args.output),
            )
            transcode.Migration(
                args.output,
                destination=args.destination,
                source_options=options("source", args.source),
                state_options=options("state", args.output),
                store_options=options("store", args.destination),
            ).initialize()
            print(
                json.dumps(
                    {
                        "initialized": True,
                        "migration_id": plan["migration_id"],
                        "snapshot_id": plan["snapshot_id"],
                        "zones": len(plan["partitions"]),
                        "units": sum(p["units"] for p in plan["partitions"].values()),
                    }
                )
            )
            return 0
        # Read routing fields before constructing independent credential chains.
        from ._migration_state import State

        state_url = args.plan.removesuffix("/plan.json")
        request = State(state_url, options("state", state_url)).read("plan.json")[
            "request"
        ]
        destination = (
            args.output
            if args.command == "zarr-init"
            else getattr(args, "store_path", None)
        )
        if getattr(args, "base_dir", None):
            raise ValueError("Do not supply an NPY tile source with --state")
        if getattr(args, "state_url", None) not in (None, state_url):
            raise ValueError("--state-url must match --state for a migration")
        migration = transcode.Migration(
            state_url,
            state_options=options("state", state_url),
            source_options=options("source", request["source"]),
            store_options=options("store", request["destination"]),
            destination=destination,
        )
        if args.command == "zarr-init":
            if (
                years is not None
                or args.no_landmask
                or args.matryoshka_depths
                or args.stretch_sample_size
            ):
                raise ValueError(
                    "Migration schema/years are fixed in the plan; omit NPY initialization options"
                )
            migration.initialize()
            return 0
        if args.command == "zarr-transcode":
            with transcode.termination_event() as stop:
                return migration.transcode(
                    zones,
                    years=years,
                    workers=args.workers,
                    spill_dir=args.spill_dir,
                    max_shards=args.max_shards,
                    max_seconds=args.max_seconds,
                    read_rows=args.read_rows,
                    io_concurrency=args.io_concurrency,
                    stop=stop,
                )
        if args.command == "zarr-scan":
            counts = Counter()
            with tempfile.TemporaryDirectory(prefix="transcode-scan-") as tmp:
                import pyarrow as pa
                import pyarrow.parquet as pq

                path = tmp + "/scan.parquet"
                schema = pa.schema(
                    [(n, pa.int32()) for n in ("zone", "year", "row", "col")]
                    + [("status", pa.string())]
                )
                with pq.ParquetWriter(path, schema, compression="zstd") as writer:
                    batch = []
                    for row in migration.scan(zones, years, checksum=args.checksums):
                        counts[row["status"]] += 1
                        batch.append(row)
                        if len(batch) == 4096:
                            writer.write_table(
                                pa.Table.from_pylist(batch, schema=schema)
                            )
                            batch.clear()
                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                if args.output:
                    from . import remote

                    fs = remote.get_fs(args.output, options("state", args.output))
                    if fs is None:
                        import shutil
                        from pathlib import Path

                        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, args.output)
                    else:
                        fs.put_file(path, args.output)
            print(json.dumps(dict(counts), sort_keys=True))
            return transcode.RESUMABLE if counts["missing"] or counts["partial"] else 0
        if args.command == "zarr-verify":
            report = migration.verify(
                zones,
                years,
                samples=args.samples,
                seed=args.seed,
                full=args.full,
                spill_dir=args.spill_dir,
            )
            if args.output:
                from . import remote

                remote.write_bytes(
                    args.output,
                    json.dumps(report).encode(),
                    options("state", args.output),
                )
            print(json.dumps(report, sort_keys=True))
            return 0
        report = migration.finalize(
            samples=args.samples, full=args.full, spill_dir=args.spill_dir
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as e:  # noqa: BLE001 - report_error re-raises unexpected errors
        return report_error(e)


def report_error(error):
    """Use the same exit codes for the controller and registry worker."""
    if isinstance(error, (ImportError, ValueError, FileNotFoundError, PermissionError)):
        retry = False
    elif isinstance(error, (BusyError, OSError)):
        retry = True
    else:
        import icechunk
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(error, ClientError):
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            code = error.response.get("Error", {}).get("Code", "")
            retry = status >= 500 or code in (
                "SlowDown",
                "RequestTimeout",
                "Throttling",
                "ThrottlingException",
            )
        elif isinstance(error, icechunk.StorageError):
            # Icechunk's storage wrapper does not expose an HTTP status.
            retry = not any(
                code in str(error)
                for code in (
                    "AccessDenied",
                    "InvalidAccessKeyId",
                    "SignatureDoesNotMatch",
                    "Forbidden",
                    "status: 403",
                )
            )
        elif isinstance(error, icechunk.IcechunkError):
            retry = False
        elif isinstance(error, BotoCoreError):
            retry = type(error).__name__ in (
                "EndpointConnectionError",
                "ConnectionClosedError",
                "ReadTimeoutError",
                "ConnectTimeoutError",
            )
        else:
            raise error
    print(
        f"Migration {'error (retryable)' if retry else 'error'}: {error}",
        file=sys.stderr,
    )
    return 75 if retry else 2
