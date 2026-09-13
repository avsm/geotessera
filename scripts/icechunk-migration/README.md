# Icechunk → GeoTessera Zarr on Fargate Spot

The migration state uses a sibling prefix in the public Source Cooperative
bucket. Use the same published version in this command and every worker image.

```sh
VERSION=REPLACE_WITH_VERSION
STATE=s3://us-west-2.opendata.source.coop/tessera/tessera/migrations/v1.1-dclimate

uvx --python 3.13 --from "geotessera[migration]==$VERSION" \
  geotessera-registry zarr-transcode-plan \
  s3://tessera-embeddings/v1.1/dclimate.icechunk \
  --snapshot-id QR7F41A6WYZ03VC92T6G \
  --state "$STATE" \
  --destination s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1.1-dclimate \
  --source-anon \
  --source-region us-west-2 \
  --store-region us-west-2 \
  --store-acl bucket-owner-full-control \
  --state-region us-west-2 \
  --state-acl bucket-owner-full-control
```

This one command inventories the pinned snapshot and initializes the Zarr v3
target. Repeating it with the same arguments repairs an interrupted run. It
preserves every source array, including Sentinel observation counts and
monthly coverage.

Run one Fargate Spot task per UTM/year. Each task receives ordinary command
arguments; it reads the state and writes receipts at `STATE`.

```sh
geotessera-registry zarr-transcode \
  --state "$STATE" \
  --zones 31 \
  --years 2024 \
  --workers 1 \
  --spill-dir /scratch \
  --source-anon \
  --source-region us-west-2 \
  --store-region us-west-2 \
  --store-acl bucket-owner-full-control \
  --state-region us-west-2 \
  --state-acl bucket-owner-full-control
```

Replace `31` and `2024` for each task. A repeated task skips committed shards
and repairs incomplete ones. Exit 0 means its UTM/year is complete. Exit 75
is retryable. Exit 2 is a configuration or data error.

The state prefix is required for restartability. It pins the Icechunk
snapshot and output layout, stores the sparse shard inventory, prevents two
attempts from owning the same UTM/year, and records completed shards. The
state is not a file to distribute to workers; `--state` is its S3 URL.

After every task succeeds, verify and consolidate the store.

```sh
uvx --python 3.13 --from "geotessera[migration]==$VERSION" \
  geotessera-registry zarr-consolidate \
  s3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1.1-dclimate \
  --state "$STATE" \
  --require-complete \
  --samples 8 \
  --source-anon \
  --source-region us-west-2 \
  --store-region us-west-2 \
  --store-acl bucket-owner-full-control \
  --state-region us-west-2 \
  --state-acl bucket-owner-full-control
```

Keep the Icechunk snapshot until finalization succeeds. The task role needs
read access to the source, Get/Put/List access to the state prefix, and
Get/Put/List access to a fresh target. Normal migration does not require
DeleteObject. Supply separate target credentials through
`GEOTESSERA_STORE_ACCESS_KEY_ID`, `GEOTESSERA_STORE_SECRET_ACCESS_KEY`, and
optionally `GEOTESSERA_STORE_SESSION_TOKEN` when the task role cannot write
to Source Cooperative.

The included Dockerfile builds the worker image for `linux/amd64` with pinned
migration dependencies. Give each task enough `/scratch` space for its shard
buffers. Start with one UTM/year before launching the full sweep.
