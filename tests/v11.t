GeoTessera v1.1 / Variant Tests
================================

Tests that exercise the new ``--dataset-version v1.1`` and
``--dataset-variant`` plumbing without doing any heavy downloads. We only
fetch the (small, cached) per-version manifest and assert on the metadata
the client reports back.

The library-mode CLI renders its output through Rich, which wraps long
lines based on the (unknown) terminal width. So for anything we need to
exactly compare — like the full year list — we drive ``Registry`` directly
from Python.

Setup
-----

  $ export TERM=dumb
  $ export XDG_CACHE_HOME="$CRAMTMP/cache"
  $ mkdir -p "$XDG_CACHE_HOME"

Test: Version Aliases (v1 == 1.0 == v1.0)
-----------------------------------------

``--dataset-version 1.0`` is accepted as an alias for ``v1`` (legacy S3 path
uses ``v1/``; the normalised version is ``1.0``):

  $ uv run python -c "
  > from geotessera.registry import _parse_dataset_version
  > for s in ('v1', '1.0', 'v1.0', '1', 'v1.1', '1.1'):
  >     vpath, vnorm = _parse_dataset_version(s)
  >     print(f'{s!r:8s} -> path={vpath!r:6s} norm={vnorm!r}')
  > "
  'v1'     -> path='v1'   norm='1.0'
  '1.0'    -> path='v1'   norm='1.0'
  'v1.0'   -> path='v1'   norm='1.0'
  '1'      -> path='v1'   norm='1.0'
  'v1.1'   -> path='v1.1' norm='1.1'
  '1.1'    -> path='v1.1' norm='1.1'

Test: v1 / vultr Year Range via Library
---------------------------------------

The legacy v1/vultr dataset covers 2017 - 2025:

  $ uv run python -c "
  > from geotessera import GeoTessera
  > import tempfile, os
  > with tempfile.TemporaryDirectory() as d:
  >     gt = GeoTessera(dataset_version='v1', dataset_variant='vultr',
  >                     cache_dir=os.path.join(d, 'c'), embeddings_dir=os.path.join(d, 'e'))
  >     print(sorted(gt.registry.get_available_years()))
  > "
  [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

Test: v1.1 / Cambridge Year Range via Library
---------------------------------------------

The Cambridge model in v1.1 ships a broader year range (2015 - 2025). The
filter applied at load time pulls the v1.1 manifest, not v1's:

  $ uv run python -c "
  > from geotessera import GeoTessera
  > import tempfile, os
  > with tempfile.TemporaryDirectory() as d:
  >     gt = GeoTessera(dataset_version='v1.1', dataset_variant='cambridge',
  >                     cache_dir=os.path.join(d, 'c'), embeddings_dir=os.path.join(d, 'e'))
  >     print(sorted(gt.registry.get_available_years()))
  > "
  [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

The numeric form ``1.1`` is equivalent:

  $ uv run python -c "
  > from geotessera import GeoTessera
  > import tempfile, os
  > with tempfile.TemporaryDirectory() as d:
  >     gt = GeoTessera(dataset_version='1.1', dataset_variant='cambridge',
  >                     cache_dir=os.path.join(d, 'c'), embeddings_dir=os.path.join(d, 'e'))
  >     print(sorted(gt.registry.get_available_years()))
  > "
  [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

Test: Bad Variant Raises a Clear ValueError
-------------------------------------------

Filtering to a ``(version, variant)`` combination that doesn't exist in the
manifest must fail loudly, not silently render an empty dataset:

  $ uv run python -c "
  > from geotessera import GeoTessera
  > import tempfile, os
  > with tempfile.TemporaryDirectory() as d:
  >     try:
  >         GeoTessera(dataset_version='v1', dataset_variant='nonexistent',
  >                    cache_dir=os.path.join(d, 'c'), embeddings_dir=os.path.join(d, 'e'))
  >     except ValueError as e:
  >         msg = str(e).splitlines()[0]
  >         print('ValueError:', msg)
  > " 2>&1 | grep -F 'ValueError: Manifest has no rows'
  ValueError: Manifest has no rows for version=1.0, variant=nonexistent. Check the dataset_version and dataset_variant arguments.

Test: Manifest URLs Are Per-Version
------------------------------------

Confirm the consumer fetches from the per-version path on S3:

  $ uv run python -c "
  > from geotessera.registry import Registry
  > import tempfile, os
  > with tempfile.TemporaryDirectory() as d:
  >     r1 = Registry(version='v1', variant='vultr',
  >                   cache_dir=os.path.join(d, 'c1'),
  >                   embeddings_dir=os.path.join(d, 'e1'))
  >     r2 = Registry(version='v1.1', variant='cambridge',
  >                   cache_dir=os.path.join(d, 'c2'),
  >                   embeddings_dir=os.path.join(d, 'e2'))
  > print('v1 URL :', r1._registry_url)
  > print('v1.1   :', r2._registry_url)
  > " 2>&1 | grep -E '^v1'
  v1 URL : https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1/manifest.parquet
  v1.1   : https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1.1/manifest.parquet

Test: S3 Embeddings Subdir Reflects Variant
-------------------------------------------

The S3 path uses the bare ``global_0.1_degree_representation`` dir for the
default ``vultr`` variant and adds a ``.<variant>`` suffix otherwise. The
local filesystem layout always uses the bare name (variant info lives in
the ``tessera_metadata.json`` sidecar):

  $ uv run python -c "
  > from geotessera.registry import _variant_subdir
  > print('vultr    ->', _variant_subdir('vultr'))
  > print('cambridge->', _variant_subdir('cambridge'))
  > "
  vultr    -> global_0.1_degree_representation
  cambridge-> global_0.1_degree_representation.cambridge

Test: Tile Download Dry-Run Accepts v1.1 / cambridge Flags
----------------------------------------------------------

A dry-run download against v1.1/cambridge should accept the flags without
error and resolve the requested point to a tile. NPY format produces three
files per tile (embedding + scales + landmask):

  $ geotessera download \
  >   --tile "0.35,51.65" \
  >   --year 2024 \
  >   --format npy \
  >   --dry-run \
  >   --dataset-version v1.1 \
  >   --dataset-variant cambridge 2>&1 | grep -oE '(Point \(0\.35, 51\.65\) -> tile grid_0\.35_51\.65|Found [0-9]+ tiles for region in year 2024|Files to download: +[0-9]+|Tiles in region: +[0-9]+)' | tr -s ' '
  Point (0.35, 51.65) -> tile grid_0.35_51.65
  Found 1 tiles for region in year 2024
  Files to download: 3
  Tiles in region: 1

Test: Tessera Metadata Sidecar Helper
-------------------------------------

The ``write_tessera_metadata`` helper records variant + version provenance
into a JSON sidecar so downstream tools can recover which dataset produced
the tiles (since the local dir layout is variant-agnostic):

  $ uv run python -c "
  > import tempfile, json
  > from geotessera.registry import write_tessera_metadata, TESSERA_METADATA_FILENAME
  > with tempfile.TemporaryDirectory() as d:
  >     p = write_tessera_metadata(d, dataset_version='v1.1', dataset_variant='cambridge',
  >                                 extra={'format': 'npy', 'year': 2024, 'tile_count': 1})
  >     payload = json.loads(p.read_text())
  > for k in ('dataset_version', 'dataset_version_path', 'dataset_variant',
  >           'embeddings_subdir', 's3_embeddings_subdir', 'format', 'year', 'tile_count'):
  >     print(f'{k}: {payload[k]}')
  > print(f'filename: {TESSERA_METADATA_FILENAME}')
  > "
  dataset_version: 1.1
  dataset_version_path: v1.1
  dataset_variant: cambridge
  embeddings_subdir: global_0.1_degree_representation
  s3_embeddings_subdir: global_0.1_degree_representation.cambridge
  format: npy
  year: 2024
  tile_count: 1
  filename: tessera_metadata.json
