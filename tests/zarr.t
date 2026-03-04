GeoTessera Zarr Format Tests
=============================

These are tests for the zarr format support added in this branch.

Setup
-----

Set environment variable to disable fancy terminal output (ANSI codes, boxes, colors):

  $ export TERM=dumb

Create a temporary directory for test outputs and cache:

  $ export TESTDIR="$CRAMTMP/test_outputs"
  $ mkdir -p "$TESTDIR"

Override XDG cache directory to use temporary location (for test isolation):

  $ export XDG_CACHE_HOME="$CRAMTMP/cache"
  $ mkdir -p "$XDG_CACHE_HOME"

Test: Zarr Format Validation
-----------------------------

Test that zarr format is recognized as a valid option:

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format zarr \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E '^.*Format:.*ZARR' | head -1 | sed 's/ *$//'
   Format:          ZARR

Test: Invalid Format Rejected
------------------------------

Test that invalid formats are properly rejected:

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format invalid \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E "Invalid format.*Must be"
  Error: Invalid format 'invalid'. Must be 'tiff', 'npy' or 'zarr'

Test: Download Dry Run for UK Tile (Zarr format)
-------------------------------------------------

Test dry-run with zarr format to verify it's processed correctly:

  $ geotessera download \
  >   --bbox "-0.1,51.3,0.1,51.5" \
  >   --year 2024 \
  >   --format zarr \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E '(Format|Year|Dataset version|Found|Tiles in region)' | sed 's/ *$//'
   Format:          ZARR
   Year:            2024
   Dataset version: v1
  Found 16 tiles for region in year 2024
   Tiles in region:     16
   Year:                2024
   Format:              ZARR

Test: Download Cambridge Tiles in Zarr Format
----------------------------------------------

Download a small region in zarr format (4 tiles for faster testing):

  $ geotessera download \
  >   --bbox "0.086174,52.183432,0.151062,52.206318" \
  >   --year 2024 \
  >   --format zarr \
  >   --output "$TESTDIR/cb_tiles_zarr" \
  >   --dataset-version v1 2>&1 | grep -E '(SUCCESS|Found.*tiles)' | sed 's/ *$//'
  Found 4 tiles for region in year 2024
  SUCCESS: Exported 4 zarr archives

Verify zarr archives were created in the registry structure:

  $ [ -n "$(find "$TESTDIR/cb_tiles_zarr/global_0.1_degree_representation/2024" -name "*.zarr" 2>/dev/null)" ] && echo "Zarr archives created"
  Zarr archives created

  $ find "$TESTDIR/cb_tiles_zarr/global_0.1_degree_representation/2024" -name "*.zarr" | wc -l | tr -d ' '
  4

Test: Info Command on Downloaded Zarr Tiles
--------------------------------------------

Test the info command on the downloaded zarr tiles.
Note that the info command may detect NPY files that are created alongside zarr:

  $ geotessera info --tiles "$TESTDIR/cb_tiles_zarr" 2>&1 | grep -E 'Total tiles|Format|Years' | sed 's/ *$//'
   Total tiles: 4
   Format:      NPY
   Years:       2024

Test: Zarr Archive Structure
-----------------------------

Verify that a zarr archive can be opened and contains expected data:

  $ ZARR_FILE=$(find "$TESTDIR/cb_tiles_zarr/global_0.1_degree_representation/2024" -name "*.zarr" | head -1)
  $ uv run python -c "
  > import xarray as xr
  > ds = xr.open_dataset('$ZARR_FILE', decode_coords='all')
  > print(f'Variables: {list(ds.data_vars.keys())}')
  > coords = sorted([c for c in ds.coords.keys() if c != 'spatial_ref'])
  > print(f'Coordinates: {coords}')
  > print(f'CRS present: {hasattr(ds, \"rio\") and ds.rio.crs is not None}')
  > print(f'Transform present: {hasattr(ds, \"rio\") and ds.rio.transform() is not None}')
  > "
  Variables: ['embedding']
  Coordinates: ['band', 'x', 'y']
  CRS present: True
  Transform present: True

Test: Band Selection with Zarr Format
--------------------------------------

Download zarr tiles with specific band selection:

  $ geotessera download \
  >   --bbox "0.086174,52.183432,0.151062,52.206318" \
  >   --year 2024 \
  >   --format zarr \
  >   --bands "0,1,2" \
  >   --output "$TESTDIR/cb_tiles_zarr_bands" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS' | sed 's/ *$//'
  SUCCESS: Exported 4 zarr archives

Verify band count in band-selected zarr archive:

  $ ZARR_FILE=$(find "$TESTDIR/cb_tiles_zarr_bands/global_0.1_degree_representation/2024" -name "*.zarr" | head -1)
  $ uv run python -c "
  > import xarray as xr
  > ds = xr.open_dataset('$ZARR_FILE', decode_coords='all')
  > print(f'Band count: {len(ds.band)}')
  > "
  Band count: 3

Test: Visualization with Zarr Format
-------------------------------------

Create a PCA visualization from the downloaded zarr tiles.
Note that tiles may be detected as npy format since both formats coexist:

  $ geotessera visualize "$TESTDIR/cb_tiles_zarr" "$TESTDIR/pca_zarr.tif" 2>&1 | grep -A 1 -E 'Found|Created PCA mosaic' | sed 's/ *$//'
  Found 4 tiles (npy format)
  Combined data shape: (3317086, 128)
  --
  Created PCA mosaic:
  * (glob)

Verify the PCA visualization file was created:

  $ test -f "$TESTDIR/pca_zarr.tif" && echo "PCA visualization created"
  PCA visualization created

Test: CLI Help Shows Zarr Format
---------------------------------

Verify that the CLI help text mentions zarr as a format option:

  $ geotessera download --help | grep -i zarr | head -1
  *zarr* (glob)

Test: Global Preview Store Structure
-------------------------------------

Build a zone store with RGB from the Cambridge tiles:

  $ geotessera-registry zarr-build \
  >   "$TESTDIR/cb_tiles_zarr" \
  >   --output-dir "$TESTDIR/zarr_global_test" \
  >   --year 2024 \
  >   --rgb 2>&1 | grep -E '(RGB preview|Zone)' | head -3 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)

Build global preview store from the zone store:

  $ geotessera-registry global-preview \
  >   "$TESTDIR/zarr_global_test" \
  >   --year 2024 \
  >   --levels 3 \
  >   --workers 2 2>&1 | grep -E '(Building|Found|Processing|Global store)' | head -4 | sed 's/ *$//'
  * (glob)
  * (glob)
  * (glob)
  * (glob)

Verify global preview store structure has multiscales metadata:

  $ uv run python -c "
  > import json
  > with open('$TESTDIR/zarr_global_test/global_rgb_2024.zarr/zarr.json') as f:
  >     meta = json.load(f)
  > ms = meta['attributes']['multiscales']
  > print(f'crs: {ms[\"crs\"]}')
  > print(f'num_levels: {len(ms[\"layout\"])}')
  > print(f'has_consolidated: {\"consolidated_metadata\" in meta}')
  > cm = meta['consolidated_metadata']['metadata']
  > has_rgb = any('rgb' in k for k in cm)
  > print(f'has_rgb_arrays: {has_rgb}')
  > first_arr = [v for k, v in cm.items() if 'rgb' in k][0]
  > has_blosc = any(c.get('name') == 'blosc' for c in first_arr.get('codecs', []))
  > print(f'has_blosc: {has_blosc}')
  > print(f'dimension_names: {first_arr.get(\"dimension_names\")}')
  > sp = meta['attributes']['spatial']
  > print(f'bounds: {sp[\"bounds\"]}')
  > "
  crs: EPSG:4326
  num_levels: 3
  has_consolidated: True
  has_rgb_arrays: True
  has_blosc: True
  dimension_names: ['lat', 'lon', 'band']
  bounds: [-180.0, -90.0, 180.0, 90.0]
