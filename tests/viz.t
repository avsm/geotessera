GeoTessera Visualization Tests
===============================

These are tests for the `geotessera visualize` and `geotessera webmap` commands.

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

Test: calculate_bbox_from_file handles CRS correctly
------------------------------------------------------

Test with a GeoJSON that has no CRS metadata (should assume WGS84):

  $ uv run python -c "
  > import json, tempfile, os
  > from geotessera.visualization import calculate_bbox_from_file
  > geojson = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0.08, 52.18], [0.15, 52.18], [0.15, 52.21], [0.08, 52.21], [0.08, 52.18]]]}, 'properties': {}}]}
  > path = os.path.join('$CRAMTMP', 'no_crs.geojson')
  > with open(path, 'w') as f: json.dump(geojson, f)
  > bbox = calculate_bbox_from_file(path)
  > print(f'{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}')
  > "
  0.08,52.18,0.15,52.21

Test with a geometry file reprojected to UTM (should reproject back to WGS84):

  $ uv run python -c "
  > import geopandas as gpd
  > from shapely.geometry import box
  > from geotessera.visualization import calculate_bbox_from_file
  > import os
  > gdf = gpd.GeoDataFrame(geometry=[box(0.08, 52.18, 0.15, 52.21)], crs='EPSG:4326')
  > gdf_utm = gdf.to_crs('EPSG:32631')
  > path = os.path.join('$CRAMTMP', 'utm.geojson')
  > gdf_utm.to_file(path, driver='GeoJSON')
  > bbox = calculate_bbox_from_file(path)
  > print(f'{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}')
  > "
  0.08,52.18,0.15,52.21

Test: Download Tiles for Cambridge Region (Bbox)
-------------------------------------------------

Download tiles covering a small area of Cambridge using a bounding box.
This bbox covers just 4 tiles for faster testing:

  $ geotessera download --source tiles \
  >   --bbox "0.086174,52.183432,0.151062,52.206318" \
  >   --year 2024 \
  >   --format tiff \
  >   --output "$TESTDIR/cb_tiles_tiff" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS|Found.*tiles' | sed 's/ *$//'
  Found 4 tiles for region in year 2024
  SUCCESS: Exported 4 GeoTIFF files

Verify TIFF files were created:

  $ [ -n "$(find "$TESTDIR/cb_tiles_tiff/global_0.1_degree_representation/2024" -name "*.tif*" 2>/dev/null)" ] && echo "TIFF files created"
  TIFF files created

Verify NPY files were also created (intermediate format retained for efficient reprocessing):

  $ [ -n "$(find "$TESTDIR/cb_tiles_tiff/global_0.1_degree_representation/2024" -name "grid_*.npy" ! -name "*_scales.npy" 2>/dev/null)" ] && echo "NPY embedding files created"
  NPY embedding files created

  $ [ -n "$(find "$TESTDIR/cb_tiles_tiff/global_0.1_degree_representation/2024" -name "*_scales.npy" 2>/dev/null)" ] && echo "NPY scales files created"
  NPY scales files created

Verify both formats coexist (count files of each type):

  $ find "$TESTDIR/cb_tiles_tiff/global_0.1_degree_representation/2024" -name "*.tif*" 2>/dev/null | wc -l | tr -d ' '
  4

  $ find "$TESTDIR/cb_tiles_tiff/global_0.1_degree_representation/2024" -name "grid_*.npy" ! -name "*_scales.npy" 2>/dev/null | wc -l | tr -d ' '
  4

Test: Visualize - Create PCA Mosaic from TIFF Files
----------------------------------------------------

Create a PCA visualization from the downloaded TIFF files.
The visualize command should:
1. Load all TIFF tiles
2. Apply PCA to reduce 128 channels to RGB
3. Create a mosaic in the target CRS (default EPSG:3857)

  $ geotessera visualize \
  >   "$TESTDIR/cb_tiles_tiff" \
  >   "$TESTDIR/cb_pca_mosaic.tif" 2>&1 | grep -A 1 -E 'Found|Created PCA mosaic' | sed 's/ *$//'
  Found 4 tiles (npy format)
  PCA fit sample: (100000, 128)
  --
  Created PCA mosaic:
  * (glob)

Verify PCA mosaic was created:

  $ [ -f "$TESTDIR/cb_pca_mosaic.tif" ] && echo "PCA mosaic created"
  PCA mosaic created

Check that it's a valid GeoTIFF with 3 bands (RGB):

  $ uv run python -c "import rasterio; r = rasterio.open('$TESTDIR/cb_pca_mosaic.tif'); print(f'Bands: {r.count}, CRS: {r.crs}')"
  Bands: 3, CRS: EPSG:3857

Test: Visualize - Custom CRS and Balance Options
-------------------------------------------------

Test creating a visualization with custom CRS and histogram balancing:

  $ geotessera visualize \
  >   "$TESTDIR/cb_tiles_tiff" \
  >   "$TESTDIR/cb_pca_4326.tif" \
  >   --crs EPSG:4326 \
  >   --balance histogram 2>&1 | grep -A 1 -E 'Created PCA mosaic' | sed 's/ *$//'
  Created PCA mosaic:
  * (glob)

Verify custom CRS mosaic was created:

  $ [ -f "$TESTDIR/cb_pca_4326.tif" ] && echo "Custom CRS mosaic created"
  Custom CRS mosaic created

  $ uv run python -c "import rasterio; r = rasterio.open('$TESTDIR/cb_pca_4326.tif'); print(f'CRS: {r.crs}')"
  CRS: EPSG:4326

Test: Visualize - NPY Format Input
-----------------------------------

Download the same region in NPY format and create a visualization:

  $ geotessera download --source tiles \
  >   --bbox "0.086174,52.183432,0.151062,52.206318" \
  >   --year 2024 \
  >   --format npy \
  >   --output "$TESTDIR/cb_tiles_npy" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS' | sed 's/ *$//'
  SUCCESS: Downloaded 4 tiles (12 files, 417.6 MB)

Create visualization from NPY format:

  $ geotessera visualize \
  >   "$TESTDIR/cb_tiles_npy" \
  >   "$TESTDIR/cb_pca_from_npy.tif" 2>&1 | grep -A 1 -E 'Found|Created PCA mosaic' | sed 's/ *$//'
  Found 4 tiles (npy format)
  PCA fit sample: (100000, 128)
  --
  Created PCA mosaic:
  * (glob)

Verify NPY-based mosaic was created:

  $ [ -f "$TESTDIR/cb_pca_from_npy.tif" ] && echo "NPY format mosaic created"
  NPY format mosaic created

Test: Webmap - Generate Web Tiles and Viewer
---------------------------------------------

Generate web tiles from the PCA mosaic and create an interactive web viewer.
This should:
1. Reproject the mosaic if needed (to EPSG:3857 for web)
2. Generate XYZ web tiles at multiple zoom levels
3. Create an HTML viewer with Leaflet

  $ geotessera webmap \
  >   "$TESTDIR/cb_pca_mosaic.tif" \
  >   --output "$TESTDIR/cb_webmap" \
  >   --min-zoom 10 \
  >   --max-zoom 13 2>&1 | grep -A 1 -E 'Web visualization ready|Created web' | grep -v '^--$' | sed 's/ *$//'
  Web visualization ready in:
  * (glob)
  Created web tiles in:
  * (glob)
  Created web viewer:
  * (glob)

Verify web map directory structure was created:

  $ test -d "$TESTDIR/cb_webmap" && echo "Web map directory created"
  Web map directory created

  $ test -f "$TESTDIR/cb_webmap/viewer.html" && echo "HTML viewer created"
  HTML viewer created

  $ [ -n "$(find "$TESTDIR/cb_webmap/tiles" -name "*.png" 2>/dev/null)" ] && echo "Web tiles (PNG) created"
  Web tiles (PNG) created

Check that tiles exist at multiple zoom levels:

  $ find "$TESTDIR/cb_webmap/tiles" -type d -name "1[0-3]" | wc -l | tr -d ' ' | grep -E '[2-4]'
  4

Test: Webmap - Custom Output and Settings
------------------------------------------

Test webmap with custom initial zoom and center:

  $ geotessera webmap \
  >   "$TESTDIR/cb_pca_4326.tif" \
  >   --output "$TESTDIR/cb_webmap_custom" \
  >   --min-zoom 10 \
  >   --max-zoom 12 \
  >   --initial-zoom 11 2>&1 | grep -A 1 -E 'Web visualization ready' | sed 's/ *$//'
  Web visualization ready in:
  * (glob)

Verify custom web map was created:

  $ test -f "$TESTDIR/cb_webmap_custom/viewer.html" && echo "Custom web map created"
  Custom web map created

Test: Info Command on Visualization Outputs
--------------------------------------------

Test that info command works on the created PCA mosaics:

  $ geotessera info --tiles "$TESTDIR/cb_tiles_tiff" 2>&1 | grep -E 'Total tiles|Format|Years' | sed 's/ *$//'
   Total tiles: 4
   Format:      GEOTIFF, NPY (USING NPY)
   Years:       2024

Test: Error Handling - Invalid Input
-------------------------------------

Test that visualize fails gracefully with non-existent input:

  $ geotessera visualize \
  >   "$TESTDIR/nonexistent" \
  >   "$TESTDIR/output.tif" 2>&1 | grep -A 1 -E 'No tiles found|Error' | grep -v '^--$'
  No tiles found in
  * (glob)

Test that webmap fails gracefully with non-TIFF input:

  $ geotessera webmap \
  >   "$TESTDIR/cb_tiles_tiff" 2>&1 | grep -E 'Error.*must be.*tif'
  Error: Input must be a GeoTIFF file (.tif/.tiff)
