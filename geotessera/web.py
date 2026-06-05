"""Web visualization utilities for GeoTessera.

This module provides functions for generating web tiles and interactive
visualizations using Leaflet and other web technologies.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Callable
import json
import logging

# Module-level logger
logger = logging.getLogger(__name__)


def prepare_mosaic_for_web(
    input_mosaic: str,
    output_path: str,
    target_crs: str = "EPSG:3857",
    progress_callback: Optional[Callable] = None,
) -> str:
    """Prepare an RGB mosaic for web visualization by reprojecting if needed.

    Args:
        input_mosaic: Path to input RGB mosaic (3-band GeoTIFF)
        output_path: Output path for web-ready mosaic
        target_crs: Target CRS for web visualization (default: Web Mercator)
        progress_callback: Optional progress callback

    Returns:
        Path to web-ready mosaic (may be same as input if no reprojection needed)
    """
    try:
        import rasterio
        from rasterio.warp import reproject, calculate_default_transform, Resampling
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")

    if progress_callback:
        progress_callback(10, 100, "Checking mosaic CRS...")

    # Check if reprojection is needed
    with rasterio.open(input_mosaic) as src:
        if str(src.crs) == target_crs:
            if progress_callback:
                progress_callback(100, 100, f"Mosaic already in {target_crs}")
            return input_mosaic

        if progress_callback:
            progress_callback(
                20, 100, f"Reprojecting from {src.crs} to {target_crs}..."
            )

        # Calculate transform and dimensions for target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Create output profile
        profile = src.profile.copy()
        profile.update(
            {
                "crs": target_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "compress": "lzw",
            }
        )

        if progress_callback:
            progress_callback(50, 100, "Writing reprojected mosaic...")

        # Write reprojected mosaic
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

            # Copy tags and color interpretation
            dst.update_tags(**src.tags())
            dst.colorinterp = src.colorinterp

    if progress_callback:
        progress_callback(100, 100, "Mosaic prepared for web visualization")

    return output_path


def geotiff_to_web_tiles(
    geotiff_path: str,
    output_dir: str,
    zoom_levels: Tuple[int, int] = (8, 15),
    use_gdal_raster: bool = False,
) -> str:
    """Convert GeoTIFF to web tiles for interactive display.

    By default uses gdal2tiles.py for stability. Optionally can use the newer
    'gdal raster tile' command which may be faster but less stable.

    Args:
        geotiff_path: Path to input GeoTIFF
        output_dir: Directory for web tiles output
        zoom_levels: Min and max zoom levels
        use_gdal_raster: If True, use 'gdal raster tile' instead of gdal2tiles

    Returns:
        Path to tiles directory
    """
    try:
        import subprocess
    except ImportError:
        raise ImportError("gdal2tiles required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_zoom, max_zoom = zoom_levels

    # Use gdal raster tile if explicitly requested and available
    if use_gdal_raster:

        def _has_gdal_raster_tile() -> bool:
            """Check if 'gdal raster tile' command is available."""
            try:
                result = subprocess.run(
                    ["gdal", "raster", "tile", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                return False

        if _has_gdal_raster_tile():
            cmd = [
                "gdal",
                "raster",
                "tile",
                "--min-zoom",
                str(min_zoom),
                "--max-zoom",
                str(max_zoom),
                "--tiling-scheme",
                "WebMercatorQuad",
                "--resampling",
                "bilinear",
                "--webviewer",
                "leaflet",
                "--num-threads",
                "1",
                geotiff_path,
                str(output_dir),
            ]

            try:
                logger.info(f"Running gdal raster tile: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    logger.debug("GDAL stdout: %s", result.stdout)
                if result.stderr:
                    logger.debug("GDAL stderr: %s", result.stderr)
                return str(output_dir)
            except subprocess.CalledProcessError as e:
                logger.error(f"gdal raster tile failed (return code {e.returncode}):")
                logger.error(f"Command: {' '.join(cmd)}")
                if e.stdout:
                    logger.error(f"Stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"Stderr: {e.stderr}")
                raise RuntimeError(f"gdal raster tile failed: {e}")
        else:
            raise RuntimeError(
                "gdal raster tile not available. Use default gdal2tiles or install gdal with raster tile support."
            )

    # Use traditional gdal2tiles.py (default)
    cmd = [
        "gdal2tiles.py",
        "-z",
        f"{min_zoom}-{max_zoom}",
        "-w",
        "leaflet",
        "-p",
        "mercator",  # Explicitly use mercator projection
        "--resampling",
        "bilinear",
        geotiff_path,
        str(output_dir),
    ]

    try:
        logger.info(f"Running gdal2tiles fallback: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.debug("gdal2tiles stdout: %s", result.stdout)
        if result.stderr:
            logger.debug("gdal2tiles stderr: %s", result.stderr)
        return str(output_dir)
    except subprocess.CalledProcessError as e:
        logger.error(f"gdal2tiles failed (return code {e.returncode}):")
        logger.error(f"Command: {' '.join(cmd)}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(
            f"Tile generation failed with both gdal raster tile and gdal2tiles: {e}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Neither 'gdal raster tile' nor 'gdal2tiles.py' found. Install GDAL tools."
        )


def _generate_boundary_js(boundary_geojson: str) -> str:
    """Generate JavaScript code to add boundary overlay to Leaflet map."""
    if not boundary_geojson:
        return "// No boundary to overlay"

    return f"""
        // Add boundary overlay
        var boundaryData = {boundary_geojson};
        var boundaryLayer = L.geoJSON(boundaryData, {{
            style: {{
                color: '#ff0000',
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.1,
                fillColor: '#ff0000'
            }}
        }});
        boundaryLayer.addTo(map);
        
        // Add boundary to layer control
        overlayMaps["Region Boundary"] = boundaryLayer;
        map.removeControl(map._controlContainer.querySelector('.leaflet-control-layers'));
        L.control.layers(baseMaps, overlayMaps).addTo(map);
    """


def create_simple_web_viewer(
    tiles_dir: str,
    output_html: str,
    center_lon: float = 0,
    center_lat: float = 0,
    zoom: int = 10,
    title: str = "GeoTessera Visualization",
    region_file: str = None,
) -> str:
    """Create a simple HTML viewer for web tiles.

    Args:
        tiles_dir: Directory containing web tiles
        output_html: Output path for HTML file
        center_lon: Initial map center longitude
        center_lat: Initial map center latitude
        zoom: Initial zoom level
        title: Page title
        region_file: Optional GeoJSON/Shapefile boundary to overlay

    Returns:
        Path to created HTML file
    """
    # Process region file if provided
    boundary_geojson = None
    if region_file:
        try:
            import geopandas as gpd

            # Read the region file and convert to GeoJSON
            gdf = gpd.read_file(region_file)
            # Convert to WGS84 if not already
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Convert to GeoJSON string
            boundary_geojson = gdf.__geo_interface__
            boundary_geojson = json.dumps(boundary_geojson)

        except Exception as e:
            logger.warning(f"Could not process region file {region_file}: {e}")
            boundary_geojson = None

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body {{ height: 100%; margin: 0; padding: 0; }}
        #map {{ height: 100%; }}
        .opacity-control {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }}
        .opacity-control label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .opacity-control input[type="range"] {{
            width: 150px;
        }}
        .opacity-value {{
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <!-- Opacity Control -->
    <div class="opacity-control">
        <label for="opacity-slider">GeoTessera Opacity</label>
        <input type="range" id="opacity-slider" min="0" max="100" value="80" step="5">
        <div class="opacity-value" id="opacity-value">80%</div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
        
        // Add OpenStreetMap base layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add GeoTessera layer
        var tesseraLayer = L.tileLayer('./tiles/{{z}}/{{x}}/{{y}}.png', {{
            attribution: 'GeoTessera data',
            opacity: 0.8,
            tms: true
        }}).addTo(map);
        
        // Layer control
        var baseMaps = {{
            "OpenStreetMap": L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png')
        }};
        
        var overlayMaps = {{
            "Tessera Data": tesseraLayer
        }};
        
        L.control.layers(baseMaps, overlayMaps).addTo(map);
        
        // Add boundary layer if provided
        {_generate_boundary_js(boundary_geojson)}
        
        // Opacity slider functionality
        var opacitySlider = document.getElementById('opacity-slider');
        var opacityValue = document.getElementById('opacity-value');
        
        opacitySlider.addEventListener('input', function() {{
            var opacity = this.value / 100;
            tesseraLayer.setOpacity(opacity);
            opacityValue.textContent = this.value + '%';
        }});
    </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html


def create_coverage_summary_map(
    geotiff_paths: List[str], output_html: str, title: str = "GeoTessera Coverage Map"
) -> str:
    """Create an HTML map showing tile coverage.

    Args:
        geotiff_paths: List of GeoTIFF file paths
        output_html: Output HTML file path
        title: Map title

    Returns:
        Path to created HTML file
    """
    from .visualization import analyze_geotiff_coverage

    # Analyze coverage
    coverage = analyze_geotiff_coverage(geotiff_paths)

    if not coverage["tiles"]:
        raise ValueError("No valid GeoTIFF files found")

    # Calculate center
    bounds = coverage["bounds"]
    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

    # Generate tile rectangles for map
    tile_geojson = {"type": "FeatureCollection", "features": []}

    for tile in coverage["tiles"]:
        min_lon, min_lat, max_lon, max_lat = tile["bounds"]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]
                ],
            },
            "properties": {
                "year": tile["year"],
                "bands": tile["bands"],
                "lon": tile["tile_lon"],
                "lat": tile["tile_lat"],
                "path": Path(tile["path"]).name,
            },
        }
        tile_geojson["features"].append(feature)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body {{ height: 100%; margin: 0; padding: 0; }}
        #map {{ height: 100%; }}
        .info {{ 
            padding: 10px; background: white; background: rgba(255,255,255,0.9);
            box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; 
            font-family: Arial, sans-serif; font-size: 12px;
        }}
        .info h4 {{ margin: 0 0 5px; color: #777; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 8);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        var geojsonData = {json.dumps(tile_geojson)};
        
        function style(feature) {{
            return {{
                fillColor: '#3388ff',
                weight: 1,
                opacity: 1,
                color: 'white',
                fillOpacity: 0.3
            }};
        }}
        
        function onEachFeature(feature, layer) {{
            var props = feature.properties;
            var popupContent = 
                "<b>Tessera Tile</b><br>" +
                "Year: " + props.year + "<br>" +
                "Bands: " + props.bands + "<br>" +
                "Position: (" + props.lon + ", " + props.lat + ")<br>" +
                "File: " + props.path;
            layer.bindPopup(popupContent);
        }}
        
        L.geoJSON(geojsonData, {{
            style: style,
            onEachFeature: onEachFeature
        }}).addTo(map);
        
        // Add info control
        var info = L.control();
        info.onAdd = function (map) {{
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        }};
        info.update = function (props) {{
            this._div.innerHTML = '<h4>GeoTessera Coverage</h4>' +
                'Total tiles: {coverage["total_files"]}<br>' +
                'Years: {", ".join(coverage["years"])}<br>' +
                'Click on tiles for details';
        }};
        info.addTo(map);
    </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html
