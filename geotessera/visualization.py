"""Visualization utilities for GeoTessera GeoTIFF files.

This module provides tools for visualizing and processing GeoTIFF files
created by GeoTessera, including bounding box calculations, mosaicking,
and web map generation for CLI display.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Callable
import tempfile
import shutil
import json

import numpy as np
import geopandas as gpd
import pandas as pd


def calculate_bbox_from_file(filepath: Union[str, Path]) -> Tuple[float, float, float, float]:
    """Calculate bounding box from a geometry file.
    
    Args:
        filepath: Path to GeoJSON, Shapefile, etc.
        
    Returns:
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    gdf = gpd.read_file(filepath)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    return tuple(bounds)


def calculate_bbox_from_points(
    points: Union[List[Dict], pd.DataFrame], 
    buffer_degrees: float = 0.1
) -> Tuple[float, float, float, float]:
    """Calculate bounding box from point data.
    
    Args:
        points: List of dicts with 'lat'/'lon' keys or DataFrame with lat/lon columns
        buffer_degrees: Buffer around points in degrees
        
    Returns:
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    if isinstance(points, list):
        df = pd.DataFrame(points)
    else:
        df = points
        
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("Points must have 'lat' and 'lon' columns")
        
    min_lon = df['lon'].min() - buffer_degrees
    max_lon = df['lon'].max() + buffer_degrees
    min_lat = df['lat'].min() - buffer_degrees
    max_lat = df['lat'].max() + buffer_degrees
    
    return (min_lon, min_lat, max_lon, max_lat)


def create_rgb_mosaic_from_geotiffs(
    geotiff_paths: List[str],
    output_path: str,
    bands: Tuple[int, int, int] = (0, 1, 2),
    normalize: bool = True,
    progress_callback: Optional[Callable] = None
) -> str:
    """Create an RGB visualization mosaic from multiple GeoTIFF files.
    
    Args:
        geotiff_paths: List of paths to GeoTIFF files
        output_path: Output path for RGB mosaic
        bands: Three band indices to map to RGB channels
        normalize: Whether to normalize each band to 0-255 range
        progress_callback: Optional callback function(current, total, status) for progress tracking
        
    Returns:
        Path to created RGB mosaic file
    """
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.enums import ColorInterp
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")
        
    if not geotiff_paths:
        raise ValueError("No GeoTIFF files provided")
    
    if progress_callback:
        progress_callback(10, 100, f"Opening {len(geotiff_paths)} GeoTIFF files...")
        
    # Open all files
    src_files = [rasterio.open(path) for path in geotiff_paths]
    
    try:
        if progress_callback:
            progress_callback(20, 100, "Merging GeoTIFF files...")
            
        # Merge the files
        merged_array, merged_transform = merge(src_files, method='first')
        
        if progress_callback:
            progress_callback(40, 100, f"Extracting RGB bands {bands}...")
        
        # Extract the three bands for RGB
        if merged_array.shape[0] < max(bands) + 1:
            raise ValueError(f"Not enough bands in source files. Requested bands {bands}, but only {merged_array.shape[0]} available")
            
        rgb_data = merged_array[list(bands)]  # Shape: (3, height, width)
        
        # Normalize if requested
        if normalize:
            if progress_callback:
                progress_callback(60, 100, "Normalizing RGB bands...")
                
            for i in range(3):
                band = rgb_data[i]
                band_min, band_max = np.nanmin(band), np.nanmax(band)
                if band_max > band_min:
                    rgb_data[i] = (band - band_min) / (band_max - band_min)
                else:
                    rgb_data[i] = 0
        else:
            if progress_callback:
                progress_callback(60, 100, "Processing RGB bands...")
                    
        if progress_callback:
            progress_callback(80, 100, "Converting to RGB format...")
            
        # Convert to uint8
        rgb_uint8 = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
        
        if progress_callback:
            progress_callback(90, 100, f"Writing RGB mosaic to {Path(output_path).name}...")
        
        # Write RGB GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rgb_uint8.shape[1],
            width=rgb_uint8.shape[2],
            count=3,
            dtype='uint8',
            crs=src_files[0].crs,
            transform=merged_transform,
            compress='lzw',
            photometric='RGB'
        ) as dst:
            dst.write(rgb_uint8)
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            dst.update_tags(
                TIFFTAG_ARTIST="GeoTessera",
                TIFFTAG_IMAGEDESCRIPTION=f"RGB visualization using bands {bands}"
            )
    
    finally:
        # Close all source files
        for src in src_files:
            src.close()
    
    if progress_callback:
        progress_callback(100, 100, f"Completed RGB mosaic: {Path(output_path).name}")
        
    return output_path


def geotiff_to_web_tiles(
    geotiff_path: str,
    output_dir: str,
    zoom_levels: Tuple[int, int] = (8, 15)
) -> str:
    """Convert GeoTIFF to web tiles for interactive display.
    
    Args:
        geotiff_path: Path to input GeoTIFF
        output_dir: Directory for web tiles output
        zoom_levels: Min and max zoom levels
        
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
    
    # Run gdal2tiles
    cmd = [
        'gdal2tiles.py',
        '-z', f'{min_zoom}-{max_zoom}',
        '-w', 'leaflet',
        geotiff_path,
        str(output_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_dir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdal2tiles failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("gdal2tiles.py not found. Install GDAL tools.")


def create_simple_web_viewer(
    tiles_dir: str,
    output_html: str,
    center_lat: float = 0,
    center_lon: float = 0,
    zoom: int = 10,
    title: str = "GeoTessera Visualization"
) -> str:
    """Create a simple HTML viewer for web tiles.
    
    Args:
        tiles_dir: Directory containing web tiles
        output_html: Output path for HTML file
        center_lat: Initial map center latitude
        center_lon: Initial map center longitude
        zoom: Initial zoom level
        title: Page title
        
    Returns:
        Path to created HTML file
    """
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
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
        
        // Add OpenStreetMap base layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add GeoTessera layer
        var tesseraLayer = L.tileLayer('./{{z}}/{{x}}/{{y}}.png', {{
            attribution: 'GeoTessera data',
            opacity: 0.8
        }}).addTo(map);
        
        // Layer control
        var baseMaps = {{
            "OpenStreetMap": L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png')
        }};
        
        var overlayMaps = {{
            "Tessera Data": tesseraLayer
        }};
        
        L.control.layers(baseMaps, overlayMaps).addTo(map);
    </script>
</body>
</html>"""
    
    with open(output_html, 'w') as f:
        f.write(html_content)
        
    return output_html


def analyze_geotiff_coverage(geotiff_paths: List[str]) -> Dict:
    """Analyze coverage and metadata of GeoTIFF files.
    
    Args:
        geotiff_paths: List of GeoTIFF file paths
        
    Returns:
        Dictionary with coverage statistics and metadata
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")
        
    if not geotiff_paths:
        return {"error": "No files provided"}
        
    coverage_info = {
        "total_files": len(geotiff_paths),
        "tiles": [],
        "bounds": {"min_lon": float('inf'), "min_lat": float('inf'),
                  "max_lon": float('-inf'), "max_lat": float('-inf')},
        "band_counts": {},
        "years": set(),
        "crs": set()
    }
    
    for path in geotiff_paths:
        try:
            with rasterio.open(path) as src:
                bounds = src.bounds
                
                # Update overall bounds
                coverage_info["bounds"]["min_lon"] = min(coverage_info["bounds"]["min_lon"], bounds.left)
                coverage_info["bounds"]["min_lat"] = min(coverage_info["bounds"]["min_lat"], bounds.bottom)
                coverage_info["bounds"]["max_lon"] = max(coverage_info["bounds"]["max_lon"], bounds.right)
                coverage_info["bounds"]["max_lat"] = max(coverage_info["bounds"]["max_lat"], bounds.top)
                
                # Track band counts
                band_count = src.count
                coverage_info["band_counts"][band_count] = coverage_info["band_counts"].get(band_count, 0) + 1
                
                # Extract metadata
                tags = src.tags()
                if "TESSERA_YEAR" in tags:
                    coverage_info["years"].add(tags["TESSERA_YEAR"])
                    
                coverage_info["crs"].add(str(src.crs))
                
                # Tile info
                coverage_info["tiles"].append({
                    "path": path,
                    "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    "bands": band_count,
                    "year": tags.get("TESSERA_YEAR", "unknown"),
                    "tile_lat": tags.get("TESSERA_TILE_LAT", "unknown"),
                    "tile_lon": tags.get("TESSERA_TILE_LON", "unknown")
                })
                
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
            continue
    
    # Convert sets to lists for JSON serialization
    coverage_info["years"] = sorted(list(coverage_info["years"]))
    coverage_info["crs"] = list(coverage_info["crs"])
    
    return coverage_info


def create_coverage_summary_map(
    geotiff_paths: List[str],
    output_html: str,
    title: str = "GeoTessera Coverage Map"
) -> str:
    """Create an HTML map showing tile coverage.
    
    Args:
        geotiff_paths: List of GeoTIFF file paths
        output_html: Output HTML file path
        title: Map title
        
    Returns:
        Path to created HTML file
    """
    # Analyze coverage
    coverage = analyze_geotiff_coverage(geotiff_paths)
    
    if not coverage["tiles"]:
        raise ValueError("No valid GeoTIFF files found")
        
    # Calculate center
    bounds = coverage["bounds"]
    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
    
    # Generate tile rectangles for map
    tile_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for tile in coverage["tiles"]:
        min_lon, min_lat, max_lon, max_lat = tile["bounds"]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat], 
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            },
            "properties": {
                "year": tile["year"],
                "bands": tile["bands"],
                "lat": tile["tile_lat"],
                "lon": tile["tile_lon"],
                "path": Path(tile["path"]).name
            }
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
                "Position: (" + props.lat + ", " + props.lon + ")<br>" +
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
    
    with open(output_html, 'w') as f:
        f.write(html_content)
        
    return output_html