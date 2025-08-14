"""Command-line interface for accessing and visualizing Tessera embeddings.

This module provides CLI commands for:
- Listing and exploring available embedding tiles
- Creating visualizations from embeddings as GeoTIFF files
- Generating interactive web maps with Leaflet.js
- Serving embedding tiles as web map tiles for visualization

The CLI supports multiple visualization modes including false-color composites
from different embedding channels and interactive web-based exploration.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import datetime
import json
import http.server
import socketserver
import webbrowser
import tempfile
import shutil
import subprocess
import os
from urllib.parse import urlparse
from .core import GeoTessera
from .io import load_roi
from . import __version__


def list_embeddings_command(args):
    """List available Tessera embedding tiles with optional limit.

    Displays all available embedding tiles in the dataset, showing year,
    latitude, and longitude for each tile. Useful for exploring coverage
    and finding specific regions.

    Args:
        args: Parsed command line arguments containing:
            - version: Dataset version to use
            - registry_dir: Optional local registry directory
            - limit: Optional maximum number of tiles to display

    Example output::

        Available embeddings (50000 total):
          - Year 2024: (51.50, -0.10)
          - Year 2024: (51.50, -0.00)
          ... and 49998 more
    """
    tessera = GeoTessera(
        version=args.dataset_version,
        registry_dir=args.registry_dir,
        auto_update=args.auto_update,
        manifests_repo_url=args.manifests_repo_url,
    )

    embeddings = list(tessera.list_available_embeddings())
    print(f"Available embeddings ({len(embeddings)} total):")

    if args.limit:
        embeddings = embeddings[: args.limit]

    for year, lat, lon in embeddings:
        print(f"  - Year {year}: ({lat:.2f}, {lon:.2f})")

    total_count = tessera.count_available_embeddings()
    if args.limit and args.limit < total_count:
        print(f"  ... and {total_count - args.limit} more")


def info_command(args):
    """Display comprehensive information about the GeoTessera dataset.

    Shows dataset metadata including version, data URLs, cache locations,
    and counts of available embeddings and land masks. Useful for debugging
    and understanding the current configuration.

    Args:
        args: Parsed command line arguments containing:
            - version: Dataset version to query
            - registry_dir: Optional local registry directory

    Information displayed:
        - Dataset version identifier
        - Remote data URLs for embeddings and land masks
        - Local cache directory path
        - Total count of available embeddings
        - Count of auxiliary land mask files
    """
    tessera = GeoTessera(
        version=args.dataset_version,
        registry_dir=args.registry_dir,
        auto_update=args.auto_update,
        manifests_repo_url=args.manifests_repo_url,
    )

    print("=== GeoTessera Dataset Information ===")
    print(f"Version: {tessera.version}")
    print(f"Base URL: {tessera._pooch.base_url}")
    print(f"Cache directory: {tessera._pooch.path}")

    # Enhanced embedding information
    total_embeddings = tessera.count_available_embeddings()
    print(f"Total embeddings: {total_embeddings:,}")

    # Show available years
    try:
        available_years = tessera.get_available_years()
        print(f"Available years: {', '.join(map(str, available_years))}")
    except Exception:
        print("Available years: Loading...")

    # Enhanced land mask information
    landmask_count = tessera._count_available_landmasks()
    print(f"Land masks available: {landmask_count:,}")
    print(
        f"Land mask Base URL: {tessera._landmask_pooch.base_url if tessera._landmask_pooch else 'Not loaded'}"
    )

    # Show registry information
    if tessera._registry_dir:
        print(f"Using local registry: {tessera._registry_dir}")
    else:
        print("Using remote registry (auto-downloaded)")

    # Show loaded blocks information
    if tessera._loaded_blocks:
        print(f"Registry blocks loaded: {len(tessera._loaded_blocks)}")
    else:
        print("Registry blocks: None loaded (will load on-demand)")


def map_command(args):
    """Generate a world map visualization showing Tessera embedding coverage.

    Creates a high-resolution map image displaying all available embedding
    tile locations as red dots on a world map background. Useful for
    understanding global coverage and identifying data-rich regions.

    Args:
        args: Parsed command line arguments containing:
            - version: Dataset version to map
            - output: Output filename for the map image (PNG format)

    Output:
        Saves a PNG image with:
        - World map base layer with country boundaries
        - Red dots marking each available 0.1° embedding tile
        - Legend and grid lines for reference
        - Timestamp and total tile count

    Note:
        The map generation may take several seconds for large datasets
        as it processes all available tile locations.
    """
    tessera = GeoTessera(
        version=args.dataset_version,
        registry_dir=args.registry_dir,
        auto_update=args.auto_update,
        manifests_repo_url=args.manifests_repo_url,
    )

    print("Generating coverage map from embedding registry data...")

    # Get all available embeddings from the library (enhanced with progress reporting)
    print("Loading embedding registry data...")
    embeddings = list(tessera.list_available_embeddings())

    if not embeddings:
        print("No embeddings available. Check registry loading.")
        return

    print(f"Processing {len(embeddings)} embedding tiles...")

    # Extract unique coordinates (lat, lon) from embeddings with better performance
    coordinates = set()
    years_found = set()
    for year, lat, lon in embeddings:
        coordinates.add((lat, lon))
        years_found.add(year)

    print(
        f"Found {len(coordinates)} unique grid points across {len(years_found)} years: {sorted(years_found)}"
    )

    # Convert to DataFrame with better column naming
    coords_list = list(coordinates)
    df = pd.DataFrame(coords_list, columns=["Latitude", "Longitude"])

    # Convert pandas DataFrame to GeoDataFrame
    print("Creating GeoDataFrame...")
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    gdf.set_crs("EPSG:4326", inplace=True)  # Set the coordinate system to WGS84

    # Plot the map
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = geopandas.read_file(world_url)

    # Create the plotting area
    print("Creating map...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Plot the world map base layer
    world.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)

    # Plot grid points on the map with smaller markers
    gdf.plot(
        ax=ax,
        marker="o",
        color="red",
        markersize=2,
        alpha=0.8,
        label="Available Embeddings",
    )

    # Add a title with the current time and grid count
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    # Enhanced title with more information
    years_str = ", ".join(map(str, sorted(years_found)))
    ax.set_title(
        f"GeoTessera Embedding Coverage Map\nGrid Points: {len(df):,} | Years: {years_str}\nGenerated: {current_time_utc}",
        fontdict={"fontsize": "16", "fontweight": "bold"},
    )

    # Add legend
    ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True, fontsize=12)

    # Set axis labels
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Save the map as an image file
    plt.tight_layout()
    output_path = args.output
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Coverage map saved to {output_path}")
    print(f"Map shows {len(df):,} unique grid points across {len(years_found)} years")

    # Close the plot to free memory
    plt.close()


def visualize_command(args):
    """Create GeoTIFF from Tessera embeddings.

    Generates a georeferenced image by mosaicking embedding tiles that
    overlap with a region file. Outputs float32 data preserving the
    dequantized embedding values.

    Args:
        args: Parsed command line arguments containing:
            - region: Path to region file defining the area
            - output: Output GeoTIFF filename
            - target_crs: Coordinate reference system for output
            - bands: Channel indices to include (0-127)
            - year: Year of embeddings to use
            - dataset_version: Dataset version

    Process:
        1. Reads geographic bounds from region file
        2. Identifies all embedding tiles overlapping the region
        3. Downloads and merges tiles with proper alignment
        4. Exports georeferenced GeoTIFF with metadata

    Example band selections:
        - None: All 128 channels (default)
        - [0, 1, 2]: First three channels
        - [10, 20, 30]: Mid-range channels
        - [25, 50, 75]: Distributed sampling

    Raises:
        SystemExit: If region file is missing or invalid
    """
    tessera = GeoTessera(
        version=args.dataset_version,
        registry_dir=args.registry_dir,
        auto_update=args.auto_update,
        manifests_repo_url=args.manifests_repo_url,
    )

    if not args.region:
        print("Error: --region is required for visualization")
        sys.exit(1)

    print(f"Analyzing region file: {args.region}")

    # Use core library method to create merged TIFF
    output_path = tessera.merge_embeddings_for_region_file(
        region_path=args.region,
        output_path=args.output,
        bands=args.bands,
        year=args.year,
        target_crs=args.target_crs,
    )

    if output_path:
        print(f"Created merged embedding visualization for region: {output_path}")
    else:
        print("Failed to create visualization")
        print(
            "Supported formats: GeoJSON, TopoJSON, Shapefile (.shp), GeoPackage (.gpkg)"
        )
        sys.exit(1)


class GeoJSONRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for interactive Tessera embedding visualization.

    Serves a Leaflet.js-based web interface that displays:
    - GeoJSON/TopoJSON features as vector overlays
    - Tessera embedding tiles as false-color base layers
    - Interactive controls for opacity and layer switching

    The handler manages three main endpoints:
    - /: Main HTML page with Leaflet map
    - /geojson: GeoJSON data for vector overlay
    - /tiles/{z}/{x}/{y}.png: Tessera visualization tiles

    Attributes:
        geojson_data: Parsed GeoJSON data to serve
        tiles_dir: Directory containing pre-generated tiles
        tile_bounds: Geographic bounds extracted from tile metadata
    """

    def __init__(self, *args, geojson_data=None, tiles_dir=None, **kwargs):
        self.geojson_data = geojson_data
        self.tiles_dir = tiles_dir
        self.tile_bounds = self._get_tile_bounds() if tiles_dir else None
        super().__init__(*args, **kwargs)

    def _get_tile_bounds(self):
        """Extract geographic bounds from tile metadata.

        Parses the tilemapresource.xml file generated by gdal2tiles
        to determine the geographic extent of available tiles. This
        helps Leaflet optimize tile loading and set appropriate bounds.

        Returns:
            list or None: Bounds as [[south, west], [north, east]] for
                         Leaflet, or None if metadata is unavailable.

        Note:
            Gracefully handles missing or malformed metadata files.
        """
        if not self.tiles_dir:
            return None

        tilemapresource_path = os.path.join(self.tiles_dir, "tilemapresource.xml")
        if not os.path.exists(tilemapresource_path):
            return None

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(tilemapresource_path)
            root = tree.getroot()

            # Find BoundingBox element
            bbox_elem = root.find("BoundingBox")
            if bbox_elem is not None:
                minx = float(bbox_elem.get("minx"))
                miny = float(bbox_elem.get("miny"))
                maxx = float(bbox_elem.get("maxx"))
                maxy = float(bbox_elem.get("maxy"))
                return [
                    [miny, minx],
                    [maxy, maxx],
                ]  # Format for Leaflet: [[south, west], [north, east]]
        except Exception as e:
            print(f"Warning: Could not parse tile bounds: {e}")

        return None

    def do_GET(self):
        """Handle HTTP GET requests for map interface and data.

        Routes requests to appropriate handlers:
        - Root path serves the interactive map HTML
        - /geojson endpoint serves vector data as JSON
        - /tiles/* paths serve pre-generated PNG tiles
        - Other paths use default file serving

        Tile requests that 404 are handled gracefully as this is
        expected behavior for areas without data coverage.
        """
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/":
            # Serve the main HTML page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html_content = self.generate_html()
            self.wfile.write(html_content.encode("utf-8"))
        elif parsed_path.path == "/geojson":
            # Serve the GeoJSON data
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(self.geojson_data).encode("utf-8"))
        elif parsed_path.path.startswith("/tiles/") and self.tiles_dir:
            # Serve custom tiles
            tile_path = parsed_path.path[7:]  # Remove '/tiles/' prefix
            full_tile_path = os.path.join(self.tiles_dir, tile_path)
            if os.path.exists(full_tile_path) and os.path.isfile(full_tile_path):
                try:
                    with open(full_tile_path, "rb") as tile_file:
                        self.send_response(200)
                        if full_tile_path.endswith(".png"):
                            self.send_header("Content-type", "image/png")
                        elif full_tile_path.endswith(".jpg") or full_tile_path.endswith(
                            ".jpeg"
                        ):
                            self.send_header("Content-type", "image/jpeg")
                        else:
                            self.send_header("Content-type", "application/octet-stream")
                        self.send_header("Cache-Control", "public, max-age=86400")
                        self.end_headers()
                        self.wfile.write(tile_file.read())
                except Exception as e:
                    print(f"Error serving tile {tile_path}: {e}")
                    self.send_error(500, f"Error serving tile: {e}")
            else:
                # Return 404 without verbose logging for missing tiles (expected behavior)
                self.send_response(404)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Tile not found")
        else:
            # Default handler for other requests
            super().do_GET()

    def generate_html(self):
        """Generate interactive map HTML with Leaflet.js integration.

        Creates a complete HTML page featuring:
        - Leaflet.js map with OpenStreetMap base layer
        - Optional Tessera false-color tile overlay
        - GeoJSON feature rendering with popups
        - Opacity slider for tile layer control
        - Layer switcher for base/overlay selection
        - Responsive design for various screen sizes

        Returns:
            str: Complete HTML document as a string

        Note:
            The generated HTML uses CDN-hosted Leaflet.js libraries
            and includes all necessary CSS and JavaScript inline.
        """
        tessera_meta = (
            '<meta name="tessera-tiles" content="true">' if self.tiles_dir else ""
        )
        tile_bounds_js = (
            f"var tileBounds = {json.dumps(self.tile_bounds)};"
            if self.tile_bounds
            else "var tileBounds = null;"
        )

        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>GeoTessera Map Viewer</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {tessera_meta}
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; width: 100vw; }}
        .info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 200px;
        }}
        .legend h4 {{ margin: 0 0 10px 0; }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #ccc;
        }}
        .opacity-control {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            width: 200px;
        }}
        .opacity-control h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
        }}
        .opacity-slider {{
            width: 100%;
            margin: 10px 0;
        }}
        .opacity-value {{
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h3>GeoTessera Map Viewer</h3>
        <p>Interactive map with tessera false color visualization</p>
    </div>
    
    <div class="legend">
        <h4>Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background-color: transparent; border: 2px solid #ff0000;"></div>
            <span>GeoJSON Outline</span>
        </div>
    </div>
    
    <div class="opacity-control" id="opacityControl" style="display: none;">
        <h4>Tessera Layer Opacity</h4>
        <input type="range" min="0" max="100" value="70" class="opacity-slider" id="opacitySlider">
        <div class="opacity-value" id="opacityValue">70%</div>
    </div>
    
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Initialize the map
        var map = L.map('map').setView([51.505, -0.09], 2);

        // Add OpenStreetMap tile layer
        var osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }});
        
        // Add custom tessera tiles layer (if available)
        var tesseraLayer = null;
        if (window.location.pathname === '/' && document.querySelector('meta[name="tessera-tiles"]')) {{
            var tileOptions = {{
                attribution: 'Tessera False Color Visualization',
                opacity: 0.7,
                maxNativeZoom: 15,  // Maximum zoom level of actual tiles
                maxZoom: 20         // Allow zooming beyond native zoom (will scale tiles)
            }};
            
            // Add bounds if available
            {tile_bounds_js}
            if (tileBounds) {{
                tileOptions.bounds = tileBounds;
            }}
            
            tesseraLayer = L.tileLayer('/tiles/{{z}}/{{x}}/{{y}}.png', tileOptions);
        }}
        
        // Create base layers object
        var baseLayers = {{
            "OpenStreetMap": osmLayer
        }};
        
        // Create overlay layers object
        var overlayLayers = {{}};
        if (tesseraLayer) {{
            overlayLayers["Tessera Visualization"] = tesseraLayer;
        }}
        
        // Add default layer
        osmLayer.addTo(map);
        
        // Add tessera layer by default if available
        if (tesseraLayer) {{
            tesseraLayer.addTo(map);
            
            // Show opacity control
            document.getElementById('opacityControl').style.display = 'block';
            
            // Set up opacity slider
            var opacitySlider = document.getElementById('opacitySlider');
            var opacityValue = document.getElementById('opacityValue');
            
            opacitySlider.addEventListener('input', function(e) {{
                var opacity = e.target.value / 100;
                tesseraLayer.setOpacity(opacity);
                opacityValue.textContent = e.target.value + '%';
            }});
        }}
        
        // Add layer control if we have tessera tiles
        if (tesseraLayer) {{
            L.control.layers(baseLayers, overlayLayers, {{
                position: 'topright'
            }}).addTo(map);
        }}

        // Function to get random color
        function getRandomColor() {{
            const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe', '#fd79a8', '#00b894'];
            return colors[Math.floor(Math.random() * colors.length)];
        }}

        // Function to style GeoJSON features
        function style(feature) {{
            return {{
                fillColor: 'transparent',
                weight: 2,
                opacity: 1,
                color: '#ff0000',
                dashArray: '',
                fillOpacity: 0
            }};
        }}

        // Function to handle feature clicks
        function onEachFeature(feature, layer) {{
            if (feature.properties) {{
                let popupContent = '<h4>Feature Properties</h4>';
                for (let key in feature.properties) {{
                    popupContent += '<b>' + key + ':</b> ' + feature.properties[key] + '<br>';
                }}
                layer.bindPopup(popupContent);
            }}
        }}

        // Fetch and display GeoJSON data
        fetch('/geojson')
            .then(response => response.json())
            .then(data => {{
                const geojsonLayer = L.geoJSON(data, {{
                    style: style,
                    onEachFeature: onEachFeature
                }}).addTo(map);
                
                // Fit map to GeoJSON bounds
                if (geojsonLayer.getBounds().isValid()) {{
                    map.fitBounds(geojsonLayer.getBounds());
                }}
            }})
            .catch(error => {{
                console.error('Error loading GeoJSON data:', error);
                alert('Error loading GeoJSON data. Please check the console for details.');
            }});

        // Add scale control
        L.control.scale().addTo(map);
    </script>
</body>
</html>"""

        return html_template.format(
            tessera_meta=tessera_meta, tile_bounds_js=tile_bounds_js
        )


def generate_static_tessera_tiles(
    region_path,
    output_dir,
    tessera_version="v1",
    year=2024,
    bands=[0, 1, 2],
    registry_dir=None,
    auto_update=False,
    manifests_repo_url="https://github.com/ucam-eo/tessera-manifests.git",
):
    """Generate web map tiles from Tessera embeddings for a region.

    Creates a pyramid of PNG tiles suitable for web mapping applications
    by processing Tessera embeddings into false-color visualizations and
    tiling them using gdal2tiles. Tiles follow the XYZ/Slippy map standard.

    Args:
        region_path: Path to region file defining the area of interest (supports multiple formats)
        output_dir: Directory to store generated tiles and metadata
        tessera_version: Dataset version to use (default: "v1")
        year: Year of embeddings to process (default: 2024)
        bands: Three channel indices for RGB visualization (default: [0,1,2])
        registry_dir: Local directory containing registry files (optional)
        auto_update: Update tessera-manifests repository to latest version

    Returns:
        str or None: Path to tiles directory if successful, None on error

    Process:
        1. Checks if tiles already exist (skips regeneration)
        2. Reads region bounds from region file
        3. Merges Tessera embeddings into single GeoTIFF
        4. Runs gdal2tiles to create tile pyramid (zoom 1-15)
        5. Cleans up intermediate files

    Note:
        Requires GDAL tools to be installed (specifically gdal2tiles.py).
        Tile generation can be time-consuming for large regions.
    """
    # Check if tiles already exist FIRST
    tiles_output_dir = os.path.join(output_dir, "tiles")
    if os.path.exists(tiles_output_dir) and os.path.isdir(tiles_output_dir):
        # Check if tiles directory has content (at least one .png file)
        has_tiles = False
        for root, dirs, files in os.walk(tiles_output_dir):
            if any(f.endswith(".png") for f in files):
                has_tiles = True
                break

        if has_tiles:
            print(f"Tiles already exist in: {tiles_output_dir}")
            print("Skipping tile generation. Delete the tiles directory to regenerate.")
            return tiles_output_dir

    print("Generating static tessera false color visualization tiles...")

    # Initialize tessera
    tessera = GeoTessera(
        version=tessera_version,
        registry_dir=registry_dir,
        auto_update=auto_update,
        manifests_repo_url=manifests_repo_url,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate tessera visualization TIFF using core library method
    print("Merging tessera embeddings for region...")
    temp_tiff = os.path.join(output_dir, "tessera_viz.tiff")

    output_path = tessera.merge_embeddings_for_region_file(
        region_path=region_path,
        output_path=temp_tiff,
        bands=bands,
        year=year,
        target_crs="EPSG:4326",
    )

    if not output_path:
        print("Failed to generate tessera visualization")
        return None

    print(f"Generated tessera visualization: {output_path}")

    try:
        # Convert float32 TIFF to 8-bit for gdal2tiles compatibility
        print("Converting to 8-bit for tile generation...")
        temp_vrt = os.path.join(output_dir, "tessera_viz_8bit.vrt")

        # Use gdal_translate to convert to 8-bit with automatic scaling
        translate_cmd = [
            "gdal_translate",
            "-of",
            "VRT",
            "-ot",
            "Byte",
            "-scale",
            output_path,
            temp_vrt,
        ]

        result = subprocess.run(translate_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting to 8-bit: {result.stderr}")
            return None

        print(f"Created 8-bit VRT: {temp_vrt}")

        # Generate tiles using gdal2tiles with the 8-bit VRT
        print("Generating tiles with gdal2tiles...")

        cmd = [
            "gdal2tiles.py",
            "--zoom=1-15",
            "--processes=4",
            "--webviewer=none",
            "--tiledriver=PNG",
            "--xyz",  # Generate tiles in XYZ format for Leaflet compatibility
            temp_vrt,  # Use the 8-bit VRT instead of the original TIFF
            tiles_output_dir,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running gdal2tiles: {result.stderr}")
            return None

        print(f"Static tiles generated successfully in: {tiles_output_dir}")

        # Clean up the intermediate files
        if os.path.exists(temp_tiff):
            os.remove(temp_tiff)
            print(f"Cleaned up intermediate TIFF: {temp_tiff}")
        if os.path.exists(temp_vrt):
            os.remove(temp_vrt)
            print(f"Cleaned up intermediate VRT: {temp_vrt}")

        return tiles_output_dir

    except Exception as e:
        print(f"Error generating static tessera tiles: {e}")
        return None


def serve_command(args):
    """Launch interactive web server for Tessera embedding visualization.

    Starts a local HTTP server that serves an interactive map interface
    combining GeoJSON vector features with Tessera embedding tiles as
    false-color base layers. Supports real-time opacity adjustment and
    layer switching.

    Args:
        args: Parsed command line arguments containing:
            - geojson: Path to GeoJSON file (required)
            - port: HTTP server port (default: 8000)
            - open: Whether to auto-open browser
            - tiles_output: Directory for tile storage (optional)
            - bands: Channel indices for visualization
            - year: Year of embeddings to use
            - version: Dataset version

    Features:
        - Automatic tile generation from embeddings
        - Interactive web map with multiple layers
        - GeoJSON feature popups with properties
        - Opacity control for embedding overlay
        - Persistent or temporary tile storage

    Example:
        geotessera serve --region city.json --open --bands 30 60 90

    Note:
        Press Ctrl+C to stop the server. Temporary tiles are
        automatically cleaned up on exit.
    """
    if not args.region:
        print("Error: --region is required for serve command")
        sys.exit(1)

    # Load region data using enhanced I/O utility (supports multiple formats)
    region_path = Path(args.region)
    if not region_path.exists():
        print(f"Error: Region file not found: {region_path}")
        sys.exit(1)

    try:
        # Load as GeoDataFrame first to validate
        gdf = load_roi(str(region_path))
        print(f"Loaded region data from: {region_path} ({len(gdf)} features)")

        # Convert to GeoJSON format for web serving
        geojson_data = json.loads(gdf.to_json())
    except Exception as e:
        print(f"Error loading region file: {e}")
        print(
            "Supported formats: GeoJSON, TopoJSON, Shapefile (.shp), GeoPackage (.gpkg)"
        )
        sys.exit(1)

    # Generate static tessera tiles (always enabled now)
    tiles_dir = None
    temp_tiles_dir = None

    if args.tiles_output:
        # Use specified output directory
        tiles_output_base = Path(args.tiles_output)
        tiles_output_base.mkdir(parents=True, exist_ok=True)
        print(f"Generating static tiles to: {tiles_output_base}")
    else:
        # Use temporary directory
        temp_tiles_dir = tempfile.mkdtemp(prefix="tessera_static_tiles_")
        tiles_output_base = Path(temp_tiles_dir)
        print(f"Generating static tiles to temporary directory: {tiles_output_base}")

    tiles_dir = generate_static_tessera_tiles(
        str(region_path),
        str(tiles_output_base),
        tessera_version=args.dataset_version,
        year=args.year,
        bands=args.bands,
        registry_dir=args.registry_dir,
        auto_update=args.auto_update,
        manifests_repo_url=args.manifests_repo_url,
    )

    if not tiles_dir:
        print("Warning: Failed to generate tessera tiles, continuing without them")
    else:
        print("Static tessera tiles ready for serving")

    # Create custom handler with GeoJSON data and tiles
    def handler_factory(*args, **kwargs):
        return GeoJSONRequestHandler(
            *args, geojson_data=geojson_data, tiles_dir=tiles_dir, **kwargs
        )

    # Start HTTP server
    port = args.port
    try:
        with socketserver.TCPServer(("", port), handler_factory) as httpd:
            print(f"Server started at http://localhost:{port}")
            if tiles_dir:
                print(f"Serving tessera tiles from: {tiles_dir}")
            print("Press Ctrl+C to stop the server")

            # Open browser if requested
            if args.open:
                webbrowser.open(f"http://localhost:{port}")

            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary tiles directory only if we created one
        if temp_tiles_dir and os.path.exists(temp_tiles_dir):
            print(f"Cleaning up temporary tiles directory: {temp_tiles_dir}")
            shutil.rmtree(temp_tiles_dir)
            print("Temporary tiles directory cleaned up")


def main():
    """Main entry point for the GeoTessera command-line interface.

    Parses command line arguments and routes to appropriate subcommands.
    Provides comprehensive help text with examples and usage instructions.

    Available commands:
        - list-embeddings: Browse available embedding tiles
        - info: Display dataset information
        - map: Generate coverage map visualization
        - visualize: Create GeoTIFF from embeddings
        - serve: Launch interactive web interface

    For checksum management, use: geotessera-registry hash

    Each command has its own help accessible via:
        geotessera <command> --help
    """
    parser = argparse.ArgumentParser(
        description="GeoTessera - Access geospatial embeddings and create land masks for alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available embeddings
  geotessera list-embeddings --limit 10
  
  # Get information about the dataset
  geotessera info
  
  # Generate a world map showing all available embedding grid points
  geotessera map --output coverage_map.png
  
  # Create full 128-band GeoTIFF (default behavior)
  geotessera visualize --region region.geojson --output full_128band.tif
  
  # Create GeoTIFF with specific bands
  geotessera visualize --region boundary.shp --output subset.tif --bands 0 1 2
  geotessera visualize --region study_area.gpkg --output selected_bands.tif --bands 10 20 30
  
  # Start HTTP server with Leaflet.js to display region overlay with tessera false color tiles
  geotessera serve --region example/CB.geojson --port 8000 --open
  geotessera serve --region boundary.shp --port 8080 --bands 25 50 75 --open
  
  # Serve with custom band selection for tessera visualization
  geotessera serve --region example/CB.geojson --bands 0 1 2 --year 2024 --open
  
  # Generate static tiles to a specific directory and serve them
  geotessera serve --region example/CB.geojson --tiles-output ./static_tiles --open
  
  # Generate SHA256 checksums (use geotessera-registry instead)
  geotessera-registry hash /path/to/data

Valid Target CRS Values:
  EPSG:4326     - WGS84 Geographic (lat/lon) - good for global/large areas
  EPSG:326XX    - UTM Northern Hemisphere (XX = zone 01-60, e.g., EPSG:32630)
  EPSG:327XX    - UTM Southern Hemisphere (XX = zone 01-60, e.g., EPSG:32730)  
  EPSG:3995     - Arctic Polar Stereographic (for areas north of 70°N)
  EPSG:3031     - Antarctic Polar Stereographic (for areas south of 70°S)

Note: The 'visualize' command creates GeoTIFFs from embeddings. Default behavior exports 
      all 128 bands. Use --bands to export only selected channels. All outputs are float32.
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"geotessera {__version__}"
    )
    parser.add_argument(
        "--dataset-version", default="v1", help="Dataset version (default: v1)"
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        help="Local directory containing registry files (if not specified, uses TESSERA_REGISTRY_DIR env var or auto-clones tessera-manifests)",
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Update tessera-manifests repository to latest version before use",
    )
    parser.add_argument(
        "--manifests-repo-url",
        default="https://github.com/ucam-eo/tessera-manifests.git",
        help="Git repository URL for tessera-manifests (default: official repository)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List embeddings command
    list_parser = subparsers.add_parser(
        "list-embeddings", help="List available embeddings"
    )
    list_parser.add_argument("--limit", type=int, help="Limit number of results shown")
    list_parser.set_defaults(func=list_embeddings_command)

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show comprehensive dataset information"
    )
    info_parser.set_defaults(func=info_command)

    # Map command
    map_parser = subparsers.add_parser(
        "map", help="Generate a world map showing all available embedding grid points"
    )
    map_parser.add_argument(
        "--output",
        type=str,
        default="embedding_coverage_map.png",
        help="Output map file path (default: embedding_coverage_map.png)",
    )
    map_parser.set_defaults(func=map_command)

    # Visualize command (embedding visualization)
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Create GeoTIFF from embeddings - full 128-band or selected bands (float32)",
    )
    viz_parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region file to visualize (GeoJSON, TopoJSON, Shapefile, GeoPackage)",
    )
    viz_parser.add_argument(
        "--output",
        type=str,
        default="region_visualization.tiff",
        help="Output visualization file path",
    )
    viz_parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS (default: EPSG:4326). See help for valid values.",
    )
    viz_parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        help="Band indices to include in output. If not specified, exports all 128 bands. For specific band selection, e.g., --bands 0 1 2",
    )
    viz_parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year of embeddings to visualize (default: 2024)",
    )
    viz_parser.set_defaults(func=visualize_command)

    # Serve command (HTTP server with Leaflet.js)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start HTTP server with Leaflet.js to display region overlay with tessera false color tiles",
    )
    serve_parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region file to display (GeoJSON, TopoJSON, Shapefile, GeoPackage)",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP server (default: 8000)"
    )
    serve_parser.add_argument(
        "--open", action="store_true", help="Open browser automatically"
    )
    serve_parser.add_argument(
        "--tiles-output",
        type=str,
        help="Directory to save generated tiles (default: temporary directory)",
    )
    serve_parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Band indices for tessera visualization. Default is RGB visualization with bands 0, 1, 2",
    )
    serve_parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year of embeddings for tessera visualization (default: 2024)",
    )
    serve_parser.set_defaults(func=serve_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
