"""GeoTessera: Simplified access to satellite embeddings as standardized GeoTIFFs.

GeoTessera provides a focused Python interface for downloading Tessera satellite
embeddings as individual GeoTIFF files with accurate metadata. This simplified
approach enables seamless integration with standard GIS workflows and downstream
processing libraries.

Key Features:
    - Download tiles within bounding boxes as numpy arrays
    - Export individual tiles as standards-compliant GeoTIFF files
    - Accurate georeferencing and metadata for each tile
    - Separate visualization utilities for creating maps and web viewers
    - Clean CLI interface for common workflows

Simplified Usage:
    >>> from geotessera import GeoTessera
    >>> gt = GeoTessera()
    >>>
    >>> # Fetch embedding tiles in a bounding box
    >>> bbox = (-0.2, 51.4, 0.1, 51.6)  # London area
    >>> tiles = gt.fetch_embeddings(bbox, year=2024)
    >>>
    >>> # Export as individual GeoTIFF files
    >>> files = gt.export_embedding_geotiffs(
    ...     bbox=bbox, 
    ...     output_dir="tiles/",
    ...     bands=[0, 1, 2]  # Select specific bands
    ... )
    >>>
    >>> # Or export a single tile
    >>> file = gt.export_embedding_geotiff(
    ...     lat=51.55, lon=-0.05,
    ...     output_path="single_tile.tif"
    ... )

The exported GeoTIFF files contain:
    - Accurate georeferencing (EPSG:4326)
    - Comprehensive metadata tags
    - Proper band descriptions
    - Standard compression and tiling

For visualization and analysis, use the separate visualization module:
    >>> from geotessera.visualization import create_rgb_mosaic_from_geotiffs
    >>> create_rgb_mosaic_from_geotiffs(files, "mosaic.tif")

This design enables the use of standard GIS tools and libraries for all
downstream processing, keeping GeoTessera focused on reliable data access.
"""

from .core import GeoTessera
from . import visualization
from . import registry

try:
    import importlib.metadata
    __version__ = importlib.metadata.version("geotessera")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development installs
    __version__ = "unknown"

__all__ = ["GeoTessera", "visualization", "registry"]