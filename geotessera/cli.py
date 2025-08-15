"""Simplified GeoTessera command-line interface.

Focused on downloading tiles and creating visualizations from the generated GeoTIFFs.
"""

import argparse
import sys
from pathlib import Path
import json

from .core import GeoTessera
from .visualization import (
    calculate_bbox_from_file,
    calculate_bbox_from_points,
    create_rgb_mosaic_from_geotiffs,
    geotiff_to_web_tiles,
    create_simple_web_viewer,
    create_coverage_summary_map,
    analyze_geotiff_coverage
)


def download_command(args):
    """Export region of interest as discrete GeoTIFF files with native UTM projections.
    
    Each tile is exported as a separate GeoTIFF file preserving the original UTM 
    coordinate system from the corresponding landmask tile. Files are not merged,
    allowing individual tile inspection and processing.
    """
    gt = GeoTessera(
        dataset_version=args.dataset_version,
        cache_dir=args.cache_dir,
        registry_dir=args.registry_dir
    )
    
    # Parse bounding box
    if args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
        if len(bbox) != 4:
            print("Error: bbox must be 'min_lon,min_lat,max_lon,max_lat'")
            return
        print(f"Using bounding box: {bbox}")
    elif args.region_file:
        try:
            bbox = calculate_bbox_from_file(args.region_file)
            print(f"Calculated bbox from {args.region_file}: {bbox}")
            print(f"  - Longitude range: {bbox[0]:.6f} to {bbox[2]:.6f}")
            print(f"  - Latitude range: {bbox[1]:.6f} to {bbox[3]:.6f}")
        except Exception as e:
            print(f"Error reading region file: {e}")
            print("Supported formats: GeoJSON, Shapefile, etc.")
            return
    else:
        print("Error: Must specify either --bbox or --region-file")
        print("Examples:")
        print("  --bbox '-0.2,51.4,0.1,51.6'  # London area")
        print("  --region-file london.geojson  # From GeoJSON file")
        return
    
    # Parse bands
    bands = None
    if args.bands:
        try:
            bands = list(map(int, args.bands.split(',')))
            print(f"Exporting {len(bands)} selected bands: {bands}")
        except ValueError:
            print("Error: bands must be comma-separated integers (0-127)")
            print("Example: --bands '0,1,2' for first 3 bands")
            return
    else:
        print("Exporting all 128 bands")
    
    print(f"\\nRegion of Interest Export:")
    print(f"  Year: {args.year}")
    print(f"  Output directory: {args.output}")
    print(f"  Compression: {args.compress}")
    print(f"  Dataset version: {args.dataset_version}")
    
    try:
        # Export tiles as discrete GeoTIFFs with UTM projections
        print(f"\\n🔄 Fetching embedding tiles and exporting as discrete GeoTIFFs...")
        files = gt.export_embedding_geotiffs(
            bbox=bbox,
            output_dir=args.output,
            year=args.year,
            bands=bands,
            compress=args.compress
        )
        
        if not files:
            print("⚠️  No tiles found in the specified region.")
            print("Try expanding your bounding box or checking data availability.")
            return
        
        print(f"\\n✅ SUCCESS: Exported {len(files)} discrete GeoTIFF files")
        print(f"   Each file preserves its native UTM projection from landmask tiles")
        print(f"   Files can be individually inspected and processed")
        
        if args.verbose or args.list_files:
            print(f"\\n📁 Created files:")
            for i, f in enumerate(files, 1):
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                print(f"  {i:2d}. {file_path.name} ({file_size:,} bytes)")
        elif len(files) > 0:
            print(f"\\n📁 Sample files (use --verbose or --list-files to see all):")
            for f in files[:3]:
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                print(f"     {file_path.name} ({file_size:,} bytes)")
            if len(files) > 3:
                print(f"     ... and {len(files) - 3} more files")
        
        # Show tile coordinate information
        print(f"\\n🗺️  Spatial Information:")
        if args.verbose:
            try:
                import rasterio
                # Sample the first file to show projection info
                with rasterio.open(files[0]) as src:
                    print(f"   CRS: {src.crs}")
                    print(f"   Transform: {src.transform}")
                    print(f"   Dimensions: {src.width} x {src.height} pixels")
                    print(f"   Data type: {src.dtypes[0]}")
            except Exception:
                pass
        
        print(f"   Output directory: {Path(args.output).resolve()}")
        print(f"\\n💡 Next steps:")
        print(f"   - Inspect individual tiles with QGIS, GDAL, or rasterio")
        print(f"   - Use 'gdalinfo <filename>' to see projection details")
        print(f"   - Process tiles individually or in groups as needed")
                
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        if args.verbose:
            import traceback
            print("\\nFull traceback:")
            traceback.print_exc()
        return


def visualize_command(args):
    """Create visualizations from GeoTIFF files."""
    # Find GeoTIFF files
    input_dir = Path(args.input)
    if input_dir.is_file():
        geotiff_paths = [str(input_dir)]
    else:
        geotiff_paths = list(map(str, input_dir.glob("*.tif")))
        geotiff_paths.extend(map(str, input_dir.glob("*.tiff")))
    
    if not geotiff_paths:
        print(f"No GeoTIFF files found in {args.input}")
        return
        
    print(f"Found {len(geotiff_paths)} GeoTIFF files")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == 'rgb':
        # Create RGB mosaic
        bands = [0, 1, 2]  # Default RGB bands
        if args.bands:
            bands = list(map(int, args.bands.split(',')))
            if len(bands) != 3:
                print("Error: RGB visualization requires exactly 3 bands")
                return
                
        output_path = output_dir / "rgb_mosaic.tif"
        
        try:
            created_file = create_rgb_mosaic_from_geotiffs(
                geotiff_paths=geotiff_paths,
                output_path=str(output_path),
                bands=tuple(bands),
                normalize=args.normalize
            )
            print(f"Created RGB mosaic: {created_file}")
            
        except Exception as e:
            print(f"Error creating RGB mosaic: {e}")
            return
            
    elif args.type == 'web':
        # Create web tiles
        if len(geotiff_paths) > 1:
            # First create RGB mosaic
            mosaic_path = output_dir / "temp_mosaic.tif"
            bands = [0, 1, 2]
            if args.bands:
                bands = list(map(int, args.bands.split(',')))[:3]
                
            try:
                create_rgb_mosaic_from_geotiffs(
                    geotiff_paths=geotiff_paths,
                    output_path=str(mosaic_path),
                    bands=tuple(bands),
                    normalize=args.normalize
                )
                source_file = str(mosaic_path)
                
            except Exception as e:
                print(f"Error creating mosaic for web tiles: {e}")
                return
        else:
            source_file = geotiff_paths[0]
            
        # Generate web tiles
        tiles_dir = output_dir / "tiles"
        
        try:
            geotiff_to_web_tiles(
                geotiff_path=source_file,
                output_dir=str(tiles_dir),
                zoom_levels=(args.min_zoom, args.max_zoom)
            )
            
            # Create HTML viewer
            html_path = output_dir / "viewer.html"
            
            # Calculate center from coverage
            coverage = analyze_geotiff_coverage(geotiff_paths)
            bounds = coverage["bounds"]
            center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
            center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
            
            create_simple_web_viewer(
                tiles_dir=str(tiles_dir),
                output_html=str(html_path),
                center_lat=center_lat,
                center_lon=center_lon,
                zoom=args.initial_zoom,
                title=f"GeoTessera - {args.input}"
            )
            
            print(f"Created web tiles in: {tiles_dir}")
            print(f"Created viewer: {html_path}")
            print(f"Open {html_path} in a web browser to view")
            
        except Exception as e:
            print(f"Error creating web visualization: {e}")
            return
            
    elif args.type == 'coverage':
        # Create coverage map
        html_path = output_dir / "coverage.html"
        
        try:
            create_coverage_summary_map(
                geotiff_paths=geotiff_paths,
                output_html=str(html_path),
                title=f"Coverage Map - {args.input}"
            )
            
            print(f"Created coverage map: {html_path}")
            print(f"Open {html_path} in a web browser to view")
            
        except Exception as e:
            print(f"Error creating coverage map: {e}")
            return


def info_command(args):
    """Show information about GeoTIFF files or library."""
    if args.geotiffs:
        # Analyze GeoTIFF files
        input_path = Path(args.geotiffs)
        if input_path.is_file():
            geotiff_paths = [str(input_path)]
        else:
            geotiff_paths = list(map(str, input_path.glob("*.tif")))
            geotiff_paths.extend(map(str, input_path.glob("*.tiff")))
            
        if not geotiff_paths:
            print(f"No GeoTIFF files found in {args.geotiffs}")
            return
            
        coverage = analyze_geotiff_coverage(geotiff_paths)
        
        print("=== GeoTIFF Analysis ===")
        print(f"Total files: {coverage['total_files']}")
        print(f"Years: {', '.join(coverage['years'])}")
        print(f"CRS: {', '.join(coverage['crs'])}")
        
        bounds = coverage['bounds']
        print(f"\\nBounding box:")
        print(f"  Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")
        print(f"  Latitude: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
        
        print(f"\\nBand counts:")
        for bands, count in coverage['band_counts'].items():
            print(f"  {bands} bands: {count} files")
            
        if args.verbose:
            print(f"\\nFirst 10 tiles:")
            for i, tile in enumerate(coverage['tiles'][:10]):
                print(f"  {Path(tile['path']).name}: ({tile['tile_lat']}, {tile['tile_lon']}) - {tile['bands']} bands")
                
    else:
        # Show library info
        gt = GeoTessera(dataset_version=getattr(args, 'dataset_version', 'v1'))
        years = gt.get_available_years()
        
        print("=== GeoTessera Library Info ===")
        print(f"Version: {gt.version}")
        print(f"Available years: {', '.join(map(str, years))}")
        print(f"Registry loaded blocks: {len(gt.registry.loaded_blocks)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GeoTessera: Download satellite embedding tiles as GeoTIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument('--dataset-version', default='v1', help='Tessera dataset version (e.g., v1, v2)')
    parser.add_argument('--cache-dir', help='Cache directory')
    parser.add_argument('--registry-dir', help='Registry directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Export region as discrete GeoTIFFs with UTM projections')
    download_parser.add_argument('--bbox', help='Bounding box: min_lon,min_lat,max_lon,max_lat')
    download_parser.add_argument('--region-file', help='GeoJSON/Shapefile to define region')
    download_parser.add_argument('--year', type=int, default=2024, help='Year of embeddings')
    download_parser.add_argument('--bands', help='Comma-separated band indices (default: all 128)')
    download_parser.add_argument('--output', '-o', required=True, help='Output directory')
    download_parser.add_argument('--compress', default='lzw', help='Compression method')
    download_parser.add_argument('--list-files', action='store_true', help='List all created files with details')
    download_parser.set_defaults(func=download_command)
    
    # Visualize command  
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations from GeoTIFFs')
    viz_parser.add_argument('input', help='Input GeoTIFF file or directory')
    viz_parser.add_argument('--output', '-o', required=True, help='Output directory')
    viz_parser.add_argument('--type', choices=['rgb', 'web', 'coverage'], default='rgb',
                           help='Visualization type')
    viz_parser.add_argument('--bands', help='Comma-separated band indices')
    viz_parser.add_argument('--normalize', action='store_true', help='Normalize bands')
    viz_parser.add_argument('--min-zoom', type=int, default=8, help='Min zoom for web tiles')
    viz_parser.add_argument('--max-zoom', type=int, default=15, help='Max zoom for web tiles')
    viz_parser.add_argument('--initial-zoom', type=int, default=10, help='Initial zoom level')
    viz_parser.set_defaults(func=visualize_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information')
    info_parser.add_argument('--geotiffs', help='Analyze GeoTIFF files/directory')
    info_parser.set_defaults(func=info_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()