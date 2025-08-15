"""Simplified GeoTessera command-line interface.

Focused on downloading tiles and creating visualizations from the generated GeoTIFFs.
"""

import sys
from pathlib import Path
from typing import Optional, List, Callable
from typing_extensions import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

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

app = typer.Typer(
    name="geotessera",
    help="GeoTessera: Download satellite embedding tiles as GeoTIFFs",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()


def create_progress_callback(progress: Progress, task_id: TaskID) -> Callable:
    """Create a progress callback for core library operations."""
    def progress_callback(current: int, total: int, status: str = None):
        if status:
            progress.update(task_id, completed=current, total=total, status=status)
        else:
            progress.update(task_id, completed=current, total=total)
    return progress_callback


@app.command()
def download(
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output directory"
    )],
    bbox: Annotated[Optional[str], typer.Option(
        "--bbox",
        help="Bounding box: 'min_lon,min_lat,max_lon,max_lat'"
    )] = None,
    region_file: Annotated[Optional[Path], typer.Option(
        "--region-file",
        help="GeoJSON/Shapefile to define region",
        exists=True
    )] = None,
    year: Annotated[int, typer.Option(
        "--year",
        help="Year of embeddings"
    )] = 2024,
    bands: Annotated[Optional[str], typer.Option(
        "--bands",
        help="Comma-separated band indices (default: all 128)"
    )] = None,
    compress: Annotated[str, typer.Option(
        "--compress",
        help="Compression method"
    )] = "lzw",
    list_files: Annotated[bool, typer.Option(
        "--list-files",
        help="List all created files with details"
    )] = False,
    dataset_version: Annotated[str, typer.Option(
        "--dataset-version",
        help="Tessera dataset version (e.g., v1, v2)"
    )] = "v1",
    cache_dir: Annotated[Optional[Path], typer.Option(
        "--cache-dir",
        help="Cache directory"
    )] = None,
    registry_dir: Annotated[Optional[Path], typer.Option(
        "--registry-dir",
        help="Registry directory"
    )] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Verbose output"
    )] = False
):
    """Export region as discrete GeoTIFFs with UTM projections.
    
    Each tile is exported as a separate GeoTIFF file preserving the original UTM 
    coordinate system from the corresponding landmask tile. Files are not merged,
    allowing individual tile inspection and processing.
    """
    
    # Initialize GeoTessera
    gt = GeoTessera(
        dataset_version=dataset_version,
        cache_dir=str(cache_dir) if cache_dir else None,
        registry_dir=str(registry_dir) if registry_dir else None
    )
    
    # Parse bounding box
    if bbox:
        try:
            bbox_coords = tuple(map(float, bbox.split(',')))
            if len(bbox_coords) != 4:
                rprint("[red]Error: bbox must be 'min_lon,min_lat,max_lon,max_lat'[/red]")
                raise typer.Exit(1)
            rprint(f"[green]Using bounding box:[/green] {bbox_coords}")
        except ValueError:
            rprint("[red]Error: Invalid bbox format. Use: 'min_lon,min_lat,max_lon,max_lat'[/red]")
            raise typer.Exit(1)
    elif region_file:
        try:
            bbox_coords = calculate_bbox_from_file(region_file)
            rprint(f"[green]Calculated bbox from {region_file}:[/green] {bbox_coords}")
            rprint(f"  • Longitude range: {bbox_coords[0]:.6f} to {bbox_coords[2]:.6f}")
            rprint(f"  • Latitude range: {bbox_coords[1]:.6f} to {bbox_coords[3]:.6f}")
        except Exception as e:
            rprint(f"[red]Error reading region file: {e}[/red]")
            rprint("Supported formats: GeoJSON, Shapefile, etc.")
            raise typer.Exit(1)
    else:
        rprint("[red]Error: Must specify either --bbox or --region-file[/red]")
        rprint("Examples:")
        rprint("  --bbox '-0.2,51.4,0.1,51.6'  # London area")
        rprint("  --region-file london.geojson  # From GeoJSON file")
        raise typer.Exit(1)
    
    # Parse bands
    bands_list = None
    if bands:
        try:
            bands_list = list(map(int, bands.split(',')))
            rprint(f"[blue]Exporting {len(bands_list)} selected bands:[/blue] {bands_list}")
        except ValueError:
            rprint("[red]Error: bands must be comma-separated integers (0-127)[/red]")
            rprint("Example: --bands '0,1,2' for first 3 bands")
            raise typer.Exit(1)
    else:
        rprint("[blue]Exporting all 128 bands[/blue]")
    
    # Display export info
    info_table = Table(show_header=False, box=None)
    info_table.add_row("Year:", str(year))
    info_table.add_row("Output directory:", str(output))
    info_table.add_row("Compression:", compress)
    info_table.add_row("Dataset version:", dataset_version)
    
    rprint(Panel(info_table, title="[bold]Region of Interest Export[/bold]", border_style="blue"))
    
    try:
        # Export tiles with progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("🔄 Processing tiles...", total=100, status="Starting...")
            
            # Export with progress callback
            files = gt.export_embedding_geotiffs(
                bbox=bbox_coords,
                output_dir=output,
                year=year,
                bands=bands_list,
                compress=compress,
                progress_callback=create_progress_callback(progress, task)
            )
        
        if not files:
            rprint("[yellow]⚠️  No tiles found in the specified region.[/yellow]")
            rprint("Try expanding your bounding box or checking data availability.")
            return
        
        rprint(f"\n[green]✅ SUCCESS: Exported {len(files)} discrete GeoTIFF files[/green]")
        rprint("   Each file preserves its native UTM projection from landmask tiles")
        rprint("   Files can be individually inspected and processed")
        
        if verbose or list_files:
            rprint(f"\n[blue]📁 Created files:[/blue]")
            file_table = Table(show_header=True, header_style="bold blue")
            file_table.add_column("#", style="dim", width=3)
            file_table.add_column("Filename")
            file_table.add_column("Size", justify="right")
            
            for i, f in enumerate(files, 1):
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                file_table.add_row(str(i), file_path.name, f"{file_size:,} bytes")
            
            console.print(file_table)
        elif len(files) > 0:
            rprint(f"\n[blue]📁 Sample files (use --verbose or --list-files to see all):[/blue]")
            for f in files[:3]:
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                rprint(f"     {file_path.name} ({file_size:,} bytes)")
            if len(files) > 3:
                rprint(f"     ... and {len(files) - 3} more files")
        
        # Show spatial information
        rprint(f"\n[blue]🗺️  Spatial Information:[/blue]")
        if verbose:
            try:
                import rasterio
                with rasterio.open(files[0]) as src:
                    rprint(f"   CRS: {src.crs}")
                    rprint(f"   Transform: {src.transform}")
                    rprint(f"   Dimensions: {src.width} x {src.height} pixels")
                    rprint(f"   Data type: {src.dtypes[0]}")
            except Exception:
                pass
        
        rprint(f"   Output directory: {Path(output).resolve()}")
        
        tips_table = Table(show_header=False, box=None)
        tips_table.add_row("• Inspect individual tiles with QGIS, GDAL, or rasterio")
        tips_table.add_row("• Use 'gdalinfo <filename>' to see projection details")
        tips_table.add_row("• Process tiles individually or in groups as needed")
        
        rprint(Panel(tips_table, title="[bold]💡 Next steps[/bold]", border_style="green"))
                
    except Exception as e:
        rprint(f"\n[red]❌ Error: {e}[/red]")
        if verbose:
            import traceback
            rprint("\n[dim]Full traceback:[/dim]")
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def visualize(
    input_path: Annotated[Path, typer.Argument(
        help="Input GeoTIFF file or directory"
    )],
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output directory"
    )],
    vis_type: Annotated[str, typer.Option(
        "--type",
        help="Visualization type"
    )] = "rgb",
    bands: Annotated[Optional[str], typer.Option(
        "--bands",
        help="Comma-separated band indices"
    )] = None,
    normalize: Annotated[bool, typer.Option(
        "--normalize",
        help="Normalize bands"
    )] = False,
    min_zoom: Annotated[int, typer.Option(
        "--min-zoom",
        help="Min zoom for web tiles"
    )] = 8,
    max_zoom: Annotated[int, typer.Option(
        "--max-zoom",
        help="Max zoom for web tiles"
    )] = 15,
    initial_zoom: Annotated[int, typer.Option(
        "--initial-zoom",
        help="Initial zoom level"
    )] = 10
):
    """Create visualizations from GeoTIFF files."""
    
    # Validate visualization type
    if vis_type not in ['rgb', 'web', 'coverage']:
        rprint(f"[red]Error: Invalid visualization type '{vis_type}'. Must be one of: rgb, web, coverage[/red]")
        raise typer.Exit(1)
    
    # Find GeoTIFF files
    if input_path.is_file():
        geotiff_paths = [str(input_path)]
    else:
        geotiff_paths = list(map(str, input_path.glob("*.tif")))
        geotiff_paths.extend(map(str, input_path.glob("*.tiff")))
    
    if not geotiff_paths:
        rprint(f"[red]No GeoTIFF files found in {input_path}[/red]")
        raise typer.Exit(1)
        
    rprint(f"[blue]Found {len(geotiff_paths)} GeoTIFF files[/blue]")
    
    output.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        if vis_type == 'rgb':
            # Create RGB mosaic
            bands_list = [0, 1, 2]  # Default RGB bands
            if bands:
                bands_list = list(map(int, bands.split(',')))
                if len(bands_list) != 3:
                    rprint("[red]Error: RGB visualization requires exactly 3 bands[/red]")
                    raise typer.Exit(1)
                    
            output_path = output / "rgb_mosaic.tif"
            
            task = progress.add_task("Creating RGB mosaic...", total=100, status="Starting...")
            
            try:
                created_file = create_rgb_mosaic_from_geotiffs(
                    geotiff_paths=geotiff_paths,
                    output_path=str(output_path),
                    bands=tuple(bands_list),
                    normalize=normalize,
                    progress_callback=create_progress_callback(progress, task)
                )
                rprint(f"[green]Created RGB mosaic: {created_file}[/green]")
                
            except Exception as e:
                rprint(f"[red]Error creating RGB mosaic: {e}[/red]")
                raise typer.Exit(1)
                
        elif vis_type == 'web':
            # Create web tiles
            if len(geotiff_paths) > 1:
                # First create RGB mosaic
                mosaic_path = output / "temp_mosaic.tif"
                bands_list = [0, 1, 2]
                if bands:
                    bands_list = list(map(int, bands.split(',')))[:3]
                    
                task1 = progress.add_task("Creating mosaic for web tiles...", total=50, status="Starting...")
                
                try:
                    # Create wrapper callback for the first phase (0-50%)
                    def mosaic_progress_callback(current: int, total: int, status: str = None):
                        overall_progress = int((current / total) * 50)
                        create_progress_callback(progress, task1)(overall_progress, 50, status)
                    
                    create_rgb_mosaic_from_geotiffs(
                        geotiff_paths=geotiff_paths,
                        output_path=str(mosaic_path),
                        bands=tuple(bands_list),
                        normalize=normalize,
                        progress_callback=mosaic_progress_callback
                    )
                    progress.update(task1, completed=50)
                    source_file = str(mosaic_path)
                    
                except Exception as e:
                    rprint(f"[red]Error creating mosaic for web tiles: {e}[/red]")
                    raise typer.Exit(1)
            else:
                source_file = geotiff_paths[0]
                
            # Generate web tiles
            tiles_dir = output / "tiles"
            
            task2 = progress.add_task("Generating web tiles...", total=100, status="Starting...")
            
            try:
                geotiff_to_web_tiles(
                    geotiff_path=source_file,
                    output_dir=str(tiles_dir),
                    zoom_levels=(min_zoom, max_zoom)
                )
                progress.update(task2, completed=80)
                
                # Create HTML viewer
                html_path = output / "viewer.html"
                
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
                    zoom=initial_zoom,
                    title=f"GeoTessera - {input_path}"
                )
                
                progress.update(task2, completed=100)
                
                rprint(f"[green]Created web tiles in: {tiles_dir}[/green]")
                rprint(f"[green]Created viewer: {html_path}[/green]")
                rprint(f"[blue]Open {html_path} in a web browser to view[/blue]")
                
            except Exception as e:
                rprint(f"[red]Error creating web visualization: {e}[/red]")
                raise typer.Exit(1)
                
        elif vis_type == 'coverage':
            # Create coverage map
            html_path = output / "coverage.html"
            
            task = progress.add_task("Creating coverage map...", total=100, status="Starting...")
            
            try:
                create_coverage_summary_map(
                    geotiff_paths=geotiff_paths,
                    output_html=str(html_path),
                    title=f"Coverage Map - {input_path}"
                )
                
                progress.update(task, completed=100)
                
                rprint(f"[green]Created coverage map: {html_path}[/green]")
                rprint(f"[blue]Open {html_path} in a web browser to view[/blue]")
                
            except Exception as e:
                rprint(f"[red]Error creating coverage map: {e}[/red]")
                raise typer.Exit(1)


@app.command()
def info(
    geotiffs: Annotated[Optional[Path], typer.Option(
        "--geotiffs",
        help="Analyze GeoTIFF files/directory"
    )] = None,
    dataset_version: Annotated[str, typer.Option(
        "--dataset-version",
        help="Tessera dataset version (e.g., v1, v2)"
    )] = "v1",
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Verbose output"
    )] = False
):
    """Show information about GeoTIFF files or library."""
    
    if geotiffs:
        # Analyze GeoTIFF files
        if geotiffs.is_file():
            geotiff_paths = [str(geotiffs)]
        else:
            geotiff_paths = list(map(str, geotiffs.glob("*.tif")))
            geotiff_paths.extend(map(str, geotiffs.glob("*.tiff")))
            
        if not geotiff_paths:
            rprint(f"[red]No GeoTIFF files found in {geotiffs}[/red]")
            raise typer.Exit(1)
            
        coverage = analyze_geotiff_coverage(geotiff_paths)
        
        # Create analysis table
        analysis_table = Table(show_header=False, box=None)
        analysis_table.add_row("Total files:", str(coverage['total_files']))
        analysis_table.add_row("Years:", ', '.join(coverage['years']))
        analysis_table.add_row("CRS:", ', '.join(coverage['crs']))
        
        rprint(Panel(analysis_table, title="[bold]📊 GeoTIFF Analysis[/bold]", border_style="blue"))
        
        bounds = coverage['bounds']
        
        bounds_table = Table(show_header=False, box=None)
        bounds_table.add_row("Longitude:", f"{bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")
        bounds_table.add_row("Latitude:", f"{bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
        
        rprint(Panel(bounds_table, title="[bold]🗺️ Bounding Box[/bold]", border_style="green"))
        
        bands_table = Table(show_header=True, header_style="bold blue")
        bands_table.add_column("Band Count")
        bands_table.add_column("Files", justify="right")
        
        for bands_count, count in coverage['band_counts'].items():
            bands_table.add_row(f"{bands_count} bands", str(count))
            
        rprint(Panel(bands_table, title="[bold]🎵 Band Information[/bold]", border_style="cyan"))
            
        if verbose:
            tiles_table = Table(show_header=True, header_style="bold blue")
            tiles_table.add_column("Filename")
            tiles_table.add_column("Coordinates")
            tiles_table.add_column("Bands", justify="right")
            
            for tile in coverage['tiles'][:10]:
                tiles_table.add_row(
                    Path(tile['path']).name,
                    f"({tile['tile_lat']}, {tile['tile_lon']})",
                    str(tile['bands'])
                )
                
            rprint(Panel(tiles_table, title="[bold]📁 First 10 Tiles[/bold]", border_style="yellow"))
                
    else:
        # Show library info
        gt = GeoTessera(dataset_version=dataset_version)
        years = gt.get_available_years()
        
        info_table = Table(show_header=False, box=None)
        info_table.add_row("Version:", gt.version)
        info_table.add_row("Available years:", ', '.join(map(str, years)))
        info_table.add_row("Registry loaded blocks:", str(len(gt.registry.loaded_blocks)))
        
        rprint(Panel(info_table, title="[bold]🌍 GeoTessera Library Info[/bold]", border_style="blue"))


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()