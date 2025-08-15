#!/usr/bin/env python3
"""
Command-line interface for managing GeoTessera registry files.

This module provides tools for generating and maintaining Pooch registry files
used by the GeoTessera package. It supports parallel processing, incremental
updates, and generation of a master registry index.
"""

import os
import hashlib
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .registry import (
    get_block_coordinates,
    get_embeddings_registry_filename,
    get_landmasks_registry_filename,
    parse_grid_coordinates,
)


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def process_file(args):
    """Process a single file and return its relative path and hash."""
    file_path, base_dir, skip_checksum = args
    try:
        rel_path = os.path.relpath(file_path, base_dir)
        if skip_checksum:
            file_hash = ""
        else:
            file_hash = calculate_sha256(file_path)
        return rel_path, file_hash
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def load_existing_registry(registry_path):
    """Load existing registry file into a dictionary."""
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        registry[parts[0]] = parts[1]
    return registry


def find_npy_files_by_blocks(base_dir):
    """Find all .npy files and organize them by year and block."""
    files_by_year_and_block = defaultdict(lambda: defaultdict(list))

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)

                # Extract year from path (assuming format ./YYYY/...)
                path_parts = rel_path.split(os.sep)
                if (
                    len(path_parts) > 0
                    and path_parts[0].isdigit()
                    and len(path_parts[0]) == 4
                ):
                    year = path_parts[0]

                    # Extract coordinates from the grid directory name
                    grid_dir = os.path.basename(os.path.dirname(file_path))
                    lon, lat = parse_grid_coordinates(grid_dir)

                    if lon is not None and lat is not None:
                        block_lon, block_lat = get_block_coordinates(lon, lat)
                        block_key = (block_lon, block_lat)
                        files_by_year_and_block[year][block_key].append(file_path)

    return files_by_year_and_block


def find_tiff_files_by_blocks(base_dir):
    """Find all .tiff files and organize them by block."""
    files_by_block = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".tiff"):
                file_path = os.path.join(root, file)

                # Extract coordinates from the tiff filename (e.g., grid_-120.55_53.45.tiff)
                filename = os.path.basename(file_path)
                tiff_name = filename.replace(".tiff", "")
                lon, lat = parse_grid_coordinates(tiff_name)

                if lon is not None and lat is not None:
                    block_lon, block_lat = get_block_coordinates(lon, lat)
                    block_key = (block_lon, block_lat)
                    files_by_block[block_key].append(file_path)

    return files_by_block


def generate_master_registry(registry_dir):
    """Generate a master registry.txt file containing hashes of all registry files."""
    # This function is no longer used but kept for compatibility
    # The actual generation of registry.txt should be done separately
    pass


def list_command(args):
    """List existing registry files in the specified directory."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    print(f"Scanning for registry files in: {base_dir}")

    # Find all embeddings_*.txt and landmasks_*.txt files
    registry_files = []
    for file in os.listdir(base_dir):
        if (
            file.startswith("embeddings_") or file.startswith("landmasks_")
        ) and file.endswith(".txt"):
            registry_path = os.path.join(base_dir, file)
            # Count entries in the registry
            try:
                with open(registry_path, "r") as f:
                    entry_count = sum(
                        1 for line in f if line.strip() and not line.startswith("#")
                    )
                registry_files.append((file, entry_count))
            except Exception:
                registry_files.append((file, -1))

    if not registry_files:
        print("No registry files found")
        return

    # Sort by filename
    registry_files.sort()

    print(f"\nFound {len(registry_files)} registry files:")
    for filename, count in registry_files:
        if count >= 0:
            print(f"  - {filename}: {count:,} entries")
        else:
            print(f"  - {filename}: (error reading file)")

    # Check for master registry
    master_registry = os.path.join(base_dir, "registry.txt")
    if os.path.exists(master_registry):
        print("\nMaster registry found: registry.txt")


def process_grid_checksum(args):
    """Process a single grid directory to generate SHA256 checksums."""
    year_dir, grid_name = args
    grid_dir = os.path.join(year_dir, grid_name)
    sha256_file = os.path.join(grid_dir, "SHA256")

    # Find all .npy files in this grid directory
    npy_files = [f for f in os.listdir(grid_dir) if f.endswith(".npy")]

    if npy_files:
        try:
            # Change to grid directory and run sha256sum
            result = subprocess.run(
                ["sha256sum"] + sorted(npy_files),
                cwd=grid_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            # Write output to SHA256 file
            with open(sha256_file, "w") as f:
                f.write(result.stdout)

            return (grid_name, len(npy_files), True, None)
        except subprocess.CalledProcessError as e:
            return (grid_name, len(npy_files), False, f"CalledProcessError: {e}")
        except Exception as e:
            return (grid_name, len(npy_files), False, f"Exception: {e}")

    return (grid_name, 0, True, None)


def generate_embeddings_checksums(base_dir):
    """Generate SHA256 checksums for .npy files in each embeddings subdirectory."""
    from tqdm import tqdm

    print("Generating SHA256 checksums for embeddings...")

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")

    # Process each year directory
    year_dirs = []
    for item in os.listdir(base_dir):
        if item.isdigit() and len(item) == 4:  # Year directories
            year_path = os.path.join(base_dir, item)
            if os.path.isdir(year_path):
                year_dirs.append(item)

    if not year_dirs:
        print("No year directories found")
        return 1

    total_grids = 0
    processed_grids = 0
    errors = []

    for year in sorted(year_dirs):
        year_dir = os.path.join(base_dir, year)
        print(f"\nProcessing year: {year}")

        # Find all grid directories
        grid_dirs = []
        for item in os.listdir(year_dir):
            if item.startswith("grid_"):
                grid_path = os.path.join(year_dir, item)
                if os.path.isdir(grid_path):
                    grid_dirs.append(item)

        total_grids += len(grid_dirs)

        # Prepare arguments for parallel processing
        grid_args = [(year_dir, grid_name) for grid_name in sorted(grid_dirs)]

        # Process grid directories in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_grid_checksum, args): args for args in grid_args
            }

            # Process results with progress bar
            with tqdm(total=len(grid_dirs), desc=f"Year {year}", unit="grids") as pbar:
                for future in as_completed(futures):
                    grid_name, num_files, success, error_msg = future.result()

                    if success:
                        if num_files > 0:
                            processed_grids += 1
                        pbar.set_postfix(files=num_files)
                    else:
                        errors.append(f"{grid_name}: {error_msg}")

                    pbar.update(1)

    # Report any errors
    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    print(f"\nProcessed {processed_grids}/{total_grids} grid directories")
    return 0 if processed_grids > 0 else 1


def process_tiff_chunk(args):
    """Process a chunk of TIFF files to generate SHA256 checksums."""
    base_dir, chunk, chunk_num = args
    temp_file = os.path.join(base_dir, f".SHA256SUM.tmp{chunk_num}")

    try:
        # Run sha256sum on this chunk
        result = subprocess.run(
            ["sha256sum"] + chunk,
            cwd=base_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Write to temporary file
        with open(temp_file, "w") as f:
            f.write(result.stdout)

        return (chunk_num, len(chunk), True, None, temp_file)
    except subprocess.CalledProcessError as e:
        return (chunk_num, len(chunk), False, f"CalledProcessError: {e}", temp_file)
    except Exception as e:
        return (chunk_num, len(chunk), False, f"Exception: {e}", temp_file)


def generate_tiff_checksums(base_dir):
    """Generate SHA256 checksums for TIFF files using chunked parallel processing."""
    from tqdm import tqdm

    print("Generating SHA256 checksums for TIFF files...")

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")

    # Find all .tiff files
    tiff_files = []
    for file in os.listdir(base_dir):
        if file.endswith(".tiff") or file.endswith(".tif"):
            tiff_files.append(file)

    if not tiff_files:
        print("No TIFF files found")
        return 1

    # Sort files for consistent ordering
    tiff_files.sort()
    total_files = len(tiff_files)
    print(f"Found {total_files} TIFF files")

    # Process in chunks to avoid command line length limits
    chunk_size = 1000  # Process 1000 files at a time
    sha256sum_file = os.path.join(base_dir, "SHA256SUM")

    # Prepare chunks for parallel processing
    chunks = []
    for i in range(0, total_files, chunk_size):
        chunk = tiff_files[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        chunks.append((base_dir, chunk, chunk_num))

    temp_files = []
    errors = []

    try:
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_tiff_chunk, args): args for args in chunks
            }

            # Process results with progress bar
            with tqdm(
                total=total_files, desc="Computing checksums", unit="files"
            ) as pbar:
                results = []
                for future in as_completed(futures):
                    chunk_num, chunk_len, success, error_msg, temp_file = (
                        future.result()
                    )

                    if success:
                        results.append((chunk_num, temp_file))
                    else:
                        errors.append(f"Chunk {chunk_num}: {error_msg}")

                    pbar.update(chunk_len)

                # Sort results by chunk number to maintain order
                results.sort(key=lambda x: x[0])
                temp_files = [temp_file for _, temp_file in results]

        if errors:
            print("\nErrors encountered during processing:")
            for error in errors:
                print(f"  - {error}")
            # Clean up any temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return 1

        # Concatenate all temporary files into final SHA256SUM
        print("Concatenating results...")
        with open(sha256sum_file, "w") as outfile:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, "r") as infile:
                        outfile.write(infile.read())

        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"Successfully generated checksums for {total_files} files")
        print(f"Checksums written to: {sha256sum_file}")
        return 0

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Clean up any temporary files
        for _, _, _, _, temp_file in chunks:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return 1


def scan_embeddings_from_checksums(base_dir, registry_dir, console):
    """Scan SHA256 files in embeddings directory and generate pooch-compatible registries."""
    console.print(Panel.fit("📡 Scanning Embeddings", style="cyan"))

    # Process each year directory
    year_dirs = []
    for item in os.listdir(base_dir):
        if item.isdigit() and len(item) == 4:  # Year directories
            year_path = os.path.join(base_dir, item)
            if os.path.isdir(year_path):
                year_dirs.append(item)

    if not year_dirs:
        console.print("[red]No year directories found[/red]")
        return False

    files_by_year_and_block = defaultdict(lambda: defaultdict(list))
    total_entries = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        for year in sorted(year_dirs):
            year_dir = os.path.join(base_dir, year)
            console.print(f"[blue]Processing year:[/blue] {year}")

            # Find all grid directories with SHA256 files
            grid_dirs = []
            for item in os.listdir(year_dir):
                if item.startswith("grid_"):
                    grid_path = os.path.join(year_dir, item)
                    sha256_file = os.path.join(grid_path, "SHA256")
                    if os.path.isdir(grid_path) and os.path.exists(sha256_file):
                        grid_dirs.append(item)

            console.print(
                f"  Found [green]{len(grid_dirs)}[/green] grid directories with SHA256 files"
            )

            # Process each grid directory
            task = progress.add_task(f"Year {year}", total=len(grid_dirs))

            for grid_name in grid_dirs:
                grid_path = os.path.join(year_dir, grid_name)
                sha256_file = os.path.join(grid_path, "SHA256")

                # Parse coordinates from grid name
                lon, lat = parse_grid_coordinates(grid_name)
                if lon is None or lat is None:
                    console.print(
                        f"  [yellow]Warning:[/yellow] Could not parse coordinates from {grid_name}"
                    )
                    progress.advance(task)
                    continue

                # Get block coordinates
                block_lon, block_lat = get_block_coordinates(lon, lat)
                block_key = (block_lon, block_lat)

                # Read SHA256 file and add entries
                try:
                    with open(sha256_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    checksum = parts[0]
                                    filename = parts[
                                        -1
                                    ]  # Take the last part as filename
                                    # Strip whitespace from all components
                                    checksum = checksum.strip()
                                    filename = filename.strip()
                                    year_clean = year.strip()
                                    grid_name_clean = grid_name.strip()
                                    # Convert to relative path from base_dir
                                    # Use forward slashes and ensure no trailing slashes or spaces
                                    rel_path = (
                                        f"{year_clean}/{grid_name_clean}/{filename}"
                                    )
                                    files_by_year_and_block[year][block_key].append(
                                        (rel_path, checksum)
                                    )
                                    total_entries += 1
                except Exception as e:
                    console.print(f"  [red]Error reading {sha256_file}:[/red] {e}")

                progress.advance(task)

    console.print(f"[green]Total entries found:[/green] {total_entries:,}")

    # Generate block-based registry files
    all_registry_files = []

    for year in sorted(files_by_year_and_block.keys()):
        blocks_for_year = files_by_year_and_block[year]
        console.print(
            f"[blue]Generating registries for year {year}:[/blue] {len(blocks_for_year)} blocks"
        )

        # Create embeddings subdirectory
        embeddings_dir = os.path.join(registry_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)

        for (block_lon, block_lat), block_entries in sorted(blocks_for_year.items()):
            registry_filename = get_embeddings_registry_filename(
                year, block_lon, block_lat
            )
            registry_file = os.path.join(embeddings_dir, registry_filename)
            all_registry_files.append(registry_file)

            console.print(
                f"  Block ({block_lon}, {block_lat}): {len(block_entries)} files → embeddings/{registry_filename}"
            )

            # Write registry file
            with open(registry_file, "w") as f:
                for rel_path, checksum in sorted(block_entries):
                    f.write(f"{rel_path} {checksum}\n")

    # Summary
    if all_registry_files:
        console.print(
            f"[green]✓ Created {len(all_registry_files)} registry files[/green]"
        )

    return len(all_registry_files) > 0


def scan_tiffs_from_checksums(base_dir, registry_dir, console):
    """Scan SHA256SUM file in TIFF directory and generate block-based pooch-compatible registry."""
    console.print(Panel.fit("🗺️  Scanning TIFF Files", style="cyan"))

    sha256sum_file = os.path.join(base_dir, "SHA256SUM")
    if not os.path.exists(sha256sum_file):
        console.print(f"[red]SHA256SUM file not found:[/red] {sha256sum_file}")
        return False

    # Read all TIFF entries from SHA256SUM
    tiff_blocks = defaultdict(list)
    total_entries = 0

    console.print("[blue]Reading SHA256SUM file...[/blue]")
    try:
        with open(sha256sum_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        checksum = parts[0]
                        filename = parts[-1]  # Take the last part as filename
                        if filename.endswith(".tiff") or filename.endswith(".tif"):
                            # Extract coordinates from filename (e.g., grid_-0.35_55.45.tiff)
                            if filename.startswith("grid_"):
                                try:
                                    # Remove 'grid_' prefix and '.tiff' suffix
                                    coords_str = (
                                        filename[5:]
                                        .replace(".tiff", "")
                                        .replace(".tif", "")
                                    )
                                    lon_str, lat_str = coords_str.split("_")
                                    lon = float(lon_str)
                                    lat = float(lat_str)

                                    # Determine block coordinates (5x5 degree blocks)
                                    block_lon, block_lat = get_block_coordinates(
                                        lon, lat
                                    )

                                    # Add to appropriate block
                                    tiff_blocks[(block_lon, block_lat)].append(
                                        (filename, checksum)
                                    )
                                    total_entries += 1
                                except (ValueError, IndexError) as e:
                                    console.print(
                                        f"[yellow]Warning: Could not parse coordinates from {filename}: {e}[/yellow]"
                                    )
                                    continue
    except Exception as e:
        console.print(f"[red]Error reading SHA256SUM file:[/red] {e}")
        return False

    console.print(f"[green]Total TIFF entries found:[/green] {total_entries:,}")
    console.print(f"[green]Total blocks:[/green] {len(tiff_blocks)}")

    if not tiff_blocks:
        console.print("[yellow]No TIFF files found in SHA256SUM[/yellow]")
        return False

    # Create landmasks subdirectory
    landmasks_dir = os.path.join(registry_dir, "landmasks")
    os.makedirs(landmasks_dir, exist_ok=True)

    # Write block-based registry files
    all_registry_files = []
    console.print("\n[blue]Writing block-based landmasks registry files:[/blue]")

    for (block_lon, block_lat), block_entries in sorted(tiff_blocks.items()):
        registry_filename = get_landmasks_registry_filename(block_lon, block_lat)
        registry_file = os.path.join(landmasks_dir, registry_filename)
        all_registry_files.append(registry_file)

        console.print(
            f"  Block ({block_lon}, {block_lat}): {len(block_entries)} files → landmasks/{registry_filename}"
        )

        # Write registry file
        with open(registry_file, "w") as f:
            for rel_path, checksum in sorted(block_entries):
                f.write(f"{rel_path} {checksum}\n")

    if all_registry_files:
        console.print(
            f"[green]✓ Created {len(all_registry_files)} landmasks registry files[/green]"
        )

    console.print("[green]✓ Landmasks registry written[/green]")

    return True


def scan_command(args):
    """Scan SHA256 checksum files and generate pooch-compatible registry files."""
    console = Console()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Error: Directory {base_dir} does not exist[/red]")
        return 1

    console.print(
        Panel.fit(f"🔍 Scanning Registry Data\n📁 {base_dir}", style="bold blue")
    )

    # Determine registry output directory
    if hasattr(args, "registry_dir") and args.registry_dir:
        registry_dir = os.path.join(os.path.abspath(args.registry_dir), "registry")
    else:
        registry_dir = os.path.join(base_dir, "registry")

    # Ensure registry directory exists
    os.makedirs(registry_dir, exist_ok=True)
    console.print(f"[cyan]Registry files will be written to:[/cyan] {registry_dir}")

    # Look for both expected directories
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")

    processed_any = False

    # Process embeddings if directory exists
    if os.path.exists(repr_dir):
        if scan_embeddings_from_checksums(repr_dir, registry_dir, console):
            processed_any = True
    else:
        console.print(f"[yellow]Embeddings directory not found:[/yellow] {repr_dir}")

    # Process TIFF files if directory exists
    if os.path.exists(tiles_dir):
        if scan_tiffs_from_checksums(tiles_dir, registry_dir, console):
            processed_any = True
    else:
        console.print(f"[yellow]TIFF directory not found:[/yellow] {tiles_dir}")

    if not processed_any:
        console.print(
            Panel.fit(
                "[red]No data directories found or no checksum files available.[/red]\n\n"
                f"Expected:\n"
                f"• {repr_dir}\n"
                f"  (with SHA256 files in grid subdirectories)\n"
                f"• {tiles_dir}\n"
                f"  (with SHA256SUM file)\n\n"
                f"[yellow]💡 Run 'geotessera-registry hash' first to generate checksum files.[/yellow]",
                style="red",
            )
        )
        return 1

    # Master registry generation removed - should be done separately

    summary_lines = ["[green]✅ Registry Scan Complete[/green]\n"]
    summary_lines.append("📊 Data processed:")
    if os.path.exists(repr_dir):
        summary_lines.append(f"• Embeddings: {repr_dir}")
        summary_lines.append("  → registry/embeddings/")
    if os.path.exists(tiles_dir):
        summary_lines.append(f"• TIFF files: {tiles_dir}")
        summary_lines.append("  → registry/landmasks/")
    summary_lines.append(f"📁 Registry root: {registry_dir}")

    console.print(Panel.fit("\n".join(summary_lines), style="green"))

    return 0


def hash_command(args):
    """Generate SHA256 checksums for embeddings and TIFF files."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return 1

    # Check if this is an embeddings directory structure
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")

    processed_any = False

    # Process embeddings if directory exists
    if os.path.exists(repr_dir):
        print(f"Processing embeddings directory: {repr_dir}")
        if generate_embeddings_checksums(repr_dir) == 0:
            processed_any = True

    # Process TIFF files if directory exists
    if os.path.exists(tiles_dir):
        print(f"Processing TIFF directory: {tiles_dir}")
        if generate_tiff_checksums(tiles_dir) == 0:
            processed_any = True

    if not processed_any:
        print("No data directories found. Expected:")
        print(f"  - {repr_dir}")
        print(f"  - {tiles_dir}")
        return 1

    return 0


def main():
    """Main entry point for the geotessera-registry CLI tool."""
    parser = argparse.ArgumentParser(
        description="GeoTessera Registry Management Tool - Generate and maintain Pooch registry files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List existing registry files
  geotessera-registry list /path/to/data
  
  # Generate SHA256 checksums for embeddings and TIFF files
  geotessera-registry hash /path/to/v1
  
  # This will:
  # - Create SHA256 files in each grid subdirectory under global_0.1_degree_representation/YYYY/
  # - Create SHA256SUM file in global_0.1_degree_tiff_all/ using chunked processing
  
  # Scan existing SHA256 checksum files and generate pooch-compatible registries
  geotessera-registry scan /path/to/v1
  
  # This will:
  # - Read SHA256 files from grid subdirectories and generate block-based registry files
  # - Read SHA256SUM file from TIFF directory and generate landmask registry files

This tool is intended for GeoTessera data maintainers to generate the registry
files that are distributed with the package. End users typically don't need
to use this tool.

Note: This tool creates block-based registries for efficient lazy loading:
  - Embeddings: Organized into 5x5 degree blocks (embeddings_YYYY_lonX_latY.txt)
  - Landmasks: Organized into 5x5 degree blocks (landmasks_lonX_latY.txt)
  - Each block contains ~2,500 tiles instead of one massive registry
  - Registry files are created in the registry/ subdirectory

Directory Structure:
  The commands expect to find these subdirectories:
  - global_0.1_degree_representation/  (contains .npy files organized by year)
  - global_0.1_degree_tiff_all/        (contains .tiff files in flat structure)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List existing registry files")
    list_parser.add_argument(
        "base_dir", help="Base directory to scan for registry files"
    )
    list_parser.set_defaults(func=list_command)

    # Hash command
    hash_parser = subparsers.add_parser(
        "hash", help="Generate SHA256 checksums for embeddings and TIFF files"
    )
    hash_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation and/or global_0.1_degree_tiff_all subdirectories",
    )
    hash_parser.set_defaults(func=hash_command)

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan existing SHA256 checksum files and generate pooch-compatible registry files",
    )
    scan_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation and/or global_0.1_degree_tiff_all subdirectories with checksum files",
    )
    scan_parser.add_argument(
        "--registry-dir",
        type=str,
        default=None,
        help="Output directory for registry files (default: same as base_dir)",
    )
    scan_parser.set_defaults(func=scan_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
