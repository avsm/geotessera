"""Tests for landmask UTM projection information and access."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from geotessera import GeoTessera
from geotessera.registry import tile_to_landmask_filename, world_to_tile_coords


class TestLandmaskUTMProjections:
    """Test landmask tiles and their UTM projection information."""

    @pytest.mark.parametrize("lat,lon,expected_epsg,location", [
        (51.5, -0.1, 32630, "London_UK"),      # UTM 30N
        (40.7, -74.0, 32618, "New_York_USA"),  # UTM 18N 
        (-33.9, 18.4, 32734, "Cape_Town_SA"),  # UTM 34S
        (35.7, 139.7, 32654, "Tokyo_Japan"),   # UTM 54N
        (55.7, 37.6, 32637, "Moscow_Russia"),  # UTM 37N
    ])
    def test_landmask_utm_projections(self, lat, lon, expected_epsg, location):
        """Test that landmask tiles contain correct UTM projections for various locations."""
        # Convert to tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        
        # Get landmask filename
        landmask_filename = tile_to_landmask_filename(tile_lat, tile_lon)
        
        # Initialize GeoTessera
        gt = GeoTessera()
        
        try:
            # Load the appropriate registry block
            gt.registry.ensure_tile_block_loaded(tile_lon, tile_lat)
            
            # Fetch the landmask file
            landmask_path = gt.registry.fetch_landmask(landmask_filename)
            
            # Verify file exists
            assert Path(landmask_path).exists(), f"Landmask file not found: {landmask_path}"
            
            # Read projection information with rasterio
            pytest.importorskip("rasterio")
            import rasterio
            
            with rasterio.open(landmask_path) as src:
                # Verify CRS
                assert src.crs is not None, f"No CRS found in landmask for {location}"
                assert src.crs.to_epsg() == expected_epsg, \
                    f"Expected EPSG:{expected_epsg} for {location}, got {src.crs.to_epsg()}"
                
                # Verify it's a projected (UTM) coordinate system, not geographic
                assert not src.crs.is_geographic, \
                    f"Expected projected CRS for {location}, got geographic: {src.crs}"
                
                # Verify it has transform information
                assert src.transform is not None, f"No transform found for {location}"
                
                # Verify reasonable bounds (UTM coordinates should be in meters)
                bounds = src.bounds
                assert bounds.left > 0, f"Invalid UTM easting for {location}: {bounds.left}"
                assert bounds.bottom > 0, f"Invalid UTM northing for {location}: {bounds.bottom}"
                
                # Verify data properties
                assert src.count == 1, f"Expected 1 band, got {src.count} for {location}"
                assert src.dtypes[0] == 'uint8', f"Expected uint8 data type for {location}"
                assert src.width > 0 and src.height > 0, f"Invalid dimensions for {location}"
                
        except Exception as e:
            pytest.fail(f"Failed to access landmask for {location}: {e}")

    def test_landmask_coordinate_alignment(self):
        """Test that landmask and embedding tiles use the same coordinate system."""
        lat, lon = 51.5, -0.1
        
        # Convert to tile coordinates (should be same for both)
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        
        # Get filenames
        landmask_filename = tile_to_landmask_filename(tile_lat, tile_lon)
        
        # Verify naming consistency
        expected_pattern = f"grid_{tile_lon:.2f}_{tile_lat:.2f}.tiff"
        assert landmask_filename == expected_pattern, \
            f"Landmask filename doesn't match expected pattern: {landmask_filename} vs {expected_pattern}"

    @pytest.mark.parametrize("lat,lon", [
        (51.5, -0.1),   # London
        (40.7, -74.0),  # New York
        (-33.9, 18.4),  # Cape Town
    ])
    def test_landmask_registry_access(self, lat, lon):
        """Test that landmask registry access works correctly."""
        gt = GeoTessera()
        
        # Convert to tile coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        
        # Get landmask filename
        landmask_filename = tile_to_landmask_filename(tile_lat, tile_lon)
        
        # Test registry block loading
        gt.registry.ensure_tile_block_loaded(tile_lon, tile_lat)
        
        # Test fetch method exists and works
        landmask_path = gt.registry.fetch_landmask(landmask_filename)
        assert isinstance(landmask_path, str), "fetch_landmask should return string path"
        assert Path(landmask_path).exists(), f"Fetched landmask file should exist: {landmask_path}"

    def test_utm_projection_extraction_function(self):
        """Test helper function to extract UTM projection from landmask."""
        lat, lon = 51.5, -0.1  # London - should be UTM 30N (EPSG:32630)
        
        gt = GeoTessera()
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        landmask_filename = tile_to_landmask_filename(tile_lat, tile_lon)
        
        gt.registry.ensure_tile_block_loaded(tile_lon, tile_lat)
        landmask_path = gt.registry.fetch_landmask(landmask_filename)
        
        pytest.importorskip("rasterio")
        import rasterio
        
        # Function that could be used in core.py
        def get_landmask_projection(landmask_path: str):
            """Extract CRS and transform from landmask tile."""
            with rasterio.open(landmask_path) as src:
                return src.crs, src.transform
        
        crs, transform = get_landmask_projection(landmask_path)
        
        # Verify we got valid projection info
        assert crs is not None, "Should have valid CRS"
        assert crs.to_epsg() == 32630, f"Expected UTM 30N, got {crs}"
        assert transform is not None, "Should have valid transform"
        assert not crs.is_geographic, "Should be projected coordinate system"


class TestLandmaskProjectionIntegration:
    """Integration tests for using landmask projections in GeoTIFF export."""
    
    def test_landmask_projection_retrieval_workflow(self):
        """Test the complete workflow for retrieving projection from landmask."""
        lat, lon = 40.7, -74.0  # New York
        year = 2024
        
        gt = GeoTessera()
        
        # Step 1: Convert coordinates
        tile_lat, tile_lon = world_to_tile_coords(lat, lon)
        
        # Step 2: Get landmask filename
        landmask_filename = tile_to_landmask_filename(tile_lat, tile_lon)
        
        # Step 3: Load registry and fetch landmask
        gt.registry.ensure_tile_block_loaded(tile_lon, tile_lat)
        landmask_path = gt.registry.fetch_landmask(landmask_filename)
        
        # Step 4: Extract projection info
        pytest.importorskip("rasterio")
        import rasterio
        
        with rasterio.open(landmask_path) as src:
            utm_crs = src.crs
            utm_transform = src.transform
            utm_bounds = src.bounds
            
        # Verify we have what we need for export
        assert utm_crs is not None, "Need valid UTM CRS"
        assert utm_crs.to_epsg() == 32618, "Should be UTM 18N for New York"
        assert utm_transform is not None, "Need valid transform"
        assert utm_bounds is not None, "Need valid bounds"
        
        # This CRS and transform should be usable for embedding export
        assert not utm_crs.is_geographic, "Should use projected coordinates"
        assert utm_transform.a > 0, "Should have positive x pixel size"  # 10m pixels expected
        assert utm_transform.e < 0, "Should have negative y pixel size"  # Standard UTM orientation


if __name__ == '__main__':
    pytest.main([__file__])