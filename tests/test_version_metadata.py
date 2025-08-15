"""Tests for version export and GeoTIFF metadata inclusion."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import numpy as np

from geotessera import GeoTessera
import geotessera


class TestVersionMetadata:
    """Test version is properly exported and included in GeoTIFF metadata."""

    def test_version_property_available(self):
        """Test that GeoTessera instance has version property."""
        gt = GeoTessera()
        assert hasattr(gt, 'version')
        assert isinstance(gt.version, str)
        assert gt.version != "unknown"  # Should have actual version
        
    def test_version_matches_package_version(self):
        """Test that instance version matches package version."""
        gt = GeoTessera()
        assert gt.version == geotessera.__version__
        
    def test_geotiff_metadata_includes_version(self):
        """Test that exported GeoTIFF includes version in metadata."""
        pytest.importorskip("rasterio")
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        
        gt = GeoTessera()
        
        # Mock the UTM projection method directly to avoid rasterio complexity
        utm_crs = CRS.from_epsg(32630)  # UTM 30N
        utm_transform = Affine(10.0, 0.0, 500000, 0.0, -10.0, 4000000)
        gt._get_utm_projection_from_landmask = Mock(return_value=(utm_crs, utm_transform))
        
        # Create mock embedding data
        mock_embedding = np.random.rand(256, 256, 128).astype(np.float32)
        gt.fetch_embedding = Mock(return_value=mock_embedding)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_tile.tif"
            
            # Export single tile
            created_file = gt.export_embedding_geotiff(
                lat=51.5,
                lon=0.0,
                output_path=str(output_path),
                year=2024,
                bands=[0, 1, 2]
            )
            
            # Verify file was created
            assert Path(created_file).exists()
            
            # Read back the file and check metadata
            import rasterio
            with rasterio.open(created_file) as src:
                tags = src.tags()
                
                # Check that version is in metadata
                assert 'GEOTESSERA_VERSION' in tags
                assert tags['GEOTESSERA_VERSION'] == gt.version
                assert tags['GEOTESSERA_VERSION'] != "unknown"
                
                # Check other expected metadata
                assert 'TESSERA_DATASET_VERSION' in tags
                assert 'TESSERA_YEAR' in tags
                assert 'TESSERA_TILE_LAT' in tags
                assert 'TESSERA_TILE_LON' in tags
                assert 'TESSERA_DESCRIPTION' in tags
                
                assert tags['TESSERA_YEAR'] == '2024'
                assert tags['TESSERA_TILE_LAT'] == '51.50'
                assert tags['TESSERA_TILE_LON'] == '0.00'

    def test_bulk_export_geotiffs_include_version(self):
        """Test that bulk exported GeoTIFFs include version in metadata."""
        pytest.importorskip("rasterio")
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        
        gt = GeoTessera()
        
        # Mock registry components
        gt.registry.get_tiles_in_bbox = Mock(return_value=[(51.55, 0.05)])
        
        # Mock the UTM projection method directly
        utm_crs = CRS.from_epsg(32630)  # UTM 30N
        utm_transform = Affine(10.0, 0.0, 500000, 0.0, -10.0, 4000000)
        gt._get_utm_projection_from_landmask = Mock(return_value=(utm_crs, utm_transform))
        
        # Create mock embedding data
        mock_embedding = np.random.rand(256, 256, 128).astype(np.float32)
        gt.fetch_embedding = Mock(return_value=mock_embedding)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export tiles
            created_files = gt.export_embedding_geotiffs(
                bbox=(0.0, 51.5, 0.1, 51.6),
                output_dir=temp_dir,
                year=2024,
                bands=[0, 1, 2]
            )
            
            # Verify at least one file was created
            assert len(created_files) > 0
            
            # Check metadata in first file
            import rasterio
            with rasterio.open(created_files[0]) as src:
                tags = src.tags()
                
                # Check that version is in metadata
                assert 'GEOTESSERA_VERSION' in tags
                assert tags['GEOTESSERA_VERSION'] == gt.version
                assert tags['GEOTESSERA_VERSION'] != "unknown"