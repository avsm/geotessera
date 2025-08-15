"""Tests for dataset_version parameter and version disambiguation."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

from geotessera import GeoTessera
from geotessera.core import GeoTessera as CoreGeoTessera


class TestDatasetVersionParameter:
    """Test dataset_version parameter functionality."""
    
    def test_default_dataset_version(self):
        """Test that default dataset_version is 'v1'."""
        with patch('geotessera.core.Registry') as mock_registry:
            gt = GeoTessera()
            assert gt.dataset_version == "v1"
            mock_registry.assert_called_once()
            # Check that Registry was called with version='v1'
            call_args = mock_registry.call_args
            assert call_args[1]['version'] == 'v1'
    
    def test_custom_dataset_version(self):
        """Test initialization with custom dataset_version."""
        with patch('geotessera.core.Registry') as mock_registry:
            gt = GeoTessera(dataset_version="v2")
            assert gt.dataset_version == "v2"
            # Check that Registry was called with version='v2'
            call_args = mock_registry.call_args
            assert call_args[1]['version'] == 'v2'
    
    def test_dataset_version_passed_to_registry(self):
        """Test that dataset_version is properly passed to Registry."""
        with patch('geotessera.core.Registry') as mock_registry:
            test_version = "v3"
            gt = GeoTessera(
                dataset_version=test_version,
                cache_dir="/tmp/test",
                auto_update=False
            )
            
            # Verify Registry was called with correct parameters
            mock_registry.assert_called_once_with(
                version=test_version,
                cache_dir="/tmp/test",
                registry_dir=None,
                auto_update=False,
                manifests_repo_url="https://github.com/ucam-eo/tessera-manifests.git"
            )
    
    def test_get_available_years_uses_registry(self):
        """Test that get_available_years delegates to registry."""
        mock_years = [2020, 2021, 2022]
        with patch('geotessera.core.Registry') as mock_registry_class:
            mock_registry_instance = Mock()
            mock_registry_instance.get_available_years.return_value = mock_years
            mock_registry_class.return_value = mock_registry_instance
            
            gt = GeoTessera(dataset_version="v1")
            years = gt.get_available_years()
            
            assert years == mock_years
            mock_registry_instance.get_available_years.assert_called_once()


class TestGeoTIFFMetadataTags:
    """Test that exported GeoTIFFs have correct metadata tags."""
    
    @patch('geotessera.core.Registry')
    @patch('rasterio.open')
    def test_geotiff_metadata_tags_separation(self, mock_rasterio_open, mock_registry_class):
        """Test that TESSERA_DATASET_VERSION and GEOTESSERA_VERSION are separate."""
        # Setup mocks
        mock_registry_instance = Mock()
        mock_registry_class.return_value = mock_registry_instance
        mock_registry_instance.load_blocks_for_region.return_value = None
        mock_registry_instance.available_embeddings = [(2024, 51.55, -0.05)]
        mock_registry_instance.ensure_block_loaded.return_value = None
        mock_registry_instance.fetch.return_value = "/fake/path"
        
        # Mock the rasterio context manager
        mock_dst = Mock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dst
        
        # Mock numpy load for embedding data
        import numpy as np
        fake_embedding = np.random.rand(100, 100, 128).astype(np.float32)
        fake_scales = np.ones((100, 100, 128), dtype=np.float32)
        
        with patch('numpy.load') as mock_np_load:
            mock_np_load.side_effect = [fake_embedding, fake_scales]
            
            with patch('geotessera.registry.tile_to_embedding_path') as mock_path:
                mock_path.return_value = ("/fake/embedding.npy", "/fake/scales.npy")
                
                with patch('geotessera.registry.get_tile_bounds') as mock_bounds:
                    mock_bounds.return_value = (-0.1, 51.5, 0.0, 51.6)
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        gt = GeoTessera(dataset_version="v2")
                        
                        # Export tiles
                        files = gt.export_tiles_to_geotiffs(
                            bbox=(-0.1, 51.5, 0.0, 51.6),
                            output_dir=temp_dir,
                            year=2024
                        )
                        
                        # Check that update_tags was called with correct metadata
                        mock_dst.update_tags.assert_called_once()
                        call_args = mock_dst.update_tags.call_args[1]
                        
                        # Verify dataset version tag
                        assert 'TESSERA_DATASET_VERSION' in call_args
                        assert call_args['TESSERA_DATASET_VERSION'] == 'v2'
                        
                        # Verify library version tag is separate
                        assert 'GEOTESSERA_VERSION' in call_args
                        assert 'geotessera' in call_args['GEOTESSERA_VERSION'].lower()
                        
                        # Verify other required tags
                        assert call_args['TESSERA_YEAR'] == '2024'
                        assert call_args['TESSERA_TILE_LAT'] == '51.55'
                        assert call_args['TESSERA_TILE_LON'] == '-0.05'


class TestCLIDatasetVersion:
    """Test CLI dataset-version argument."""
    
    def test_cli_dataset_version_argument_exists(self):
        """Test that CLI accepts --dataset-version argument."""
        from geotessera.cli import main
        import sys
        from io import StringIO
        
        # Capture help output
        old_stdout = sys.stdout
        old_argv = sys.argv
        
        try:
            sys.stdout = mystdout = StringIO()
            sys.argv = ['geotessera', '--help']
            
            with pytest.raises(SystemExit):
                main()
            
            help_output = mystdout.getvalue()
            assert '--dataset-version' in help_output
            assert 'Tessera dataset version' in help_output
            
        finally:
            sys.stdout = old_stdout
            sys.argv = old_argv
    
    @patch('geotessera.cli.GeoTessera')
    def test_cli_passes_dataset_version_to_geotessera(self, mock_geotessera_class):
        """Test that CLI properly passes dataset-version to GeoTessera."""
        from geotessera.cli import download_command
        from argparse import Namespace
        
        # Mock GeoTessera instance
        mock_gt_instance = Mock()
        mock_gt_instance.export_tiles_to_geotiffs.return_value = ["test.tif"]
        mock_geotessera_class.return_value = mock_gt_instance
        
        # Create mock args
        args = Namespace(
            dataset_version='v2',
            cache_dir=None,
            registry_dir=None,
            bbox='0,51,1,52',
            region_file=None,
            year=2024,
            bands=None,
            output='/tmp/test',
            compress='lzw',
            verbose=False
        )
        
        # Call download command
        download_command(args)
        
        # Verify GeoTessera was initialized with correct dataset_version
        mock_geotessera_class.assert_called_once_with(
            dataset_version='v2',
            cache_dir=None,
            registry_dir=None
        )


if __name__ == '__main__':
    pytest.main([__file__])