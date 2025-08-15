"""Tests for CLI discrete GeoTIFF export functionality."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import sys
from io import StringIO

from geotessera.cli import main, download_command
from argparse import Namespace


class TestCLIDiscreteExport:
    """Test CLI functionality for exporting discrete GeoTIFF files."""

    def test_download_command_help_updated(self):
        """Test that download command help shows updated description."""
        old_argv = sys.argv
        old_stdout = sys.stdout
        
        try:
            sys.argv = ['geotessera', 'download', '--help']
            sys.stdout = mystdout = StringIO()
            
            with pytest.raises(SystemExit):
                main()
            
            help_output = mystdout.getvalue()
            assert '--list-files' in help_output, "Should have --list-files option"
            assert 'discrete' not in help_output.lower(), "Help text should not mention discrete (it's in the description)"
            
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

    @patch('geotessera.cli.GeoTessera')
    def test_download_command_enhanced_output(self, mock_geotessera_class):
        """Test that the enhanced download command provides detailed output."""
        # Mock GeoTessera instance
        mock_gt_instance = Mock()
        mock_files = [
            "/tmp/tessera_2024_lat51.55_lon-0.05.tif",
            "/tmp/tessera_2024_lat51.65_lon-0.05.tif",
        ]
        mock_gt_instance.export_embedding_geotiffs.return_value = mock_files
        mock_geotessera_class.return_value = mock_gt_instance
        
        # Mock file stats
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB
            with patch('pathlib.Path.exists', return_value=True):
                
                # Create mock args
                args = Namespace(
                    dataset_version='v1',
                    cache_dir=None,
                    registry_dir=None,
                    bbox='-0.1,51.5,0.0,51.6',
                    region_file=None,
                    year=2024,
                    bands=None,
                    output='/tmp/test',
                    compress='lzw',
                    verbose=False,
                    list_files=False
                )
                
                # Capture output
                old_stdout = sys.stdout
                try:
                    sys.stdout = mystdout = StringIO()
                    download_command(args)
                    output = mystdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                # Verify enhanced output messages
                assert "Region of Interest Export" in output
                assert "discrete GeoTIFF files" in output
                assert "UTM projection" in output
                assert "Next steps:" in output
                assert "individually inspected" in output or "individual tile" in output

    @patch('geotessera.cli.GeoTessera')
    def test_download_command_list_files_option(self, mock_geotessera_class):
        """Test that --list-files option shows detailed file information."""
        # Mock GeoTessera instance
        mock_gt_instance = Mock()
        mock_files = [
            "/tmp/tessera_2024_lat51.55_lon-0.05.tif",
            "/tmp/tessera_2024_lat51.65_lon-0.05.tif",
        ]
        mock_gt_instance.export_embedding_geotiffs.return_value = mock_files
        mock_geotessera_class.return_value = mock_gt_instance
        
        # Mock file stats  
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 2048 * 1024  # 2MB
            with patch('pathlib.Path.exists', return_value=True):
                
                # Create mock args with list_files=True
                args = Namespace(
                    dataset_version='v1',
                    cache_dir=None,
                    registry_dir=None,
                    bbox='-0.1,51.5,0.0,51.6',
                    region_file=None,
                    year=2024,
                    bands='0,1,2',
                    output='/tmp/test',
                    compress='lzw',
                    verbose=False,
                    list_files=True
                )
                
                # Capture output
                old_stdout = sys.stdout
                try:
                    sys.stdout = mystdout = StringIO()
                    download_command(args)
                    output = mystdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                # Verify file listing
                assert "Created files:" in output
                assert "tessera_2024_lat51.55_lon-0.05.tif" in output
                assert "2,097,152 bytes" in output  # 2MB formatted
                assert "1." in output and "2." in output  # File numbering

    @patch('geotessera.cli.GeoTessera')
    def test_download_command_verbose_projection_info(self, mock_geotessera_class):
        """Test that verbose mode shows projection information."""
        # Mock GeoTessera instance
        mock_gt_instance = Mock()
        mock_files = ["/tmp/tessera_2024_lat51.55_lon-0.05.tif"]
        mock_gt_instance.export_embedding_geotiffs.return_value = mock_files
        mock_geotessera_class.return_value = mock_gt_instance
        
        # Mock rasterio file reading
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024 * 1024
            with patch('pathlib.Path.exists', return_value=True):
                with patch('rasterio.open') as mock_rasterio_open:
                    mock_src = Mock()
                    mock_src.crs = "EPSG:32630"
                    mock_src.transform = "Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)"
                    mock_src.width = 738
                    mock_src.height = 1140  
                    mock_src.dtypes = ['float32']
                    mock_src.__enter__ = Mock(return_value=mock_src)
                    mock_src.__exit__ = Mock(return_value=None)
                    mock_rasterio_open.return_value = mock_src
                    
                    # Create mock args with verbose=True
                    args = Namespace(
                        dataset_version='v1',
                        cache_dir=None,
                        registry_dir=None,
                        bbox='-0.1,51.5,0.0,51.6',
                        region_file=None,
                        year=2024,
                        bands=None,
                        output='/tmp/test',
                        compress='lzw',
                        verbose=True,
                        list_files=False
                    )
                    
                    # Capture output
                    old_stdout = sys.stdout
                    try:
                        sys.stdout = mystdout = StringIO()
                        download_command(args)
                        output = mystdout.getvalue()
                    finally:
                        sys.stdout = old_stdout
                    
                    # Verify projection info is shown
                    assert "CRS:" in output
                    assert "Transform:" in output
                    assert "Dimensions:" in output
                    assert "738 x 1140" in output

    @patch('geotessera.cli.calculate_bbox_from_file')
    @patch('geotessera.cli.GeoTessera')
    def test_download_command_region_file_support(self, mock_geotessera_class, mock_calc_bbox):
        """Test that region file input shows detailed bbox information."""
        # Mock bbox calculation from file
        mock_calc_bbox.return_value = (-0.2, 51.4, 0.1, 51.6)
        
        # Mock GeoTessera instance
        mock_gt_instance = Mock()
        mock_gt_instance.export_embedding_geotiffs.return_value = ["/tmp/test.tif"]
        mock_geotessera_class.return_value = mock_gt_instance
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024
            with patch('pathlib.Path.exists', return_value=True):
                
                # Create mock args with region file
                args = Namespace(
                    dataset_version='v1',
                    cache_dir=None,
                    registry_dir=None,
                    bbox=None,
                    region_file='london.geojson',
                    year=2024,
                    bands=None,
                    output='/tmp/test',
                    compress='lzw',
                    verbose=False,
                    list_files=False
                )
                
                # Capture output
                old_stdout = sys.stdout
                try:
                    sys.stdout = mystdout = StringIO()
                    download_command(args)
                    output = mystdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                # Verify detailed bbox info from region file
                assert "Calculated bbox from london.geojson" in output
                assert "Longitude range:" in output
                assert "Latitude range:" in output
                assert "51.400000 to 51.600000" in output

    def test_download_command_error_handling(self):
        """Test that download command handles missing arguments gracefully."""
        # Test missing bbox and region-file
        args = Namespace(
            dataset_version='v1',
            cache_dir=None,
            registry_dir=None,
            bbox=None,
            region_file=None,
            year=2024,
            bands=None,
            output='/tmp/test',
            compress='lzw',
            verbose=False,
            list_files=False
        )
        
        # Capture output
        old_stdout = sys.stdout
        try:
            sys.stdout = mystdout = StringIO()
            download_command(args)
            output = mystdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Verify helpful error messages
        assert "Must specify either --bbox or --region-file" in output
        assert "Examples:" in output
        assert "--bbox '-0.2,51.4,0.1,51.6'" in output
        assert "--region-file london.geojson" in output


if __name__ == '__main__':
    pytest.main([__file__])