"""Tests for CLI discrete GeoTIFF export functionality."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from typer.testing import CliRunner
from geotessera.cli import app


class TestCLIDiscreteExport:
    """Test CLI functionality for exporting discrete GeoTIFF files."""

    def test_cli_help(self):
        """Test that CLI help works and shows expected commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "download" in result.stdout
        assert "visualize" in result.stdout
        assert "info" in result.stdout

    def test_download_command_help(self):
        """Test that download command help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout
        assert "--bbox" in result.stdout
        assert "--region-file" in result.stdout
        assert "--year" in result.stdout
        assert "--bands" in result.stdout
        assert "--list-files" in result.stdout

    def test_info_command_help(self):
        """Test that info command help works."""
        runner = CliRunner()
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "--geotiffs" in result.stdout
        assert "--dataset-version" in result.stdout

    def test_visualize_command_help(self):
        """Test that visualize command help works."""
        runner = CliRunner()
        result = runner.invoke(app, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout
        assert "--type" in result.stdout
        assert "--bands" in result.stdout

    @patch('geotessera.cli.GeoTessera')
    def test_info_command_execution(self, mock_geotessera_class):
        """Test that info command executes without error."""
        # Mock GeoTessera instance
        mock_gt_instance = Mock()
        mock_gt_instance.version = "0.4.0"
        mock_gt_instance.get_available_years.return_value = [2022, 2023, 2024]
        mock_gt_instance.registry.loaded_blocks = {}
        mock_geotessera_class.return_value = mock_gt_instance
        
        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "0.4.0" in result.stdout
        assert "2024" in result.stdout

    def test_download_missing_output(self):
        """Test that download command requires output directory."""
        runner = CliRunner()
        result = runner.invoke(app, ["download", "--bbox", "0,51,1,52"])
        
        # Should fail because --output is required
        assert result.exit_code != 0

    def test_download_missing_bbox_and_region(self):
        """Test that download command requires either bbox or region-file."""
        runner = CliRunner()
        result = runner.invoke(app, ["download", "--output", "/tmp/test"])
        
        # Should fail because neither --bbox nor --region-file is provided
        assert result.exit_code == 1  # typer.Exit(1)
        assert "Must specify either --bbox or --region-file" in result.stdout

    @patch('geotessera.cli.GeoTessera')
    def test_download_invalid_bbox_format(self, mock_geotessera_class):
        """Test that download command validates bbox format."""
        runner = CliRunner()
        result = runner.invoke(app, ["download", "--output", "/tmp/test", "--bbox", "invalid"])
        
        # Should fail due to invalid bbox format
        assert result.exit_code == 1
        assert "Invalid bbox format" in result.stdout

    @patch('geotessera.cli.GeoTessera')
    def test_download_invalid_bbox_length(self, mock_geotessera_class):
        """Test that download command validates bbox has 4 values."""
        runner = CliRunner()
        result = runner.invoke(app, ["download", "--output", "/tmp/test", "--bbox", "0,51,1"])
        
        # Should fail due to wrong number of bbox values
        assert result.exit_code == 1
        assert "bbox must be 'min_lon,min_lat,max_lon,max_lat'" in result.stdout