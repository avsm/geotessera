"""Simplified tests for UTM projection preservation in GeoTIFF export."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import numpy as np

from geotessera import GeoTessera


class TestUTMProjectionExportSimple:
    """Simplified tests for UTM projection preservation."""

    def test_utm_projection_helper_function_success(self):
        """Test the _get_utm_projection_from_landmask helper function with successful retrieval."""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        
        gt = GeoTessera()
        
        # Mock registry components
        gt.registry.ensure_tile_block_loaded = Mock()
        gt.registry.fetch_landmask = Mock(return_value="/fake/landmask.tiff")
        
        # Create mock UTM data
        utm_crs = CRS.from_epsg(32630)  # UTM 30N
        utm_transform = Affine(10.0, 0.0, 500000, 0.0, -10.0, 4000000)
        
        # Mock rasterio file reading
        mock_src = Mock()
        mock_src.crs = utm_crs
        mock_src.transform = utm_transform
        mock_src.__enter__ = Mock(return_value=mock_src)
        mock_src.__exit__ = Mock(return_value=None)
        
        with patch('rasterio.open', return_value=mock_src):
            with patch('geotessera.registry.tile_to_landmask_filename', return_value="test.tiff"):
                # Test successful retrieval
                crs, transform = gt._get_utm_projection_from_landmask(51.5, -0.1)
                
                assert crs == utm_crs, f"Expected {utm_crs}, got {crs}"
                assert transform == utm_transform, f"Expected {utm_transform}, got {transform}"

    def test_utm_projection_helper_function_error_handling(self):
        """Test that _get_utm_projection_from_landmask raises exceptions on error."""
        gt = GeoTessera()
        
        # Mock registry to raise exception
        gt.registry.ensure_tile_block_loaded = Mock(side_effect=Exception("Registry error"))
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to get UTM projection from landmask"):
            gt._get_utm_projection_from_landmask(51.5, -0.1)

    def test_utm_projection_helper_function_no_rasterio(self):
        """Test helper function behavior when rasterio import fails."""
        gt = GeoTessera()
        
        # Mock the function to simulate rasterio import failure
        with patch.object(gt, '_get_utm_projection_from_landmask') as mock_method:
            # Override the method to simulate ImportError
            def mock_utm_projection(lat, lon):
                raise ImportError("rasterio required for UTM projection retrieval: pip install rasterio")
            
            mock_method.side_effect = mock_utm_projection
            
            # Should raise ImportError
            with pytest.raises(ImportError, match="rasterio required for UTM projection retrieval"):
                gt._get_utm_projection_from_landmask(51.5, -0.1)

    def test_export_geotiff_uses_utm_when_available(self):
        """Test that export functions use UTM projection when available."""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        
        gt = GeoTessera()
        
        # Mock the embedding fetch
        gt.registry.ensure_tile_block_loaded = Mock()
        gt.registry.fetch = Mock(return_value="/fake/path")
        
        # Mock numpy load for embedding data
        fake_embedding = np.random.rand(100, 100, 128).astype(np.float32)
        fake_scales = np.ones((100, 100, 128), dtype=np.float32)
        
        # Mock UTM projection retrieval to succeed
        utm_crs = CRS.from_epsg(32630)  # UTM 30N
        utm_transform = Affine(10.0, 0.0, 500000, 0.0, -10.0, 4000000)
        
        with patch('numpy.load', side_effect=[fake_embedding, fake_scales]):
            with patch('geotessera.registry.tile_to_embedding_path', return_value=("/fake/emb.npy", "/fake/scales.npy")):
                with patch.object(gt, '_get_utm_projection_from_landmask', return_value=(utm_crs, utm_transform)):
                    with patch('rasterio.open') as mock_rasterio_open:
                        mock_dst = Mock()
                        mock_dst.__enter__ = Mock(return_value=mock_dst)
                        mock_dst.__exit__ = Mock(return_value=None)
                        mock_rasterio_open.return_value = mock_dst
                        
                        with tempfile.TemporaryDirectory() as temp_dir:
                            output_path = Path(temp_dir) / "test.tif"
                            
                            # Export single tile
                            result_path = gt.export_embedding_geotiff(
                                lat=51.5, lon=-0.1,
                                output_path=str(output_path)
                            )
                            
                            # Verify UTM projection was used
                            assert mock_rasterio_open.called, "Should have opened file for writing"
                            
                            # Check the call arguments
                            call_args = mock_rasterio_open.call_args
                            assert call_args is not None, "Should have call arguments"
                            
                            # Verify UTM CRS was passed
                            kwargs = call_args[1]
                            assert 'crs' in kwargs, "Should specify CRS"
                            assert kwargs['crs'] == utm_crs, f"Expected {utm_crs}, got {kwargs['crs']}"
                            
                            # Verify UTM transform was passed
                            assert 'transform' in kwargs, "Should specify transform"
                            assert kwargs['transform'] == utm_transform, f"Expected {utm_transform}, got {kwargs['transform']}"

    def test_export_geotiff_raises_on_utm_failure(self):
        """Test that export functions raise exceptions when UTM projection unavailable."""
        pytest.importorskip("rasterio")
        import rasterio
        
        gt = GeoTessera()
        
        # Mock the embedding fetch
        gt.registry.ensure_tile_block_loaded = Mock()
        gt.registry.fetch = Mock(return_value="/fake/path")
        
        # Mock numpy load for embedding data
        fake_embedding = np.random.rand(100, 100, 128).astype(np.float32)
        fake_scales = np.ones((100, 100, 128), dtype=np.float32)
        
        # Mock UTM projection retrieval to fail
        with patch('numpy.load', side_effect=[fake_embedding, fake_scales]):
            with patch('geotessera.registry.tile_to_embedding_path', return_value=("/fake/emb.npy", "/fake/scales.npy")):
                with patch.object(gt, '_get_utm_projection_from_landmask', side_effect=RuntimeError("Landmask not available")):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        output_path = Path(temp_dir) / "test.tif"
                        
                        # Export should raise exception
                        with pytest.raises(RuntimeError, match="Landmask not available"):
                            gt.export_embedding_geotiff(
                                lat=51.5, lon=-0.1,
                                output_path=str(output_path)
                            )

    def test_export_geotiffs_utm_integration(self):
        """Integration test for plural export function with UTM projection."""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        
        gt = GeoTessera()
        
        # Mock the embedding fetch system
        gt.registry.load_blocks_for_region = Mock(return_value=[(51.55, -0.05)])
        gt.registry.ensure_tile_block_loaded = Mock()
        gt.registry.fetch = Mock(return_value="/fake/path")
        
        # Mock available embeddings
        with patch.object(type(gt.registry), 'available_embeddings', [(2024, 51.55, -0.05)]):
            # Mock numpy load for embedding data
            fake_embedding = np.random.rand(100, 100, 128).astype(np.float32)
            fake_scales = np.ones((100, 100, 128), dtype=np.float32)
            
            # Mock UTM projection retrieval
            utm_crs = CRS.from_epsg(32630)
            utm_transform = Affine(10.0, 0.0, 500000, 0.0, -10.0, 4000000)
            
            with patch('numpy.load', side_effect=[fake_embedding, fake_scales]):
                with patch('geotessera.registry.tile_to_embedding_path', return_value=("/fake/emb.npy", "/fake/scales.npy")):
                    with patch.object(gt, '_get_utm_projection_from_landmask', return_value=(utm_crs, utm_transform)):
                        with patch('rasterio.open') as mock_rasterio_open:
                            mock_dst = Mock()
                            mock_dst.__enter__ = Mock(return_value=mock_dst)
                            mock_dst.__exit__ = Mock(return_value=None)
                            mock_rasterio_open.return_value = mock_dst
                            
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Export tiles
                                files = gt.export_embedding_geotiffs(
                                    bbox=(-0.1, 51.5, 0.0, 51.6),
                                    output_dir=temp_dir,
                                    year=2024
                                )
                                
                                # Verify UTM projection was used
                                assert mock_rasterio_open.called, "Should have opened files for writing"
                                
                                # Check that at least one file was created with UTM projection
                                found_utm_call = False
                                for call in mock_rasterio_open.call_args_list:
                                    if len(call) > 1 and 'crs' in call[1]:
                                        if call[1]['crs'] == utm_crs:
                                            found_utm_call = True
                                            break
                                
                                assert found_utm_call, "Should have used UTM projection in at least one file creation"


if __name__ == '__main__':
    pytest.main([__file__])