"""
Test suite for geospatial_airquality module.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Part 1: Import Reliability - 导入与可选依赖测试
# =============================================================================

class TestImportReliability:
    """Test module imports and optional dependency handling."""
    
    def test_module_import(self):
        """Test that the module and key functions can be imported."""
        from airqualitylib import geospatial_airquality
        from airqualitylib.geospatial_airquality import (
            load_air_quality,
            load_admin_boundaries,
        )
        assert geospatial_airquality is not None
        assert callable(load_air_quality)
        assert callable(load_admin_boundaries)
    
    def test_missing_dependencies_raise_errors(self):
        """Test that missing dependencies raise ImportError."""
        from airqualitylib import geospatial_airquality
        
        original_xr = geospatial_airquality.xr
        geospatial_airquality.xr = None
        try:
            with pytest.raises(ImportError, match="xarray is required"):
                geospatial_airquality.load_air_quality("dummy.nc")
        finally:
            geospatial_airquality.xr = original_xr


# =============================================================================
# Part 2: I/O (NetCDF / GeoPackage) - 测试读取与写入功能
# =============================================================================

class TestIOFunctions:
    """Test I/O functions for NetCDF and GeoPackage files."""
    
    @pytest.fixture
    def testdata_dir(self):
        """Get the testdata directory path."""
        return Path(__file__).parent / "testdata"
    
    @pytest.fixture
    def portugal_no2_path(self, testdata_dir):
        """Path to portugal_no2_subset.nc."""
        return str(testdata_dir / "portugal_no2_subset.nc")
    
    @pytest.fixture
    def eu_boundaries_path(self, testdata_dir):
        """Path to EU_countryboundaries.gpkg."""
        return str(testdata_dir / "EU_countryboundaries.gpkg")
    
    @pytest.fixture
    def landcover_path(self, testdata_dir):
        """Path to landcover.nc."""
        return str(testdata_dir / "landcover.nc")
    
    # -------------------------------------------------------------------------
    # Test load_air_quality
    # -------------------------------------------------------------------------
    
    def test_load_air_quality_basic(self, portugal_no2_path):
        """Test basic loading of air quality NetCDF file."""
        from airqualitylib.geospatial_airquality import load_air_quality
        import xarray as xr
        
        data = load_air_quality(portugal_no2_path)
        
        assert isinstance(data, xr.DataArray)
        assert data.size > 0
        assert len(data.dims) >= 2  # Has spatial dimensions
    
    def test_load_air_quality_invalid_inputs(self, portugal_no2_path):
        """Test error handling for invalid inputs."""
        from airqualitylib.geospatial_airquality import load_air_quality
        
        # Invalid variable name
        with pytest.raises(KeyError, match="Variable 'nonexistent'"):
            load_air_quality(portugal_no2_path, var="nonexistent")
        
        # Nonexistent file
        with pytest.raises((FileNotFoundError, OSError)):
            load_air_quality("nonexistent_file.nc")
    
    # -------------------------------------------------------------------------
    # Test load_admin_boundaries
    # -------------------------------------------------------------------------
    
    def test_load_admin_boundaries_basic(self, eu_boundaries_path):
        """Test basic loading of administrative boundaries."""
        from airqualitylib.geospatial_airquality import load_admin_boundaries
        import geopandas as gpd
        
        gdf = load_admin_boundaries(path=eu_boundaries_path)
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'geometry' in gdf.columns
        assert not gdf.empty
        assert gdf.crs.to_epsg() == 4326  # WGS84
    
    def test_load_admin_boundaries_error_handling(self):
        """Test error handling for boundaries loading."""
        from airqualitylib.geospatial_airquality import load_admin_boundaries
        
        with pytest.raises((FileNotFoundError, Exception)):
            load_admin_boundaries(path="nonexistent.gpkg")
    
    # -------------------------------------------------------------------------
    # Test get_admin_boundary
    # -------------------------------------------------------------------------
    
    def test_get_admin_boundary_basic(self, eu_boundaries_path):
        """Test getting a specific administrative boundary."""
        from airqualitylib.geospatial_airquality import get_admin_boundary, load_admin_boundaries
        import geopandas as gpd
        
        # Get available names
        gdf_all = load_admin_boundaries(path=eu_boundaries_path)
        
        if 'NAME' in gdf_all.columns:
            available_names = gdf_all['NAME'].dropna().astype(str).unique()
            if len(available_names) > 0:
                test_name = available_names[0]
                gdf = get_admin_boundary(test_name, by="NAME", path=eu_boundaries_path)
                
                assert isinstance(gdf, gpd.GeoDataFrame)
                assert not gdf.empty
                assert len(gdf) <= len(gdf_all)
    
    def test_get_admin_boundary_error_handling(self, eu_boundaries_path):
        """Test error handling for get_admin_boundary."""
        from airqualitylib.geospatial_airquality import get_admin_boundary
        
        # Invalid query
        with pytest.raises(ValueError, match="No match for"):
            get_admin_boundary("NonexistentCountry123", path=eu_boundaries_path)
        
        # Invalid column
        with pytest.raises(ValueError, match="Column .* not found"):
            get_admin_boundary("Something", by="NONEXISTENT_COLUMN", path=eu_boundaries_path)
    
    # -------------------------------------------------------------------------
    # Test landcover loading
    # -------------------------------------------------------------------------
    
    def test_load_landcover_basic(self, landcover_path):
        """Test basic loading of landcover NetCDF file."""
        from airqualitylib.geospatial_airquality import load_air_quality
        import xarray as xr
        
        # Find the main landcover variable
        ds = xr.open_dataset(landcover_path)
        available_vars = list(ds.data_vars)
        ds.close()
        
        var_to_load = None
        for var_name in ['lccs_class', 'landcover', 'land_cover', 'lc']:
            if var_name in available_vars:
                var_to_load = var_name
                break
        if var_to_load is None and available_vars:
            var_to_load = available_vars[0]
        
        data = load_air_quality(landcover_path, var=var_to_load)
        
        assert isinstance(data, xr.DataArray)
        assert data.size > 0
        assert len(data.dims) >= 2  # Has spatial dimensions


# =============================================================================
# Part 3: CRS Tools - CRS 工具测试
# =============================================================================

class TestCRSTools:
    """Test CRS utility functions."""
    
    @pytest.fixture
    def testdata_dir(self):
        """Get the testdata directory path."""
        return Path(__file__).parent / "testdata"
    
    @pytest.fixture
    def eu_boundaries_path(self, testdata_dir):
        """Path to EU_countryboundaries.gpkg."""
        return str(testdata_dir / "EU_countryboundaries.gpkg")
    
    @pytest.fixture
    def portugal_no2_path(self, testdata_dir):
        """Path to portugal_no2_subset.nc."""
        return str(testdata_dir / "portugal_no2_subset.nc")
    
    def test_ensure_crs_geodataframe(self, eu_boundaries_path):
        """Test ensure_crs with GeoDataFrame."""
        from airqualitylib.geospatial_airquality import load_admin_boundaries, ensure_crs
        import geopandas as gpd
        
        # Load boundaries
        gdf = load_admin_boundaries(path=eu_boundaries_path)
        
        # Test 1: Object already in WGS84 should be returned unchanged
        gdf_wgs84 = ensure_crs(gdf, epsg=4326)
        assert isinstance(gdf_wgs84, gpd.GeoDataFrame)
        assert gdf_wgs84.crs.to_epsg() == 4326
        
        # Test 2: Reproject to different CRS
        gdf_utm = gdf.to_crs("EPSG:3857")  # Web Mercator
        gdf_back = ensure_crs(gdf_utm, epsg=4326)
        assert gdf_back.crs.to_epsg() == 4326
        
        # Test 3: Object with no CRS should have CRS set
        gdf_no_crs = gdf.copy()
        gdf_no_crs.crs = None
        gdf_with_crs = ensure_crs(gdf_no_crs, epsg=4326)
        assert gdf_with_crs.crs is not None
        assert gdf_with_crs.crs.to_epsg() == 4326
    
    def test_ensure_crs_xarray(self, portugal_no2_path):
        """Test ensure_crs with xarray DataArray."""
        from airqualitylib.geospatial_airquality import load_air_quality, ensure_crs
        import xarray as xr
        
        # Check if rioxarray is available
        try:
            import rioxarray  # noqa: F401
            has_rioxarray = True
        except ImportError:
            has_rioxarray = False
        
        if not has_rioxarray:
            pytest.skip("rioxarray is not available")
        
        # Load air quality data
        data = load_air_quality(portugal_no2_path)
        
        # Test 1: DataArray without CRS should have CRS set
        data_with_crs = ensure_crs(data, epsg=4326)
        assert isinstance(data_with_crs, xr.DataArray)
        
        # Check if CRS was set (rioxarray adds .rio accessor)
        if hasattr(data_with_crs, 'rio') and hasattr(data_with_crs.rio, 'crs'):
            assert data_with_crs.rio.crs is not None
        
        # Test 2: DataArray already with CRS should be returned
        data_check = ensure_crs(data_with_crs, epsg=4326)
        assert isinstance(data_check, xr.DataArray)
    
    def test_ensure_crs_invalid_type(self):
        """Test ensure_crs with invalid object types."""
        from airqualitylib.geospatial_airquality import ensure_crs
        import numpy as np
        
        # Test with invalid types
        invalid_objects = [
            "string",
            123,
            [1, 2, 3],
            {"key": "value"},
            np.array([1, 2, 3])
        ]
        
        for obj in invalid_objects:
            with pytest.raises(TypeError, match="ensure_crs\\(\\) only supports"):
                ensure_crs(obj)


# =============================================================================
# Part 4: Clipping Operations - 裁剪操作测试
# =============================================================================

class TestClippingOperations:
    """Test clipping operations for spatial data."""
    
    @pytest.fixture
    def testdata_dir(self):
        """Get the testdata directory path."""
        return Path(__file__).parent / "testdata"
    
    @pytest.fixture
    def portugal_no2_path(self, testdata_dir):
        """Path to portugal_no2_subset.nc."""
        return str(testdata_dir / "portugal_no2_subset.nc")
    
    @pytest.fixture
    def eu_boundaries_path(self, testdata_dir):
        """Path to EU_countryboundaries.gpkg."""
        return str(testdata_dir / "EU_countryboundaries.gpkg")
    
    def test_clip_to_aoi_basic(self, portugal_no2_path, eu_boundaries_path):
        """Test clipping data to AOI boundary using data bbox intersection."""
        from airqualitylib.geospatial_airquality import (
            load_air_quality,
            load_admin_boundaries,
            clip_to_aoi
        )
        from shapely.geometry import box
        import xarray as xr
        
        # Check if rioxarray is available
        try:
            import rioxarray  # noqa: F401
            has_rioxarray = True
        except ImportError:
            has_rioxarray = False
        
        if not has_rioxarray:
            pytest.skip("rioxarray is not available")
        
        # Load air quality data
        data = load_air_quality(portugal_no2_path)
        
        # Extract data bbox from coordinates
        lon_min = float(data.lon.min())
        lon_max = float(data.lon.max())
        lat_min = float(data.lat.min())
        lat_max = float(data.lat.max())
        data_bbox = box(lon_min, lat_min, lon_max, lat_max)
        
        # Load boundaries
        gdf = load_admin_boundaries(path=eu_boundaries_path)
        
        # Find boundaries that intersect with data bbox
        intersecting = gdf[gdf.geometry.intersects(data_bbox)]
        
        if len(intersecting) > 0:
            # Use the first intersecting boundary as AOI
            aoi = intersecting.iloc[[0]]
            
            # Clip data to AOI
            clipped = clip_to_aoi(data, aoi)
            
            # Verify result
            assert isinstance(clipped, xr.DataArray)
            assert clipped.size > 0
            
            # Clipped data should be smaller or equal to original
            assert clipped.size <= data.size
            
            # Should still have lat/lon dimensions
            assert 'lat' in clipped.dims or 'latitude' in clipped.dims
            assert 'lon' in clipped.dims or 'longitude' in clipped.dims
    
    def test_clip_to_aoi_missing_coordinates(self, eu_boundaries_path):
        """Test error handling when data lacks lat/lon coordinates."""
        from airqualitylib.geospatial_airquality import (
            load_admin_boundaries,
            clip_to_aoi
        )
        import xarray as xr
        import numpy as np
        
        # Check if rioxarray is available
        try:
            import rioxarray  # noqa: F401
            has_rioxarray = True
        except ImportError:
            has_rioxarray = False
        
        if not has_rioxarray:
            pytest.skip("rioxarray is not available")
        
        # Create a DataArray without lat/lon dimensions
        data_no_coords = xr.DataArray(
            np.random.rand(10, 10),
            dims=['x', 'y'],  # Wrong dimension names
            coords={'x': np.arange(10), 'y': np.arange(10)}
        )
        
        # Load boundaries
        gdf = load_admin_boundaries(path=eu_boundaries_path)
        aoi = gdf.iloc[[0]]
        
        # Should raise ValueError about missing lat/lon
        with pytest.raises(ValueError, match="clip_to_aoi expects dims include"):
            clip_to_aoi(data_no_coords, aoi)


# =============================================================================
# Part 5: Temporal Aggregation - 时间聚合测试
# =============================================================================

class TestTemporalAggregation:
    """Test monthly aggregation utilities."""

    @pytest.fixture
    def testdata_dir(self):
        return Path(__file__).parent / "testdata"

    @pytest.fixture
    def portugal_no2_path(self, testdata_dir):
        return str(testdata_dir / "portugal_no2_subset.nc")

    def test_monthly_mean_overall(self, portugal_no2_path):
        """整体月平均：输出应与手工 mean 匹配且移除时间维度。"""
        from airqualitylib.geospatial_airquality import load_air_quality, monthly_mean
        import xarray.testing as xrt

        da = load_air_quality(portugal_no2_path)

        result = monthly_mean(da)
        expected = da.mean(dim="time", skipna=True)

        assert "time" not in result.dims
        xrt.assert_allclose(result, expected)

    def test_monthly_mean_specific_month(self, portugal_no2_path):
        """特定月份平均："""
        from airqualitylib.geospatial_airquality import load_air_quality, monthly_mean
        import xarray.testing as xrt

        da = load_air_quality(portugal_no2_path)

        result = monthly_mean(da, month="2022-02", time_dim="time")
        manual_slice = da.sel(time=slice("2022-02-01", "2022-02-28"))
        expected = manual_slice.mean(dim="time", skipna=True)

        assert result.dims == ("lat", "lon")
        xrt.assert_allclose(result, expected)

    def test_monthly_mean_missing_time_dim(self):
        """缺少时间维度时应报错。"""
        from airqualitylib.geospatial_airquality import monthly_mean
        import numpy as np
        import xarray as xr

        da_no_time = xr.DataArray(
            np.random.rand(3, 4),
            dims=["lat", "lon"],
            coords={"lat": [1, 2, 3], "lon": [1, 2, 3, 4]},
        )

        with pytest.raises(ValueError, match="time_dim='time' not in da.dims"):
            monthly_mean(da_no_time)


# =============================================================================
# Part 6: Landcover Operations - 土地覆盖操作测试
# =============================================================================

class TestLandcoverOperations:
    """Use真实 testdata/landcover.nc 与 NO₂ 数据进行土地覆盖相关测试。"""

    @pytest.fixture
    def testdata_dir(self):
        return Path(__file__).parent / "testdata"

    @pytest.fixture
    def landcover_path(self, testdata_dir):
        return str(testdata_dir / "landcover.nc")

    @pytest.fixture
    def portugal_no2_path(self, testdata_dir):
        return str(testdata_dir / "portugal_no2_subset.nc")

    def test_load_landcover_uses_lat_lon_and_squeezes_time(self, landcover_path):
        from airqualitylib.geospatial_airquality import load_landcover

        lc = load_landcover(landcover_path,var="lccs_class")

        assert lc.dims == ("lat", "lon")
        assert lc.sizes["lat"] == 6 and lc.sizes["lon"] == 5
        assert float(lc.lat[0]) < float(lc.lat[-1])  # 保持递增

    def test_align_categorical_to_reference_matches_aq_grid(self, landcover_path, portugal_no2_path):
        from airqualitylib.geospatial_airquality import load_landcover, load_air_quality, align_categorical_to_reference
        import numpy as np

        lc = load_landcover(landcover_path,var="lccs_class")
        aq = load_air_quality(portugal_no2_path)

        lc_on_aq = align_categorical_to_reference(lc, aq, method="nearest")

        assert lc_on_aq.shape == aq.isel(time=0).shape
        np.testing.assert_allclose(lc_on_aq.lat.values, aq.lat.values)
        np.testing.assert_allclose(lc_on_aq.lon.values, aq.lon.values)

    def test_reclass_landcover_default_mapping(self, landcover_path):
        from airqualitylib.geospatial_airquality import load_landcover, reclass_landcover
        import numpy as np

        lc = load_landcover(landcover_path,var="lccs_class")
        lc6 = reclass_landcover(lc)

        unique_vals = set(np.unique(lc6.values))
        # 默认映射输出为 0..6（0 为 nodata_out）
        assert unique_vals <= set(range(7))
        assert lc6.attrs.get("scheme") == "6class"

    def test_reclass_landcover_custom_mapping_and_nodata(self, landcover_path):
        from airqualitylib.geospatial_airquality import load_landcover, reclass_landcover
        import numpy as np

        lc = load_landcover(landcover_path,var="lccs_class")
        sample_val = int(lc.values.flat[0])

        lc_custom = reclass_landcover(lc, mapping={sample_val: 9}, nodata_out=-1, name="custom")

        unique_vals = set(np.unique(lc_custom.values))
        assert unique_vals <= {9, -1}
        assert lc_custom.name == "custom"

    def test_overlay_aq_with_landcover_masks_nan(self, landcover_path, portugal_no2_path):
        try:
            import rioxarray  # noqa: F401
        except ImportError:
            pytest.skip("rioxarray is not available")

        from airqualitylib.geospatial_airquality import load_landcover, load_air_quality, overlay_aq_with_landcover
        import numpy as np

        aq = load_air_quality(portugal_no2_path)
        aq = aq.copy()
        aq[0, 0, 0] = np.nan  # 注入一个 NaN 以测试掩膜

        lc = load_landcover(landcover_path,var="lccs_class")

        aq_out, lc_on_aq = overlay_aq_with_landcover(aq, lc, reclass=True, mask_by_aq=True)

        assert set(lc_on_aq.dims) == set(aq_out.dims)
        for d in aq_out.dims:
            assert lc_on_aq.sizes[d] == aq_out.sizes[d]
        lc_t = lc_on_aq.transpose(*aq_out.dims)
        assert np.isnan(lc_t[0, 0, 0])

    def test_get_landcover_6_colormap(self):
        from airqualitylib.geospatial_airquality import get_landcover_6_colormap

        cmap, norm, labels = get_landcover_6_colormap()

        assert cmap.N == 6
        assert len(labels) == 6
        assert len(norm.boundaries) == 7


# =============================================================================
# Part 7: Stats - 统计分析测试
# =============================================================================

class TestStats:
    """Test landcover-based statistics on real test data."""

    @pytest.fixture
    def testdata_dir(self):
        return Path(__file__).parent / "testdata"

    @pytest.fixture
    def landcover_path(self, testdata_dir):
        return str(testdata_dir / "landcover.nc")

    @pytest.fixture
    def portugal_no2_path(self, testdata_dir):
        return str(testdata_dir / "portugal_no2_subset.nc")

    def test_landcover_stats_basic(self, landcover_path, portugal_no2_path):
        """测试基本统计功能，使用合成数据确保地理范围一致。"""
        from airqualitylib.geospatial_airquality import (
            landcover_stats, LANDCOVER_6_LABELS
        )
        import xarray as xr
        import numpy as np

        # 创建合成 AQ 数据（lat=3, lon=4）
        aq2d = xr.DataArray(
            np.array([[10.0, 20.0, 30.0, 50.0],
                      [15.0, 25.0, 35.0, 45.0],
                      [12.0, 22.0, 32.0, 55.0]]),
            coords={"lat": [35.0, 36.0, 37.0], "lon": [5.0, 6.0, 7.0, 8.0]},
            dims=["lat", "lon"],
        )

        # 创建合成 landcover 数据（相同网格）
        lc6 = xr.DataArray(
            np.array([[1, 2, 3, 5],
                      [2, 3, 5, 1],
                      [3, 5, 1, 2]]),
            coords={"lat": [35.0, 36.0, 37.0], "lon": [5.0, 6.0, 7.0, 8.0]},
            dims=["lat", "lon"],
        )

        df = landcover_stats(aq2d, lc6, class_labels=LANDCOVER_6_LABELS)

        assert not df.empty, f"DataFrame is empty! df={df}"
        required_cols = {"class", "label", "mean", "exceed_ratio"}
        assert required_cols.issubset(df.columns), f"Missing columns. Expected {required_cols}, got {df.columns.tolist()}"
        assert (df["exceed_ratio"] >= 0).all() and (df["exceed_ratio"] <= 1).all()


    def test_landcover_stats_with_max_and_median(self, landcover_path, portugal_no2_path):
        """测试包含最大值和中位数的统计，使用合成数据。"""
        from airqualitylib.geospatial_airquality import (
            landcover_stats, LANDCOVER_6_LABELS
        )
        import xarray as xr
        import numpy as np

        # 创建合成 AQ 数据
        aq2d = xr.DataArray(
            np.array([[10.0, 20.0, 30.0],
                      [15.0, 25.0, 35.0]]),
            coords={"lat": [35.0, 36.0], "lon": [5.0, 6.0, 7.0]},
            dims=["lat", "lon"],
        )

        # 创建合成 landcover 数据
        lc6 = xr.DataArray(
            np.array([[1, 2, 3],
                      [2, 3, 1]]),
            coords={"lat": [35.0, 36.0], "lon": [5.0, 6.0, 7.0]},
            dims=["lat", "lon"],
        )

        df = landcover_stats(aq2d, lc6, include_max=True, include_median=True, class_labels=LANDCOVER_6_LABELS)

        assert "max" in df.columns and "median" in df.columns
        assert (df["max"] >= df["mean"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
