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
# Part 1: Import Reliability
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
    
    def test_missing_xarray_raises_import_error(self, monkeypatch):
        """
        When xarray is missing, load_air_quality should raise ImportError.
        """
        from airqualitylib import geospatial_airquality

        # Temporarily remove xarray from the module
        monkeypatch.setattr(geospatial_airquality, "xr", None, raising=False)

        with pytest.raises(ImportError, match="xarray is required"):
            geospatial_airquality.load_air_quality("dummy.nc")

    def test_missing_geopandas_raises_import_error(self, monkeypatch):
        """
        When geopandas is missing, load_admin_boundaries should raise ImportError.
        """
        from airqualitylib import geospatial_airquality

        # Temporarily remove geopandas from the module
        monkeypatch.setattr(geospatial_airquality, "gpd", None, raising=False)

        with pytest.raises(ImportError, match="geopandas is required"):
            geospatial_airquality.load_admin_boundaries(path="dummy.gpkg")


# =============================================================================
# Part 2: I/O (NetCDF / GeoPackage)
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
# Part 3: CRS Tools - CRS 
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
# Part 4: Clipping Operations 
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
# Part 5: Temporal Aggregation 
# =============================================================================

class TestTemporalAggregation:
    """Tests for time aggregation helpers (monthly_mean)."""

    @pytest.fixture(scope="session")
    def testdata_dir(self) -> Path:
        """Return the path to the tests/testdata directory."""
        return Path(__file__).parent / "testdata"

    @pytest.fixture(scope="session")
    def portugal_no2_path(self, testdata_dir: Path) -> Path:
        """Return the path to the sample NO2 subset NetCDF."""
        return testdata_dir / "portugal_no2_subset.nc"

    @pytest.fixture
    def da(self, portugal_no2_path: Path):
        """Load the sample air-quality cube once per test."""
        from airqualitylib.geospatial_airquality import load_air_quality
        return load_air_quality(str(portugal_no2_path))

    def test_monthly_mean_overall_matches_manual(self, da):
        """If month=None, monthly_mean should equal mean over the full time axis and drop 'time'."""
        from airqualitylib.geospatial_airquality import monthly_mean
        import xarray.testing as xrt

        result = monthly_mean(da)
        expected = da.mean(dim="time", skipna=True)

        assert "time" not in result.dims
        xrt.assert_allclose(result, expected)

    def test_monthly_mean_specific_month_matches_manual(self, da):
        """If month is provided, monthly_mean should match a manual time slice mean."""
        from airqualitylib.geospatial_airquality import monthly_mean
        import xarray.testing as xrt
        import pandas as pd

        month = "2022-02"
        result = monthly_mean(da, month=month, time_dim="time")

        # Robust month boundaries (avoids hardcoding 28/29/30/31 days)
        start = pd.Period(month, freq="M").start_time
        end = pd.Period(month, freq="M").end_time
        expected = da.sel(time=slice(start, end)).mean(dim="time", skipna=True)

        # Output should be 2D (lat, lon)
        assert set(result.dims) == {"lat", "lon"}
        xrt.assert_allclose(result, expected)

    def test_monthly_mean_raises_if_missing_time_dim(self):
        """monthly_mean should raise if the specified time_dim is not present."""
        from airqualitylib.geospatial_airquality import monthly_mean
        import numpy as np
        import xarray as xr

        da_no_time = xr.DataArray(
            np.random.rand(3, 4),
            dims=["lat", "lon"],
            coords={"lat": [1, 2, 3], "lon": [1, 2, 3, 4]},
        )

        with pytest.raises(ValueError, match="time_dim=.*not in da.dims"):
            monthly_mean(da_no_time)
   

# =============================================================================
# Part 6: Landcover Operations 
# =============================================================================

class TestLandcoverOperations:
    """Tests for landcover helpers using testdata (landcover.nc + portugal_no2_subset.nc)."""

    @pytest.fixture(scope="session")
    def testdata_dir(self) -> Path:
        """Return the path to the tests/testdata directory."""
        return Path(__file__).parent / "testdata"

    @pytest.fixture(scope="session")
    def landcover_path(self, testdata_dir: Path) -> Path:
        """Return the path to the sample landcover NetCDF."""
        return testdata_dir / "landcover.nc"

    @pytest.fixture(scope="session")
    def portugal_no2_path(self, testdata_dir: Path) -> Path:
        """Return the path to the sample NO2 subset NetCDF."""
        return testdata_dir / "portugal_no2_subset.nc"

    @pytest.fixture
    def lc(self, landcover_path: Path):
        """Load landcover once per test."""
        from airqualitylib.geospatial_airquality import load_landcover
        return load_landcover(str(landcover_path), var="lccs_class")

    @pytest.fixture
    def aq(self, portugal_no2_path: Path):
        """Load AQ cube once per test."""
        from airqualitylib.geospatial_airquality import load_air_quality
        return load_air_quality(str(portugal_no2_path))

    def test_load_landcover_returns_2d_lat_lon(self, lc):
        """load_landcover should normalize dims to (lat, lon) and squeeze extra dims."""
        assert lc.dims == ("lat", "lon")
        assert lc.sizes["lat"] > 0 and lc.sizes["lon"] > 0

        # Coordinates should be monotonic (either ascending or descending is fine)
        lat0, lat1 = float(lc.lat[0]), float(lc.lat[-1])
        assert lat0 != lat1

    def test_align_categorical_to_reference_matches_aq_grid(self, lc, aq):
        """Aligned landcover must match the AQ spatial grid."""
        from airqualitylib.geospatial_airquality import align_categorical_to_reference
        import numpy as np

        ref = aq.isel(time=0) if "time" in aq.dims else aq
        lc_on_ref = align_categorical_to_reference(lc, ref, method="nearest")

        assert lc_on_ref.shape == ref.shape
        np.testing.assert_allclose(lc_on_ref.lat.values, ref.lat.values)
        np.testing.assert_allclose(lc_on_ref.lon.values, ref.lon.values)

    def test_reclass_landcover_default_mapping_outputs_0_to_6(self, lc):
        """Default reclass should output only {0..6} and set scheme metadata."""
        from airqualitylib.geospatial_airquality import reclass_landcover
        import numpy as np

        lc6 = reclass_landcover(lc)
        unique_vals = set(np.unique(lc6.values))

        assert unique_vals <= set(range(7))  # 0 is nodata_out by default
        assert lc6.attrs.get("scheme") == "6class"

    def test_reclass_landcover_custom_mapping_and_nodata(self, lc):
        """Custom mapping should remap one code and use nodata_out for others."""
        from airqualitylib.geospatial_airquality import reclass_landcover
        import numpy as np

        sample_val = int(np.asarray(lc.values).ravel()[0])
        lc_custom = reclass_landcover(lc, mapping={sample_val: 9}, nodata_out=-1, name="custom")

        unique_vals = set(np.unique(lc_custom.values))
        assert unique_vals <= {9, -1}
        assert lc_custom.name == "custom"

    def test_overlay_aq_with_landcover_masks_nan(self, lc, aq):
        """If mask_by_aq=True, landcover should be NaN where AQ is NaN."""
        try:
            import rioxarray  # noqa: F401
        except ImportError:
            pytest.skip("rioxarray is not available")

        from airqualitylib.geospatial_airquality import overlay_aq_with_landcover
        import numpy as np

        # Typical use: overlay expects a 2D AQ map (monthly mean), but it can accept 3D too.
        # Test with a 2D slice to keep semantics clean.
        aq2d = aq.isel(time=0).copy() if "time" in aq.dims else aq.copy()
        aq2d.values[0, 0] = np.nan  # inject a NaN to test masking

        aq_out, lc_on_aq = overlay_aq_with_landcover(aq2d, lc, reclass=True, mask_by_aq=True)

        assert aq_out.dims == ("lat", "lon")
        assert set(lc_on_aq.dims) == {"lat", "lon"}
        assert lc_on_aq.sizes["lat"] == aq_out.sizes["lat"]
        assert lc_on_aq.sizes["lon"] == aq_out.sizes["lon"]

        # Masking: where AQ is NaN, LC should be NaN too
        assert np.isnan(lc_on_aq.values[0, 0])

    def test_get_landcover_6_colormap(self):
        """Colormap helper should return 6 colors and matching norm/labels."""
        from airqualitylib.geospatial_airquality import get_landcover_6_colormap

        cmap, norm, labels = get_landcover_6_colormap()

        assert cmap.N == 6
        assert len(labels) == 6
        assert len(norm.boundaries) == 7

# =============================================================================
# Part 7: Stats 
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
        """Test basic statistics output using synthetic data on a shared grid."""
        from airqualitylib.geospatial_airquality import (
            landcover_stats, LANDCOVER_6_LABELS
        )
        import xarray as xr
        import numpy as np
 
        # Create synthetic AQ data (lat=3, lon=4)
        aq2d = xr.DataArray(
            np.array([[10.0, 20.0, 30.0, 50.0],
                      [15.0, 25.0, 35.0, 45.0],
                      [12.0, 22.0, 32.0, 55.0]]),
            coords={"lat": [35.0, 36.0, 37.0], "lon": [5.0, 6.0, 7.0, 8.0]},
            dims=["lat", "lon"],
        )

        # Create synthetic landcover data on the same grid
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

    def test_landcover_stats_ignores_nodata_class(self):
        """Landcover class 0 (nodata) should be ignored in statistics."""
        from airqualitylib.geospatial_airquality import landcover_stats
        import xarray as xr
        import numpy as np

        aq2d = xr.DataArray(
            np.array([[10.0, 20.0],
                    [30.0, 40.0]]),
            coords={"lat": [1, 2], "lon": [1, 2]},
            dims=["lat", "lon"],
        )

        # landcover includes 0 (nodata)
        lc6 = xr.DataArray(
            np.array([[0, 1],
                    [2, 0]]),
            coords={"lat": [1, 2], "lon": [1, 2]},
            dims=["lat", "lon"],
        )

        df = landcover_stats(aq2d, lc6)

        assert 0 not in df["class"].values

    def test_landcover_stats_with_max_and_median(self, landcover_path, portugal_no2_path):
        """Test statistics including max and median values using synthetic data."""
        from airqualitylib.geospatial_airquality import (
            landcover_stats, LANDCOVER_6_LABELS
        )
        import xarray as xr
        import numpy as np

         # Create synthetic AQ data
        aq2d = xr.DataArray(
            np.array([[10.0, 20.0, 30.0],
                      [15.0, 25.0, 35.0]]),
            coords={"lat": [35.0, 36.0], "lon": [5.0, 6.0, 7.0]},
            dims=["lat", "lon"],
        )

         # Create synthetic landcover data
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

    def test_landcover_stats_handles_nan_aq(self):
        """NaN values in AQ should be safely ignored."""
        from airqualitylib.geospatial_airquality import landcover_stats
        import xarray as xr
        import numpy as np

        aq2d = xr.DataArray(
            np.array([[np.nan, 20.0],
                    [30.0, np.nan]]),
            coords={"lat": [1, 2], "lon": [1, 2]},
            dims=["lat", "lon"],
        )

        lc6 = xr.DataArray(
            np.array([[1, 1],
                    [1, 1]]),
            coords={"lat": [1, 2], "lon": [1, 2]},
            dims=["lat", "lon"],
        )

        df = landcover_stats(aq2d, lc6)

        assert not df.empty
        assert np.isfinite(df["mean"]).all()