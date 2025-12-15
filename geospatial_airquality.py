"""
AirQuality-Lib
A lightweight geospatial library for air quality analysis.

Core workflow (mainline):
1) Load air quality raster/datacube (PM2.5 / PM10 / NO2)
2) Load administrative boundaries (vector)
3) Ensure CRS consistency
4) Clip to a country/region
5) Temporal aggregation (annual / seasonal)
6) Indicators (min/max/mean + exceedance ratio)
7) Zonal statistics (per administrative unit)
8) Period comparison (change/anomaly)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Tuple

import numpy as np

# Optional imports: keep them here so the module can be imported even if deps are missing.
# Real implementations will require: geopandas, xarray, rioxarray, rasterio, pandas, shapely
try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None  # type: ignore

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# -----------------------------
# Configuration / constants
# -----------------------------

@dataclass(frozen=True)
class Thresholds:
    """Default threshold configuration (can be overridden by user)."""
    pm25_ugm3: float = 25.0
    pm10_ugm3: float = 50.0
    no2_ugm3: float = 40.0


DEFAULT_THRESHOLDS = Thresholds()


# -----------------------------
# I/O
# -----------------------------

def load_admin_boundaries(path: str):
    """
    Load administrative boundaries from GeoJSON/GPKG/Shapefile.

    Parameters
    ----------
    path : str
        Path to vector file.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if gpd is None:
        raise ImportError("geopandas is required for load_admin_boundaries().")
    return gpd.read_file(path)


def load_air_quality(path: str, var: Optional[str] = None):
    """
    Load air quality data from NetCDF (preferred) or GeoTIFF.

    Parameters
    ----------
    path : str
        Path to NetCDF/GeoTIFF.
    var : str, optional
        Variable name if NetCDF contains multiple variables.

    Returns
    -------
    xarray.DataArray
    """
    if xr is None:
        raise ImportError("xarray is required for load_air_quality().")
    # NOTE: skeleton only. Real implementation will decide by extension and use rioxarray/rasterio as needed.
    raise NotImplementedError("Implement NetCDF/GeoTIFF loading here.")


# -----------------------------
# CRS / spatial
# -----------------------------

def ensure_same_crs(admin_gdf, target_crs: str = "EPSG:4326"):
    """
    Ensure boundaries are in target CRS (default EPSG:4326).

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if gpd is None:
        raise ImportError("geopandas is required for ensure_same_crs().")
    if admin_gdf.crs is None or str(admin_gdf.crs) != target_crs:
        admin_gdf = admin_gdf.to_crs(target_crs)
    return admin_gdf


def clip_to_aoi(da, aoi_gdf):
    """
    Clip air quality DataArray to AOI polygons.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with spatial coords.
    aoi_gdf : geopandas.GeoDataFrame
        AOI boundary polygons.

    Returns
    -------
    xarray.DataArray
    """
    # NOTE: skeleton only. Real implementation will use rioxarray.clip().
    raise NotImplementedError("Implement rioxarray-based clip here.")


# -----------------------------
# Temporal aggregation
# -----------------------------

def annual_mean(da, time_dim: str = "time"):
    """Compute annual mean along time dimension."""
    raise NotImplementedError("Implement with xarray resample/groupby.")


def seasonal_mean(da, time_dim: str = "time"):
    """Compute seasonal mean (DJF/MAM/JJA/SON)."""
    raise NotImplementedError("Implement with xarray groupby('time.season').")


# -----------------------------
# Indicators
# -----------------------------

def summary_stats(values: Union[np.ndarray, Sequence[float]]) -> Dict[str, float]:
    """
    Compute min/max/mean for an array-like input (NaNs ignored).

    This is dependency-light so it can be tested without xarray/raster data.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}


def exceedance_ratio(values: Union[np.ndarray, Sequence[float]], threshold: float) -> float:
    """
    Compute the fraction of values strictly greater than threshold (NaNs ignored).

    Returns a number in [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float((arr > threshold).sum() / arr.size)


# -----------------------------
# Zonal statistics / change
# -----------------------------

def zonal_mean(da, admin_gdf, id_col: str):
    """
    Compute mean air quality per administrative unit.

    Notes
    -----
    Skeleton only. Real implementation will:
    - clip/mask per polygon
    - compute spatial mean per time step (optional)
    - return a DataFrame
    """
    if pd is None:
        raise ImportError("pandas is required for zonal_mean().")
    raise NotImplementedError("Implement zonal mean here.")


def period_diff(da, period_a: Tuple[str, str], period_b: Tuple[str, str], agg: str = "annual"):
    """
    Compare two periods (A vs B) and return difference (B - A).

    Example
    -------
    period_a = ("2010-01-01", "2014-12-31")
    period_b = ("2020-01-01", "2024-12-31")
    """
    raise NotImplementedError("Implement period difference here.")


# -----------------------------
# Sanity check
# -----------------------------

def hello():
    """Simple test function."""
    print("AirQuality-Lib skeleton is ready!")
