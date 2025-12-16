"""
AirQuality-Lib
A lightweight geospatial library for air quality analysis.

Core workflow (mainline):
1️⃣ 读取空气质量栅格（NetCDF, datacube）
✔ load_air_quality()
2️⃣ 读取行政区边界（vector）
✔ load_admin_boundaries() / get_admin_boundary()
3️⃣ CRS 统一 & 裁剪 AOI
✔ ensure_crs() + clip_to_aoi()
4️⃣ 时间聚合（月均）
✔ monthly_mean()
5️⃣ 叠加土地利用图（categorical raster）
✔ raster × raster overlay（
6️⃣ 区域统计（zonal stats）
按：
	•	行政区
	•	土地利用类型
7️⃣ 导出与可视化
2.生成土地类型reclassiff ied_.tif图
3.CAMP POLLUTANT月度statistics 叠加土地类型的图 
4.	统计图：按土地利用类型的 NO₂（平均值 + 超标比例）
5.	总结表：每个 landcover 类别的像元占比、mean、exceed_ratio、（可选：max / median）

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Tuple

import numpy as np
import importlib
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

def load_air_quality(path: str, var: Optional[str] = None, chunks: Optional[dict] = None):
    """
    Load air quality data from a NetCDF (.nc) file as an xarray.DataArray.

    Notes
    -----
    - Uses lazy loading (does not read full data into memory immediately).
    - If `var` is None and the file contains exactly one data variable, it will be selected automatically.
    - `chunks` enables dask chunking (optional). Example: {"time": 24}.

    Parameters
    ----------
    path : str
        Path to NetCDF file.
    var : str, optional
        Variable name to load (e.g., "no2", "pm10", "pm2p5").
    chunks : dict, optional
        Chunk sizes for dask. If None, no chunking.

    Returns
    -------
    xr.DataArray
        The selected variable as a DataArray.
    """
    if xr is None:
        raise ImportError("xarray is required for load_air_quality().")

    # Open lazily. If chunks is provided, xarray will use dask.
    ds = xr.open_dataset(path, chunks=chunks)

    data_vars = list(ds.data_vars)
    if var is None:
        if len(data_vars) == 1:
            var = data_vars[0]
        else:
            raise ValueError(
                f"Multiple variables found in NetCDF: {data_vars}. "
                "Please specify var=..."
            )

    if var not in ds.data_vars:
        raise KeyError(f"Variable '{var}' not found. Available variables: {data_vars}")

    return ds[var]



def load_admin_boundaries(path: Optional[str] = None, layer: str = "boundaries"):
    """
    Load built-in Europe administrative boundaries (GeoPackage) or a user-provided vector file.

    Default built-in file:
    data/boundaries/Europe_administrative_boundaries.gpkg
    """
    if gpd is None:
        raise ImportError("geopandas is required for load_admin_boundaries().")

    if path is None:
        path = "data/boundaries/Europe_administrative_boundaries.gpkg"

    gdf = gpd.read_file(path, layer=layer)

    # Ensure WGS84 for consistent clipping with lat/lon grids
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf

def get_admin_boundary(
    query: str,
    by: str = "NAME",
    path: Optional[str] = None,
    layer: str = "boundaries"
):
    """
    Get a single boundary feature from built-in Europe boundaries.

    Parameters
    ----------
    query : str
        Value to match, e.g. "Italy" or "IT" or "ITA".
    by : str
        Column name to match. Recommended: "NAME" (default), "iso_a2", "ISO_A3".
    path : str, optional
        Path to GPKG. If None, use built-in.
    layer : str
        GPKG layer name.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the matched feature(s).
    """
    gdf = load_admin_boundaries(path=path, layer=layer)

    if by not in gdf.columns:
        raise ValueError(f"Column '{by}' not found. Available: {list(gdf.columns)}")

    # string-safe match
    out = gdf[gdf[by].astype(str) == str(query)]

    if out.empty:
        # show helpful examples
        examples = gdf[by].dropna().astype(str).unique()[:20]
        raise ValueError(
            f"No match for {query!r} in column '{by}'. Example values: {list(examples)}"
        )

    return out
    

# -----------------------------
# CRS 
# -----------------------------

from typing import Union, Optional

def ensure_crs(obj, epsg: int = 4326):
    """
    Ensure CRS for either:
    - GeoPandas GeoDataFrame/GeoSeries: reproject to EPSG
    - xarray DataArray/Dataset (with rioxarray): attach EPSG if missing

    Notes
    -----
    For CAMS data with lat/lon, we usually only *attach* EPSG:4326 (no reprojection).
    """
    target = f"EPSG:{epsg}"

    # --- Vector: GeoPandas
    try:
        import geopandas as gpd
        if isinstance(obj, (gpd.GeoDataFrame, gpd.GeoSeries)):
            if obj.crs is None:
                # If missing CRS, assume it is already EPSG:4326 (common for downloaded boundaries)
                obj = obj.set_crs(target)
                return obj
            if obj.crs.to_string() != target:
                return obj.to_crs(target)
            return obj
    except Exception:
        pass

    # --- Raster/Datacube: xarray + rioxarray
    try:
        import xarray as xr
        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            # requires rioxarray accessor
            import rioxarray  # noqa: F401
            if obj.rio.crs is None:
                return obj.rio.write_crs(target)
            # 这里一般不做 reproject（lat/lon 规则网格，强行重投影会慢且容易踩坑）
            return obj
    except Exception:
        pass

    raise TypeError("ensure_crs() only supports GeoDataFrame/GeoSeries or xarray DataArray/Dataset.")


# -----------------------------
# Clip
# -----------------------------

def clip_to_aoi(
    da,
    aoi_gdf,
    all_touched: bool = True,
    drop: bool = True,
    method: str = "rio",
):
    """
    Clip an xarray DataArray (lat/lon grid) to a polygon AOI (GeoDataFrame).

    Notes
    -----
    - Designed for CAMS-like lat/lon gridded products: dims usually include (time, lat, lon)
    - Uses rioxarray for polygon masking.
    - IMPORTANT: rioxarray/rasterio expects y axis to be north->south (descending lat).
      Many datasets store lat ascending (south->north). We fix that here.

    Parameters
    ----------
    da : xarray.DataArray
        Air quality datacube with dims including lat/lon.
    aoi_gdf : geopandas.GeoDataFrame
        AOI boundary polygon(s).
    all_touched : bool
        If True, include all pixels touched by geometry edges.
    drop : bool
        If True, drop data outside AOI.
    method : str
        Currently supports: "rio" (default). "auto" is accepted and treated as "rio".

    Returns
    -------
    xarray.DataArray
        Clipped DataArray.
    """
    method = (method or "rio").lower()
    if method not in {"rio", "auto"}:
        raise ValueError("method must be 'rio' or 'auto'.")

    # Lazy imports so the library can import without optional deps
    try:
        import rioxarray  # noqa: F401
    except ImportError as e:
        raise ImportError("rioxarray is required for clip_to_aoi().") from e

    # --- Ensure both have CRS (your unified helper)
    da = ensure_crs(da)
    aoi_gdf = ensure_crs(aoi_gdf)

    # --- Ensure the DataArray has lon/lat dims
    if "lon" not in da.dims or "lat" not in da.dims:
        raise ValueError(
            f"clip_to_aoi expects DataArray dims to include ('lat','lon'), got {da.dims}"
        )

    # --- Tell rioxarray which dims are spatial
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    # --- CRITICAL FIX: rasterio expects y (lat) descending (north -> south)
    # Many products have lat ascending (south -> north). Flip if needed.
    try:
        if da["lat"].size > 1 and (da["lat"][0].item() < da["lat"][-1].item()):
            da = da.sortby("lat", ascending=False)
    except Exception:
        # If lat can't be compared cleanly, just proceed
        pass

    # --- Clip
    clipped = da.rio.clip(
        aoi_gdf.geometry,
        aoi_gdf.crs,
        drop=drop,
        all_touched=all_touched,
    )

    return clipped
    
# -----------------------------
# Temporal aggregation
# -----------------------------
def monthly_mean(
    da,
    *,
    month: str | None = None,     # e.g. "2013-01"; None=对现有所有时间步求均值（适合单月文件）
    time_dim: str = "time",
):
    """
    Compute monthly mean map (2D lat/lon) from a datacube.

    Parameters
    ----------
    da : xarray.DataArray
        Expected dims include (time, lat, lon).
        Can be full-year or single-month file.
    month : str | None
        If provided, selects that month within the time series (e.g. "2013-01") then averages over time.
        If None, simply averages over all available time steps (useful when file already contains one month).
    time_dim : str
        Time dimension name (default "time").

    Returns
    -------
    xarray.DataArray
        2D DataArray (lat, lon) monthly mean.
    """
    import pandas as pd

    if time_dim not in da.dims:
        raise ValueError(f"time_dim='{time_dim}' not in da.dims={da.dims}")

    if month is not None:
        start = pd.Period(month, freq="M").start_time
        end = pd.Period(month, freq="M").end_time
        da = da.sel({time_dim: slice(start, end)})

        if da.sizes.get(time_dim, 0) == 0:
            raise ValueError(f"No data found for month={month}. Check time_dim='{time_dim}'.")

    return da.mean(dim=time_dim, skipna=True)


def monthly_map_and_stats(
    nc_path: str,
    var: str,
    aoi_query: str = "Italy",
    *,
    aoi_by: str = "NAME",
    month: str | None = None,          # e.g. "2013-01"
    time_dim: str = "time",
    threshold: float = 25.0,
    all_touched: bool = True,
    drop: bool = True,
    plot: bool = True,
    title: str | None = None,
):
    """
    One-click workflow:
    load AQ -> ensure CRS -> load AOI -> clip -> monthly mean -> stats -> optional plot

    Returns
    -------
    (da_month, stats)
        da_month: 2D monthly mean map
        stats: dict(min/max/mean/exceed_ratio)
    """
    import numpy as np
    import pandas as pd

    try:
        import xarray as xr  # noqa: F401
    except ImportError:
        raise ImportError("xarray is required.")
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    # 1) load data
    da = load_air_quality(nc_path, var=var)
    da = ensure_crs(da)

    # 2) AOI
    aoi = get_admin_boundary(aoi_query, by=aoi_by)
    aoi = ensure_crs(aoi)

    # 3) clip
    da_clip = clip_to_aoi(da, aoi, all_touched=all_touched, drop=drop)

    # 4) determine month
    if month is None:
        t0 = da_clip[time_dim].values[0]
        month = pd.Timestamp(t0).strftime("%Y-%m")  # "2013-01"

    # monthly mean map (use reusable helper)
    da_month = monthly_mean(da_clip, month=month, time_dim=time_dim)

    # 5) stats over space
    spatial_dims = [d for d in da_month.dims if d in ("lat", "lon", "x", "y")]
    if not spatial_dims:
        raise ValueError(f"Cannot find spatial dims in {da_month.dims}. Expected lat/lon or x/y.")

    stats = {
        "min": float(da_month.min(dim=spatial_dims, skipna=True).values),
        "max": float(da_month.max(dim=spatial_dims, skipna=True).values),
        "mean": float(da_month.mean(dim=spatial_dims, skipna=True).values),
    }

    if threshold is not None:
        mask = da_month > threshold
        stats[f"exceed_ratio_{threshold:g}"] = float(mask.mean(dim=spatial_dims, skipna=True).values)

    # 6) plot
    if plot:
        import matplotlib.pyplot as plt
        units = da.attrs.get("units", "")
        if title is None:
            title = f"{aoi_query} {var.upper()} monthly mean ({month}, {units})".strip()

        da_month.plot()
        plt.title(title)
        plt.show()

    return da_month, stats

# -----------------------------
# 5️⃣ 叠加土地利用图（categorical raster）
# -----------------------------

def load_landcover(
    path: str,
    var: str | None = None,
    chunks: dict | None = None,
):
    """
    Load land cover categorical raster (NetCDF) as xarray.DataArray.

    Parameters
    ----------
    path : str
        NetCDF path.
    var : str | None
        Variable name. If None, pick the first ds.data_vars.
    chunks : dict | None
        Dask chunks for lazy loading.

    Returns
    -------
    xarray.DataArray
        Land cover DataArray (categorical).
    """
    import xarray as xr

    ds = xr.open_dataset(path, chunks=chunks)
    if var is None:
        var = list(ds.data_vars)[0]
    da = ds[var]

    # normalize coords (common variants)
    rename = {}
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if rename:
        da = da.rename(rename)

    return da


def align_categorical_to_reference(lc_da, ref_da, method: str = "nearest"):
    """
    Align categorical landcover to the grid of reference AQ map using xarray interp.

    IMPORTANT: for categorical, use nearest (default).

    Parameters
    ----------
    lc_da : xarray.DataArray
        Categorical landcover raster.
    ref_da : xarray.DataArray
        Reference AQ 2D map with lon/lat.
    method : str
        Interpolation method: 'nearest' recommended for categorical.

    Returns
    -------
    xarray.DataArray
        Landcover resampled onto ref grid.
    """
    # sanity checks
    for need in ("lon", "lat"):
        if need not in lc_da.coords and need not in lc_da.dims:
            raise ValueError(f"Landcover must have '{need}' in coords/dims, got dims={lc_da.dims}")
        if need not in ref_da.coords and need not in ref_da.dims:
            raise ValueError(f"Reference must have '{need}' in coords/dims, got dims={ref_da.dims}")

    # xarray interp generally expects monotonic coords; handle descending lat
    ref_lat = ref_da["lat"].values
    if np.any(np.diff(ref_lat) < 0):
        ref_lat_for_interp = ref_lat[::-1]
        flip_back = True
    else:
        ref_lat_for_interp = ref_lat
        flip_back = False

    lc_on_ref = lc_da.interp(
        lon=ref_da["lon"].values,
        lat=ref_lat_for_interp,
        method=method,
    )

    if flip_back:
        lc_on_ref = lc_on_ref.sel(lat=ref_lat)

    return lc_on_ref


# 22→6 的默认映射（你们后续可以按产品文档精修）
DEFAULT_LCCS_TO_6CLASS: dict[int, int] = {
    # 1 Agriculture / Cropland
    10: 1, 11: 1, 12: 1, 20: 1, 30: 1, 40: 1,
    # 2 Forest
    50: 2, 60: 2, 61: 2, 62: 2, 70: 2, 71: 2, 72: 2, 80: 2, 81: 2, 82: 2, 90: 2, 100: 2, 160: 2, 170: 2,
    # 3 Grassland / Herbaceous
    110: 3, 130: 3,
    # 4 Wetland
    180: 4,
    # 5 Settlement / Urban
    190: 5,
    # 6 Other (shrub/sparse/bare/water/snow-ice etc.)
    120: 6, 121: 6, 122: 6, 140: 6, 150: 6, 151: 6, 152: 6, 153: 6,
    200: 6, 201: 6, 202: 6, 210: 6, 220: 6,
}


def reclass_landcover(
    lc_da,
    mapping: dict[int, int] | None = None,
    nodata_out: int = 0,
    name: str = "landcover_6",
):
    """
    Reclass landcover codes into fewer macro classes (default 6 classes).

    Parameters
    ----------
    lc_da : xarray.DataArray
        Categorical landcover.
    mapping : dict[int,int] | None
        Old->new code mapping.
    nodata_out : int
        Code for unmapped values.
    name : str
        Output variable name.

    Returns
    -------
    xarray.DataArray
        Reclassified categorical raster (int16).
    """
    import xarray as xr

    if mapping is None:
        mapping = DEFAULT_LCCS_TO_6CLASS

    data = lc_da.data
    out = np.full_like(data, fill_value=nodata_out, dtype=np.int16)

    # vectorized remap (loop over classes; OK for <= few hundred)
    for k, v in mapping.items():
        out = np.where(data == k, v, out)

    out_da = xr.DataArray(
        out,
        coords=lc_da.coords,
        dims=lc_da.dims,
        name=name,
        attrs={"scheme": "6class", "nodata": nodata_out},
    )
    return out_da


def overlay_aq_with_landcover(
    aq_map,
    lc_da,
    *,
    reclass: bool = True,
    mapping: dict[int, int] | None = None,
    method: str = "nearest",
    mask_by_aq: bool = True,
    force_reproject_match: bool = False,
):
    """
    Raster × raster overlay with CRS handling:
    - ensure CRS exists (assume EPSG:4326 if missing via ensure_crs)
    - if CRS differs OR force_reproject_match=True: use rioxarray.reproject_match (nearest for categorical)
    - else: use xarray interp to align lon/lat (fast for lat/lon grids)
    - optionally reclass landcover to fewer classes
    - optionally mask landcover where AQ is NaN

    Parameters
    ----------
    aq_map : xarray.DataArray
        2D AQ map (lat, lon), e.g. output of monthly_mean().
    lc_da : xarray.DataArray
        Landcover raster (categorical).
    reclass : bool
        Reclass to 6 classes.
    mapping : dict[int,int] | None
        Old->new mapping.
    method : str
        For categorical alignment: 'nearest' recommended.
    mask_by_aq : bool
        If True, set landcover to NaN where AQ is NaN.
    force_reproject_match : bool
        If True, always use reproject_match (slower but robust).

    Returns
    -------
    (aq_map, lc_on_aq)
    """
    import numpy as np

    # 1) Ensure CRS exists on both (your helper should attach EPSG:4326 if missing)
    aq_map = ensure_crs(aq_map)
    lc_da = ensure_crs(lc_da)

    # 2) Ensure spatial dims set for rioxarray (important if we need reproject_match)
    if "lon" in aq_map.dims and "lat" in aq_map.dims:
        aq_map = aq_map.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if "lon" in lc_da.dims and "lat" in lc_da.dims:
        lc_da = lc_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    aq_crs = getattr(aq_map.rio, "crs", None)
    lc_crs = getattr(lc_da.rio, "crs", None)

    # 3) Align grid
    use_match = force_reproject_match or (aq_crs is not None and lc_crs is not None and aq_crs != lc_crs)

    if use_match:
        # robust path: match AQ grid exactly (categorical => nearest)
        lc_on_aq = lc_da.rio.reproject_match(aq_map, resampling=0)  # 0 = nearest
    else:
        # fast path: assume both are lon/lat; align by interp
        lc_on_aq = align_categorical_to_reference(lc_da, aq_map, method=method)

    # 4) Reclass
    if reclass:
        lc_on_aq = reclass_landcover(lc_on_aq, mapping=mapping)

    # 5) Mask
    if mask_by_aq:
        lc_on_aq = lc_on_aq.where(np.isfinite(aq_map))

    return aq_map, lc_on_aq
    
# -----------------------------
# Zonal statistics / change
# -----------------------------

def landcover_stats(
    aq2d,
    lc6,
    *,
    threshold: float = 25.0,
    include_max: bool = True,
    include_median: bool = True,
    class_labels: dict[int, str] | None = None,
):
    """
    Compute per-landcover stats for a 2D air quality map.

    Parameters
    ----------
    aq2d : xarray.DataArray
        2D pollutant map (lat, lon), NaN outside AOI.
    lc6 : xarray.DataArray
        Landcover classes on the same grid as aq2d.
        Can be (lat, lon) or (time, lat, lon) with time=1.
        Values expected in {0,1,2,3,4,5,6}; 0 = nodata/unmapped.
    threshold : float
        Exceedance threshold.
    include_max / include_median : bool
        Whether to compute extra stats.
    class_labels : dict[int,str] | None
        Optional mapping for pretty names.

    Returns
    -------
    pandas.DataFrame
        columns: class, label, pixel_ratio, n_pixels, mean, exceed_ratio, (max), (median)
    """
    import numpy as np
    import pandas as pd

    # squeeze landcover to (lat,lon)
    if "time" in lc6.dims:
        lc6_2d = lc6.isel(time=0)
    else:
        lc6_2d = lc6

    # mask: only where AQ is valid
    valid = np.isfinite(aq2d.values)
    aq_vals = aq2d.values[valid]
    lc_vals = lc6_2d.values[valid]

    # drop class 0 (nodata)
    keep = (lc_vals != 0) & np.isfinite(lc_vals)
    aq_vals = aq_vals[keep]
    lc_vals = lc_vals[keep].astype(int)

    total = lc_vals.size
    rows = []
    for c in sorted(np.unique(lc_vals)):
        m = (lc_vals == c)
        v = aq_vals[m]
        if v.size == 0:
            continue

        row = {
            "class": int(c),
            "label": class_labels.get(int(c), str(int(c))) if class_labels else str(int(c)),
            "n_pixels": int(v.size),
            "pixel_ratio": float(v.size / total) if total > 0 else np.nan,
            "mean": float(np.nanmean(v)),
            "exceed_ratio": float(np.nanmean(v > threshold)) if threshold is not None else np.nan,
        }
        if include_max:
            row["max"] = float(np.nanmax(v))
        if include_median:
            row["median"] = float(np.nanmedian(v))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    return df

def plot_landcover_stats(
    df,
    *,
    title: str = "NO₂ by land cover",
    ylabel_left: str = "Mean NO₂ (µg/m³)",
    ylabel_right: str = "Exceed ratio",
    savepath: str | None = None,
):
    """
    Plot mean and exceed_ratio by landcover class (dual-axis).

    Parameters
    ----------
    df : pandas.DataFrame
        output of landcover_stats()
    savepath : str | None
        if provided, save PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(df))
    labels = df["label"].tolist()
    means = df["mean"].values
    ex = df["exceed_ratio"].values

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(x, means)
    ax1.set_ylabel(ylabel_left)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, ex, marker="o")
    ax2.set_ylabel(ylabel_right)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

def export_landcover_tif(
    lc6_on_aq,
    out_tif: str,
    *,
    nodata: int = 0,
):
    """
    Export reclassified landcover (on AQ grid) to GeoTIFF.

    lc6_on_aq should already be aligned to AQ grid (lat/lon) and have CRS via ensure_crs().
    """
    import rioxarray  # noqa: F401

    da = lc6_on_aq
    if "time" in da.dims:
        da = da.isel(time=0)

    da = ensure_crs(da)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    da = da.rio.write_nodata(nodata, inplace=False)
    da.rio.to_raster(out_tif)
    return out_tif

def plot_monthly_map_with_landcover_overlay(
    aq2d,
    lc6_on_aq,
    *,
    title: str = "Monthly mean (AOI)",
    landcover_alpha: float = 0.25,
    show_landcover: bool = True,
    savepath: str | None = None,
):
    """
    Plot AQ monthly mean map. Optionally overlay landcover classes (semi-transparent).

    Parameters
    ----------
    aq2d : xarray.DataArray (lat,lon)
    lc6_on_aq : xarray.DataArray (lat,lon) or (time,lat,lon)
    """
    import matplotlib.pyplot as plt

    if "time" in lc6_on_aq.dims:
        lc2d = lc6_on_aq.isel(time=0)
    else:
        lc2d = lc6_on_aq

    fig, ax = plt.subplots(figsize=(8, 5))
    aq2d.plot(ax=ax)
    ax.set_title(title)

    if show_landcover:
        # overlay as another image layer
        lc2d.plot(ax=ax, alpha=landcover_alpha, add_colorbar=False)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

def monthly_landcover_report(
    *,
    aq_nc_path: str,
    aq_var: str,
    lc_nc_path: str,
    country: str = "Italy",
    month: str | None = None,
    threshold: float = 25.0,
    outdir: str = "outputs",
    landcover_var: str | None = None,
    plot: bool = True,
):
    """
    Produce deliverables:
    2) landcover reclassified tif
    3) monthly AQ map + overlay fig
    4) landcover stats plot (mean + exceed)
    5) summary table CSV

    Returns dict of paths + objects.
    """
    import os
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)

    # --- 1) AQ load -> AOI -> clip -> monthly mean
    da = load_air_quality(aq_nc_path, var=aq_var)
    da = ensure_crs(da)
    aoi = ensure_crs(get_admin_boundary(country, by="NAME"))
    da_clip = clip_to_aoi(da, aoi, all_touched=True, drop=True)
    aq2d = monthly_mean(da_clip, month=month, time_dim="time")  # 2D

    # --- 2) Landcover load -> (建议先 bbox clip 再 overlay，省内存)
    lc = load_landcover(lc_nc_path, var=landcover_var)
    lc = ensure_crs(lc)

    # bbox clip landcover to aq extent (加速、避免卡)
    lat_min = float(aq2d["lat"].min().values)
    lat_max = float(aq2d["lat"].max().values)
    lon_min = float(aq2d["lon"].min().values)
    lon_max = float(aq2d["lon"].max().values)

    # 注意 lat 可能是降序，所以 slice 要兼容
    lat0 = float(lc["lat"][0].values)
    lat1 = float(lc["lat"][-1].values)
    if lat0 > lat1:  # descending
        lc_small = lc.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:            # ascending
        lc_small = lc.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # --- 3) overlay + reclass to 6
    aq2d, lc6 = overlay_aq_with_landcover(
        aq2d, lc_small,
        reclass=True,
        mask_by_aq=True,
        force_reproject_match=False
    )

    # --- 2) export reclassified tif
    tif_path = os.path.join(outdir, f"{country}_landcover6_on_aq_{month or 'auto'}.tif")
    export_landcover_tif(lc6, tif_path, nodata=0)

    # --- 5) summary table
    df = landcover_stats(
        aq2d, lc6,
        threshold=threshold,
        include_max=True,
        include_median=True,
        class_labels={
            1: "Agriculture",
            2: "Forest",
            3: "Grassland",
            4: "Wetland",
            5: "Settlement",
            6: "Other",
        },
    )
    csv_path = os.path.join(outdir, f"{country}_{aq_var}_{month or 'auto'}_landcover_stats.csv")
    df.to_csv(csv_path, index=False)

    # --- 3) map + overlay fig
    map_png = os.path.join(outdir, f"{country}_{aq_var}_{month or 'auto'}_monthly_map_overlay.png")
    if plot:
        plot_monthly_map_with_landcover_overlay(
            aq2d, lc6,
            title=f"{country} {aq_var.upper()} monthly mean ({month or 'auto'})",
            landcover_alpha=0.25,
            show_landcover=True,
            savepath=map_png,
        )

    # --- 4) stats plot
    stats_png = os.path.join(outdir, f"{country}_{aq_var}_{month or 'auto'}_landcover_stats.png")
    if plot:
        plot_landcover_stats(
            df,
            title=f"{country} {aq_var.upper()} by land cover ({month or 'auto'})",
            ylabel_left=f"{aq_var.upper()} (µg/m³)",
            savepath=stats_png,
        )

    return {
        "aq2d": aq2d,
        "lc6": lc6,
        "df": df,
        "tif_landcover6": tif_path,
        "csv_summary": csv_path,
        "png_map_overlay": map_png,
        "png_stats": stats_png,
    }

# -----------------------------
# Sanity check
# -----------------------------

def hello():
    """Simple test function."""
    return "Hello, AirQualityLib"
