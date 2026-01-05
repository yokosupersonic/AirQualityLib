from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Optional deps: keep import light so `import airqualitylib` doesn't explode
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


# =========================================================
# 1) I/O
# =========================================================

def load_air_quality(path: str, var: Optional[str] = None, chunks: Optional[dict] = None):
    """Load air quality data from NetCDF as xarray.DataArray (lazy)."""
    if xr is None:
        raise ImportError("xarray is required for load_air_quality().")

    ds = xr.open_dataset(path, chunks=chunks)

    data_vars = list(ds.data_vars)
    if var is None:
        if len(data_vars) == 1:
            var = data_vars[0]
        else:
            raise ValueError(f"Multiple variables found in NetCDF: {data_vars}. Please specify var=...")

    if var not in ds.data_vars:
        raise KeyError(f"Variable '{var}' not found. Available variables: {data_vars}")

    return ds[var]


def load_admin_boundaries(path: Optional[str] = None, layer: str = "boundaries"):
    """Load built-in Europe administrative boundaries (GeoPackage) or a user path."""
    if gpd is None:
        raise ImportError("geopandas is required for load_admin_boundaries().")

    if path is None:
        path = str(
            Path(__file__).resolve().parents[1]
            / "data" / "boundaries" / "Europe_administrative_boundaries.gpkg"
        )

    gdf = gpd.read_file(path, layer=layer)

    # Ensure WGS84 for consistent clipping with lat/lon grids
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def get_admin_boundary(query: str, by: str = "NAME", path: Optional[str] = None, layer: str = "boundaries"):
    """Get one boundary feature (e.g., Italy) from built-in boundaries."""
    gdf = load_admin_boundaries(path=path, layer=layer)

    if by not in gdf.columns:
        raise ValueError(f"Column '{by}' not found. Available: {list(gdf.columns)}")

    out = gdf[gdf[by].astype(str) == str(query)]
    if out.empty:
        examples = gdf[by].dropna().astype(str).unique()[:20]
        raise ValueError(f"No match for {query!r} in column '{by}'. Example values: {list(examples)}")

    return out


# =========================================================
# 2) CRS utils
# =========================================================

def ensure_crs(obj, epsg: int = 4326):
    """Attach/ensure CRS on GeoPandas or xarray (rioxarray) objects."""
    target = f"EPSG:{epsg}"

    # Vector: GeoPandas
    if gpd is not None and isinstance(obj, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if obj.crs is None:
            return obj.set_crs(target)
        if obj.crs.to_string() != target:
            return obj.to_crs(target)
        return obj

    # Raster: xarray + rioxarray
    if xr is not None and isinstance(obj, (xr.DataArray, xr.Dataset)):
        try:
            import rioxarray  # noqa: F401
        except Exception as e:
            raise ImportError("rioxarray is required for ensure_crs() on xarray objects.") from e

        if obj.rio.crs is None:
            return obj.rio.write_crs(target)
        return obj

    raise TypeError("ensure_crs() only supports GeoDataFrame/GeoSeries or xarray DataArray/Dataset.")


# =========================================================
# 3) Clip
# =========================================================

def clip_to_aoi(da, aoi_gdf, all_touched: bool = True, drop: bool = True):
    """Clip xarray.DataArray (lat/lon) to an AOI polygon (GeoDataFrame) using rioxarray."""
    try:
        import rioxarray  # noqa: F401
    except ImportError as e:
        raise ImportError("rioxarray is required for clip_to_aoi().") from e

    da = ensure_crs(da)
    aoi_gdf = ensure_crs(aoi_gdf)

    if "lon" not in da.dims or "lat" not in da.dims:
        raise ValueError(f"clip_to_aoi expects dims include ('lat','lon'), got {da.dims}")

    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    # rasterio expects y descending; flip if needed
    try:
        if da["lat"].size > 1 and (da["lat"][0].item() < da["lat"][-1].item()):
            da = da.sortby("lat", ascending=False)
    except Exception:
        pass

    return da.rio.clip(aoi_gdf.geometry, aoi_gdf.crs, drop=drop, all_touched=all_touched)


# =========================================================
# 4) Temporal aggregation
# =========================================================

def monthly_mean(da, *, month: str | None = None, time_dim: str = "time"):
    """Compute monthly mean 2D map (lat, lon) from a time cube."""
    if pd is None:
        raise ImportError("pandas is required for monthly_mean().")

    if time_dim not in da.dims:
        raise ValueError(f"time_dim='{time_dim}' not in da.dims={da.dims}")

    if month is not None:
        start = pd.Period(month, freq="M").start_time
        end = pd.Period(month, freq="M").end_time
        da = da.sel({time_dim: slice(start, end)})
        if da.sizes.get(time_dim, 0) == 0:
            raise ValueError(f"No data found for month={month}. Check time_dim='{time_dim}'.")

    return da.mean(dim=time_dim, skipna=True)


# =========================================================
# 5) Landcover: load / align / reclass / overlay
# =========================================================

def load_landcover(path: str, var: str | None = None, chunks: dict | None = None):
    """Load landcover NetCDF as xarray.DataArray, normalize to dims (lat, lon)."""
    if xr is None:
        raise ImportError("xarray is required for load_landcover().")

    ds = xr.open_dataset(path, chunks=chunks)
    if var is None:
        var = list(ds.data_vars)[0]
    da = ds[var]

    rename = {}
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if rename:
        da = da.rename(rename)

    # squeeze common extra dims
    for extra in ("time", "band"):
        if extra in da.dims and da.sizes.get(extra, 0) > 0:
            da = da.isel({extra: 0})
    return da.squeeze(drop=True)


def align_categorical_to_reference(lc_da, ref_da, method: str = "nearest"):
    """Align categorical landcover onto ref (AQ) grid using xarray interp (nearest)."""
    for need in ("lon", "lat"):
        if need not in lc_da.coords and need not in lc_da.dims:
            raise ValueError(f"Landcover must have '{need}', got dims={lc_da.dims}")
        if need not in ref_da.coords and need not in ref_da.dims:
            raise ValueError(f"Reference must have '{need}', got dims={ref_da.dims}")

    ref_lat = ref_da["lat"].values
    if np.any(np.diff(ref_lat) < 0):
        ref_lat_for_interp = ref_lat[::-1]
        flip_back = True
    else:
        ref_lat_for_interp = ref_lat
        flip_back = False

    lc_on_ref = lc_da.interp(lon=ref_da["lon"].values, lat=ref_lat_for_interp, method=method)
    if flip_back:
        lc_on_ref = lc_on_ref.sel(lat=ref_lat)

    return lc_on_ref


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
    # 6 Other
    120: 6, 121: 6, 122: 6, 140: 6, 150: 6, 151: 6, 152: 6, 153: 6,
    200: 6, 201: 6, 202: 6, 210: 6, 220: 6,
}
# -----------------------------
# Landcover 6-class definition
# -----------------------------

LANDCOVER_6_LABELS: dict[int, str] = {
    1: "Agriculture",
    2: "Forest",
    3: "Grassland",
    4: "Wetland",
    5: "Settlement",
    6: "Other",
}

LANDCOVER_6_COLORS: dict[int, str] = {
    1: "#E6E2AF",  # Agriculture - warm light beige
    2: "#3B7A57",  # Forest - deep green
    3: "#A6D854",  # Grassland - fresh light green
    4: "#66C2A5",  # Wetland - teal
    5: "#B2182B",  # Settlement - dark red
    6: "#9E9E9E",  # Other - neutral gray
}

# -----------------------------
# Unified plotting style (ALL PNG SAME SIZE)
# -----------------------------
PLOT_FIGSIZE: tuple[float, float] = (11.5, 5.2)  # 统一大小
PLOT_DPI: int = 200                              # 统一清晰度


def reclass_landcover(lc_da, mapping: dict[int, int] | None = None, nodata_out: int = 0, name: str = "landcover_6"):
    """Reclass landcover codes into 6 macro classes using DEFAULT_LCCS_TO_6CLASS."""
    if xr is None:
        raise ImportError("xarray is required for reclass_landcover().")

    if mapping is None:
        mapping = DEFAULT_LCCS_TO_6CLASS

    data = lc_da.data
    out = np.full_like(data, fill_value=nodata_out, dtype=np.int16)
    for k, v in mapping.items():
        out = np.where(data == k, v, out)

    return xr.DataArray(
        out,
        coords=lc_da.coords,
        dims=lc_da.dims,
        name=name,
        attrs={"scheme": "6class", "nodata": nodata_out},
    )


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
    """Align landcover to AQ grid, optional reclass to 6 classes, optional mask by AQ NaNs."""
    try:
        import rioxarray  # noqa: F401
    except ImportError as e:
        raise ImportError("rioxarray is required for overlay_aq_with_landcover().") from e

    aq_map = ensure_crs(aq_map)
    lc_da = ensure_crs(lc_da)

    if "lon" in aq_map.dims and "lat" in aq_map.dims:
        aq_map = aq_map.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if "lon" in lc_da.dims and "lat" in lc_da.dims:
        lc_da = lc_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    aq_crs = getattr(aq_map.rio, "crs", None)
    lc_crs = getattr(lc_da.rio, "crs", None)

    use_match = force_reproject_match or (aq_crs is not None and lc_crs is not None and aq_crs != lc_crs)

    if use_match:
        lc_on_aq = lc_da.rio.reproject_match(aq_map, resampling=0)  # nearest
    else:
        lc_on_aq = align_categorical_to_reference(lc_da, aq_map, method=method)

    if reclass:
        lc_on_aq = reclass_landcover(lc_on_aq, mapping=mapping)

    if mask_by_aq:
        lc_on_aq = lc_on_aq.where(np.isfinite(aq_map))

    return aq_map, lc_on_aq


# =========================================================
# 6) Stats
# =========================================================

def landcover_stats(
    aq2d,
    lc6,
    *,
    threshold: float = 40.0,
    include_max: bool = True,
    include_median: bool = True,
    class_labels: dict[int, str] | None = None,
):

    """Compute per-landcover stats for a 2D AQ map."""
    if pd is None:
        raise ImportError("pandas is required for landcover_stats().")



    if class_labels is None:
        class_labels = LANDCOVER_6_LABELS


    lc6_2d = lc6.isel(time=0) if "time" in lc6.dims else lc6

    valid = np.isfinite(aq2d.values)
    aq_vals = aq2d.values[valid]
    lc_vals = lc6_2d.values[valid]

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

    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)

def get_landcover_6_colormap():
    """
    Return (cmap, norm, labels) for 6-class landcover plotting.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    labels = [LANDCOVER_6_LABELS[i] for i in range(1, 7)]
    colors = [LANDCOVER_6_COLORS[i] for i in range(1, 7)]

    cmap = ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm, labels

def plot_monthly_mean_map(
    aq2d,
    *,
    pollutant_name: str = "Pollutant",
    units: str = "ug/m3",
    title: str | None = None,
    cmap_aq: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot monthly mean pollutant map (NO2/PM2.5/PM10...) in unified style & size.
    Output PNG size is unified by default via PLOT_FIGSIZE/PLOT_DPI.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if figsize is None:
        figsize = PLOT_FIGSIZE
    if dpi is None:
        dpi = PLOT_DPI

    if title is None:
        title = f"{pollutant_name} monthly mean"

    lon = aq2d["lon"].values
    lat = aq2d["lat"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    origin = "upper" if (lat.size > 1 and lat[0] > lat[-1]) else "lower"

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        aq2d.values,
        origin=origin,
        extent=extent,
        cmap=cmap_aq,
        vmin=vmin,
        vmax=vmax,
        alpha=0.95
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"{pollutant_name} ({units})")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_landcover_6_map(
    lc6_on_aq,
    *,
    title: str = "Landcover (6 classes)",
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot reclassified landcover (6 classes) using the library's unified colors/labels.

    This plot intentionally uses a near-square figure size,
    independent from global AQ plot settings.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plotting.") from e

    figsize = (6.5, 6.5)
    dpi = PLOT_DPI

    # squeeze to 2D
    lc2d = lc6_on_aq.isel(time=0) if "time" in lc6_on_aq.dims else lc6_on_aq

    cmap, norm, labels = get_landcover_6_colormap()

    fig, ax = plt.subplots(figsize=figsize)

    im = lc2d.plot(
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    )

    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5, 6])
    cbar.ax.set_yticklabels(labels)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 7) Export & Visualization
# =========================================================

def export_landcover_tif(lc6_on_aq, out_tif: str, *, nodata: int = 0):
    """Export reclassified landcover (already on AQ grid) to GeoTIFF."""
    try:
        import rioxarray  # noqa: F401
    except ImportError as e:
        raise ImportError("rioxarray is required for export_landcover_tif().") from e

    da = lc6_on_aq.isel(time=0) if "time" in lc6_on_aq.dims else lc6_on_aq
    da = ensure_crs(da)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    da = da.rio.write_nodata(nodata, inplace=False)
    da.rio.to_raster(out_tif)
    return out_tif

def plot_monthly_map_with_landcover_overlay(
    aq2d,
    lc6_on_aq,
    *,
    pollutant_name: str = "Pollutant",
    units: str = "ug/m3",
    title: str | None = None,
    landcover_alpha: float = 0.95,
    linewidth: float = 0.4,
    cmap_aq: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot AQ monthly mean map as background, and draw landcover class boundaries as contour lines.
    Legend is placed outside (right side) to avoid overlap.
    Output PNG size is unified by default via PLOT_FIGSIZE/PLOT_DPI.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if figsize is None:
        figsize = PLOT_FIGSIZE
    if dpi is None:
        dpi = PLOT_DPI

    lc_plot = lc6_on_aq.isel(time=0) if "time" in lc6_on_aq.dims else lc6_on_aq
    aq_plot = aq2d

    lon = aq_plot["lon"].values
    lat = aq_plot["lat"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    origin = "upper" if (lat.size > 1 and lat[0] > lat[-1]) else "lower"

    cmap_lc, norm_lc, labels_lc = get_landcover_6_colormap()

    if title is None:
        title = f"Monthly Mean {pollutant_name} with Land Cover Overlay"

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.78)

    im = ax.imshow(
        aq_plot.values,
        origin=origin,
        extent=extent,
        cmap=cmap_aq,
        vmin=vmin,
        vmax=vmax,
        alpha=0.9
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"{pollutant_name} ({units})")

    for cls_id in range(1, 7):
        mask = (lc_plot.values == cls_id).astype(float)
        ax.contour(
            mask,
            levels=[0.5],
            origin=origin,
            extent=extent,
            linewidths=linewidth,
            colors=[cmap_lc(norm_lc(cls_id))],
            alpha=landcover_alpha
        )

    patches = [
        mpatches.Patch(color=cmap_lc(norm_lc(i)), label=lab)
        for i, lab in zip(range(1, 7), labels_lc)
    ]

    fig.legend(
        handles=patches,
        title="Land cover (6 classes)",
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=True
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_landcover_stats(
    df,
    *,
    title: str = "Pollutant by land cover",
    ylabel_left: str = "Mean (ug/m3)",
    ylabel_right: str = "Exceed ratio",
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    savepath: str | None = None,
    show: bool = True,
):
    """Dual-axis plot for landcover mean and exceed ratio. Output size unified by default."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if figsize is None:
        figsize = PLOT_FIGSIZE
    if dpi is None:
        dpi = PLOT_DPI

    x = np.arange(len(df))
    labels = df["label"].tolist()
    means = df["mean"].values
    ex = df["exceed_ratio"].values

    # bar colors follow landcover class colors
    bar_colors = [LANDCOVER_6_COLORS.get(int(c), "#CCCCCC") for c in df["class"].values]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(x, means, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(ylabel_left)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, ex, marker="o", linewidth=2)
    ax2.set_ylabel(ylabel_right)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# Sanity
# =========================================================

def hello():
    return "Hello, AirQualityLib"
