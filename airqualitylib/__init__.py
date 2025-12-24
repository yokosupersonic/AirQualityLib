"""
AirQualityLib public API.

This package exposes a small, stable set of functions for:
- loading air quality data
- loading boundaries
- clipping and monthly aggregation
- landcover overlay/reclass
- landcover statistics and plots
"""

from .geospatial_airquality import (
    # I/O
    load_air_quality,
    load_admin_boundaries,
    get_admin_boundary,

    # CRS / clip / aggregation
    ensure_crs,
    clip_to_aoi,
    monthly_mean,

    # Landcover
    load_landcover,
    align_categorical_to_reference,
    reclass_landcover,
    overlay_aq_with_landcover,

    # Landcover definitions (labels/colors) + helper
    DEFAULT_LCCS_TO_6CLASS,
    LANDCOVER_6_LABELS,
    LANDCOVER_6_COLORS,
    get_landcover_6_colormap,

    # Stats + plots + export
    landcover_stats,
    export_landcover_tif,
    plot_monthly_map_with_landcover_overlay,
    plot_landcover_stats,

    # Sanity
    hello,
)

__all__ = [
    "load_air_quality",
    "load_admin_boundaries",
    "get_admin_boundary",
    "ensure_crs",
    "clip_to_aoi",
    "monthly_mean",
    "load_landcover",
    "align_categorical_to_reference",
    "reclass_landcover",
    "overlay_aq_with_landcover",
    "DEFAULT_LCCS_TO_6CLASS",
    "LANDCOVER_6_LABELS",
    "LANDCOVER_6_COLORS",
    "get_landcover_6_colormap",
    "landcover_stats",
    "export_landcover_tif",
    "plot_monthly_map_with_landcover_overlay",
    "plot_landcover_stats",
    "hello",
]

__version__ = "0.1.0"
