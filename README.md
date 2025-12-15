# AirQuality-Lib

A lightweight Python geospatial library for air quality analysis (PM2.5 / PM10 / NO2).

## What this library will do
- Load air quality raster/datacube (user-provided)
- Use built-in administrative boundaries (optional)
- Clip to a country/region
- Temporal aggregation: annual / seasonal
- Indicators: min / max / mean / exceedance ratio
- Zonal statistics per administrative unit
- Period comparison (change/anomaly)

## Quickstart (skeleton)
```python
import geospatial_airquality as aq
aq.hello()
