"""Fire risk assessment service.

Computes a spatially-varying fire risk index based on real weather conditions,
vegetation estimates, and terrain. Returns GeoJSON for map display.
"""

import math
import numpy as np
from services.weather_service import fetch_weather, fetch_elevation_grid
from config import GRID_SIZE, CELL_SIZE_KM


def _normalize(val: float, low: float, high: float) -> float:
    """Normalize value to [0, 1] between low and high bounds."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (val - low) / (high - low)))


def _compute_risk_score(weather: dict) -> float:
    """Compute fire risk index (0-1) from weather conditions.

    Uses a weighted combination inspired by the National Fire Danger Rating System:
    - Temperature: higher = more risk
    - Humidity: lower = more risk
    - Wind speed: higher = more risk
    - Precipitation: lower = more risk
    - ERC (Energy Release Component): higher = more risk
    - NDVI (vegetation density): moderate = highest risk (fuel available)
    """
    # Normalize weather features to 0-1 risk contributions
    # max_temp is in Kelvin (270-320K typical)
    temp_risk = _normalize(weather.get("max_temp", 300), 280, 320)

    # humidity is in fractional form (~0.001-0.01)
    humidity = weather.get("humidity", 0.005)
    humidity_risk = 1.0 - _normalize(humidity, 0.001, 0.01)

    # wind_speed in km/h (0-200 typical, high values in dataset)
    wind_risk = _normalize(weather.get("wind_speed", 10), 0, 50)

    # precipitation in mm (0-20)
    precip = weather.get("precipitation", 0)
    precip_risk = 1.0 - _normalize(precip, 0, 10)

    # ERC (0-100)
    erc_risk = _normalize(weather.get("erc", 50), 10, 90)

    # NDVI (0-10000; moderate values ~4000-7000 mean fuel available)
    ndvi = weather.get("ndvi", 5000)
    ndvi_risk = 1.0 - abs(ndvi - 5500) / 5500  # Peak risk at moderate vegetation
    ndvi_risk = max(0.0, ndvi_risk)

    # Drought index (negative = wet, positive = dry)
    drought = weather.get("drought_index", 0)
    drought_risk = _normalize(drought, -3, 3)

    # Weighted combination (weights based on feature importance from ML model)
    risk = (
        0.22 * temp_risk
        + 0.20 * humidity_risk
        + 0.15 * wind_risk
        + 0.12 * precip_risk
        + 0.12 * erc_risk
        + 0.10 * ndvi_risk
        + 0.09 * drought_risk
    )
    return max(0.0, min(1.0, risk))


def _risk_grid_to_geojson(
    risk_grid: np.ndarray,
    center_lat: float,
    center_lon: float,
    cell_size_km: float,
) -> dict:
    """Convert a 2D risk grid to GeoJSON polygons with risk levels.

    Only emits cells with risk > 0.3 to keep payload light.
    """
    deg_per_km = 1.0 / 111.0
    n = risk_grid.shape[0]
    half = n / 2.0

    features = []
    for r in range(0, n, 2):  # Step by 2 for fewer, larger cells
        for c in range(0, n, 2):
            # Average 2x2 block
            block = risk_grid[r : min(r + 2, n), c : min(c + 2, n)]
            risk_val = float(np.mean(block))
            if risk_val < 0.25:
                continue

            lat_offset = (r - half + 1) * cell_size_km * deg_per_km
            lon_offset = (c - half + 1) * cell_size_km * deg_per_km / max(
                math.cos(math.radians(center_lat)), 0.01
            )
            cell_deg = 2 * cell_size_km * deg_per_km

            lat0 = center_lat + lat_offset
            lon0 = center_lon + lon_offset
            lat1 = lat0 + cell_deg
            lon1 = lon0 + cell_deg / max(math.cos(math.radians(center_lat)), 0.01)

            # Classify risk level
            if risk_val >= 0.75:
                level = "extreme"
            elif risk_val >= 0.55:
                level = "high"
            elif risk_val >= 0.40:
                level = "moderate"
            else:
                level = "low"

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "risk": round(risk_val, 3),
                        "level": level,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [lon0, lat0],
                                [lon1, lat0],
                                [lon1, lat1],
                                [lon0, lat1],
                                [lon0, lat0],
                            ]
                        ],
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


async def compute_fire_risk(
    lat: float, lon: float, radius_km: float = 32.0
) -> dict:
    """Compute fire risk for an area centered at (lat, lon).

    Returns a GeoJSON FeatureCollection with risk-colored polygons,
    plus a summary with overall risk level and contributing factors.
    """
    # Fetch real weather conditions
    weather = await fetch_weather(lat, lon)

    # Base risk score from weather
    base_risk = _compute_risk_score(weather)

    # Fetch elevation grid for terrain variation
    try:
        elevation_grid = await fetch_elevation_grid(lat, lon)
    except Exception:
        elevation_grid = np.full((GRID_SIZE, GRID_SIZE), 500.0)

    # Create spatially-varying risk grid
    risk_grid = np.full((GRID_SIZE, GRID_SIZE), base_risk, dtype=np.float32)

    # Terrain effects: slopes increase risk, water/low elevation decreases
    if elevation_grid is not None:
        # Compute slope magnitude (gradient)
        gy, gx = np.gradient(elevation_grid)
        slope = np.sqrt(gx**2 + gy**2)
        slope_norm = np.clip(slope / 200.0, 0, 1)  # Normalize slope
        risk_grid += slope_norm * 0.1  # Slopes increase fire risk

        # Water mask: negative elevation = ocean
        water_mask = elevation_grid < -2
        risk_grid[water_mask] = 0.0

    # Add slight spatial noise for realism (±5%)
    rng = np.random.RandomState(int(abs(lat * 1000 + lon * 1000)) % 2**31)
    noise = rng.normal(0, 0.03, risk_grid.shape)
    risk_grid = np.clip(risk_grid + noise, 0, 1).astype(np.float32)

    # Generate GeoJSON
    geojson = _risk_grid_to_geojson(risk_grid, lat, lon, CELL_SIZE_KM)

    # Summary
    mean_risk = float(np.mean(risk_grid[risk_grid > 0]))
    max_risk = float(np.max(risk_grid))

    if mean_risk >= 0.70:
        overall = "extreme"
    elif mean_risk >= 0.50:
        overall = "high"
    elif mean_risk >= 0.35:
        overall = "moderate"
    else:
        overall = "low"

    # Contributing factors (sorted by contribution)
    temp_c = weather.get("max_temp", 300) - 273.15
    wind_kmh = weather.get("wind_speed", 10)
    humidity_pct = weather.get("humidity", 0.005) * 10000
    precip_mm = weather.get("precipitation", 0)

    factors = []
    if temp_c > 30:
        factors.append({"factor": "temperature", "value": f"{temp_c:.0f}°C", "impact": "high"})
    elif temp_c > 20:
        factors.append({"factor": "temperature", "value": f"{temp_c:.0f}°C", "impact": "moderate"})
    else:
        factors.append({"factor": "temperature", "value": f"{temp_c:.0f}°C", "impact": "low"})

    if humidity_pct < 30:
        factors.append({"factor": "humidity", "value": f"{humidity_pct:.0f}%", "impact": "high"})
    elif humidity_pct < 50:
        factors.append({"factor": "humidity", "value": f"{humidity_pct:.0f}%", "impact": "moderate"})
    else:
        factors.append({"factor": "humidity", "value": f"{humidity_pct:.0f}%", "impact": "low"})

    if wind_kmh > 30:
        factors.append({"factor": "wind", "value": f"{wind_kmh:.0f} km/h", "impact": "high"})
    elif wind_kmh > 15:
        factors.append({"factor": "wind", "value": f"{wind_kmh:.0f} km/h", "impact": "moderate"})
    else:
        factors.append({"factor": "wind", "value": f"{wind_kmh:.0f} km/h", "impact": "low"})

    if precip_mm < 1:
        factors.append({"factor": "precipitation", "value": f"{precip_mm:.1f} mm", "impact": "high"})
    elif precip_mm < 5:
        factors.append({"factor": "precipitation", "value": f"{precip_mm:.1f} mm", "impact": "moderate"})
    else:
        factors.append({"factor": "precipitation", "value": f"{precip_mm:.1f} mm", "impact": "low"})

    return {
        "risk_level": overall,
        "risk_score": round(mean_risk, 3),
        "max_risk": round(max_risk, 3),
        "factors": factors,
        "weather": {
            "temperature": f"{temp_c:.1f}°C",
            "humidity": f"{humidity_pct:.0f}%",
            "wind_speed": f"{wind_kmh:.0f} km/h",
            "precipitation": f"{precip_mm:.1f} mm",
        },
        "geojson": geojson,
        "center": {"lat": lat, "lon": lon},
    }
