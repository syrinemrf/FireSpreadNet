"""Fetch weather data from Open-Meteo (free, no API key)."""

import httpx
from config import OPEN_METEO_BASE


async def fetch_weather(lat: float, lon: float) -> dict:
    """Get current weather for a coordinate. Returns dict with model channel values."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "wind_direction_10m",
            "precipitation",
        ]),
        "daily": "temperature_2m_min,temperature_2m_max",
        "timezone": "auto",
        "forecast_days": 1,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(OPEN_METEO_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    daily = data.get("daily", {})

    return {
        "wind_speed": current.get("wind_speed_10m", 5.0),
        "wind_direction": current.get("wind_direction_10m", 180.0),
        "min_temp": daily.get("temperature_2m_min", [8.0])[0] + 273.15,
        "max_temp": daily.get("temperature_2m_max", [25.0])[0] + 273.15,
        "humidity": current.get("relative_humidity_2m", 40.0) / 10000.0,
        "precipitation": current.get("precipitation", 0.0),
        "drought_index": 0.0,
        "ndvi": 5000.0,
        "erc": 50.0,
        "population": 25.0,
    }


async def fetch_elevation(lat: float, lon: float, grid_size: int = 64) -> list:
    """Fetch elevation for centre point. Returns uniform grid (simplified)."""
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            elev = resp.json()["results"][0]["elevation"]
    except Exception:
        elev = 500.0
    return float(elev)
