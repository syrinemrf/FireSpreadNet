"""Fetch weather data from Open-Meteo (free, no API key)."""

import math
import numpy as np
import httpx
from config import OPEN_METEO_BASE, GRID_SIZE, CELL_SIZE_KM


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

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        # Fallback weather if API fails
        return _default_weather()

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
        "ndvi": _estimate_ndvi(lat),
        "erc": _estimate_erc(current.get("temperature_2m", 25), current.get("relative_humidity_2m", 40)),
        "population": 25.0,
    }


def _default_weather() -> dict:
    """Fallback weather when API is unavailable."""
    return {
        "wind_speed": 10.0,
        "wind_direction": 225.0,
        "min_temp": 285.0,
        "max_temp": 305.0,
        "humidity": 0.003,
        "precipitation": 0.0,
        "drought_index": 0.0,
        "ndvi": 5000.0,
        "erc": 50.0,
        "population": 25.0,
    }


def _estimate_ndvi(lat: float) -> float:
    """Estimate NDVI based on latitude (tropical=high, desert/polar=low)."""
    abs_lat = abs(lat)
    if abs_lat < 15:
        return 7000.0   # Tropical — dense vegetation
    elif abs_lat < 30:
        return 4500.0   # Subtropical — moderate
    elif abs_lat < 45:
        return 5500.0   # Temperate — moderate-high
    elif abs_lat < 60:
        return 4000.0   # Boreal — moderate
    else:
        return 2000.0   # Polar — sparse


def _estimate_erc(temp_c: float, rh: float) -> float:
    """Estimate Energy Release Component from temperature and humidity."""
    # ERC increases with heat and dryness
    erc = max(0, (temp_c - 10) * 2.0 - rh * 0.5)
    return min(100, max(5, erc))


async def fetch_elevation(lat: float, lon: float) -> float:
    """Fetch elevation for centre point using Open-Meteo (fast, reliable)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                OPEN_METEO_BASE,
                params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
            )
            resp.raise_for_status()
            data = resp.json()
            elev = data.get("elevation", 500.0)
            if elev is not None:
                return float(elev)
    except Exception:
        pass
    return 500.0


async def fetch_elevation_grid(lat: float, lon: float) -> np.ndarray:
    """Fetch elevation for a grid of points to create realistic terrain variation.
    
    Uses Open-Meteo's multi-point API for speed (single request, ~200ms).
    Samples a sparse grid and interpolates to full resolution.
    """
    deg_per_km = 1.0 / 111.0
    half_extent = (GRID_SIZE / 2) * CELL_SIZE_KM * deg_per_km

    # Sample a 5x5 sparse grid
    sparse_n = 5
    sparse_lats = []
    sparse_lons = []
    for r in range(sparse_n):
        for c in range(sparse_n):
            frac_r = r / (sparse_n - 1)
            frac_c = c / (sparse_n - 1)
            pt_lat = lat + half_extent - frac_r * 2 * half_extent
            pt_lon = lon - half_extent + frac_c * 2 * half_extent
            sparse_lats.append(pt_lat)
            sparse_lons.append(pt_lon)

    try:
        lats_str = ",".join(f"{x:.4f}" for x in sparse_lats)
        lons_str = ",".join(f"{x:.4f}" for x in sparse_lons)
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                OPEN_METEO_BASE,
                params={
                    "latitude": lats_str,
                    "longitude": lons_str,
                    "current": "temperature_2m",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract elevations from multi-point response
        sparse_elevs = np.zeros((sparse_n, sparse_n), dtype=np.float32)
        if isinstance(data, list):
            for i, item in enumerate(data):
                r, c = divmod(i, sparse_n)
                sparse_elevs[r, c] = float(item.get("elevation", 500))
        else:
            # Single point response
            elev = float(data.get("elevation", 500))
            sparse_elevs[:] = elev

        # Bilinear interpolation to full grid
        from numpy import linspace
        sparse_rows = linspace(0, GRID_SIZE - 1, sparse_n)
        sparse_cols = linspace(0, GRID_SIZE - 1, sparse_n)
        full_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                # Find surrounding sparse indices
                ri = np.searchsorted(sparse_rows, r, side='right') - 1
                ci = np.searchsorted(sparse_cols, c, side='right') - 1
                ri = max(0, min(ri, sparse_n - 2))
                ci = max(0, min(ci, sparse_n - 2))

                # Bilinear weights
                r_frac = (r - sparse_rows[ri]) / max(1, sparse_rows[ri + 1] - sparse_rows[ri])
                c_frac = (c - sparse_cols[ci]) / max(1, sparse_cols[ci + 1] - sparse_cols[ci])
                r_frac = max(0, min(1, r_frac))
                c_frac = max(0, min(1, c_frac))

                v00 = sparse_elevs[ri, ci]
                v01 = sparse_elevs[ri, ci + 1]
                v10 = sparse_elevs[ri + 1, ci]
                v11 = sparse_elevs[ri + 1, ci + 1]

                full_grid[r, c] = (
                    v00 * (1 - r_frac) * (1 - c_frac) +
                    v01 * (1 - r_frac) * c_frac +
                    v10 * r_frac * (1 - c_frac) +
                    v11 * r_frac * c_frac
                )

        return full_grid

    except Exception:
        # Fallback: uniform elevation with slight random variation
        center_elev = await fetch_elevation(lat, lon)
        grid = np.full((GRID_SIZE, GRID_SIZE), center_elev, dtype=np.float32)
        # Add gentle slope variation
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                grid[r, c] += (r - GRID_SIZE / 2) * 2.0 + (c - GRID_SIZE / 2) * 1.5
        return grid
