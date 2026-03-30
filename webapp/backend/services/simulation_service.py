"""Fire spread simulation service."""

import uuid
import numpy as np
from typing import Optional

from config import GRID_SIZE, CELL_SIZE_KM
from services.model_service import model_service
from services.weather_service import fetch_weather, fetch_elevation


# In-memory simulation store (for demo — use Redis in production)
_simulations: dict = {}


async def _is_water(lat: float, lon: float) -> bool:
    """Check if a coordinate is over water using Open-Meteo marine API."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://marine-api.open-meteo.com/v1/marine",
                params={"latitude": lat, "longitude": lon, "current": "wave_height"}
            )
            data = resp.json()
            # If marine API returns current data, it's water
            if "current" in data and data["current"].get("wave_height") is not None:
                return True
    except Exception:
        pass
    return False


async def _build_land_mask(lat: float, lon: float) -> np.ndarray:
    """Build a land mask for the simulation grid. 1=land, 0=water.
    Uses elevation as proxy: ocean is typically elevation <= 0 at coast.
    Also checks a few grid boundary points via marine API."""
    import httpx
    deg_per_km = 1.0 / 111.0
    land_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Sample corners and edges for water detection
    sample_points = []
    for row_frac in [0, 0.25, 0.5, 0.75, 1.0]:
        for col_frac in [0, 0.25, 0.5, 0.75, 1.0]:
            r = int(row_frac * (GRID_SIZE - 1))
            c = int(col_frac * (GRID_SIZE - 1))
            pt_lat = lat + (GRID_SIZE / 2 - r) * CELL_SIZE_KM * deg_per_km
            pt_lon = lon + (c - GRID_SIZE / 2) * CELL_SIZE_KM * deg_per_km
            sample_points.append((r, c, pt_lat, pt_lon))

    try:
        lats = ",".join([f"{p[2]:.4f}" for p in sample_points])
        lons = ",".join([f"{p[3]:.4f}" for p in sample_points])
        locations = "|".join([f"{p[2]},{p[3]}" for p in sample_points])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                water_points = []
                for i, result in enumerate(results):
                    elev = result.get("elevation", 100)
                    if elev is not None and elev <= -5:
                        water_points.append(sample_points[i])

                # For each water sample point, mark a region around it as water
                for r, c, _, _ in water_points:
                    r_radius = GRID_SIZE // 4
                    for dr in range(-r_radius, r_radius + 1):
                        for dc in range(-r_radius, r_radius + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                                dist = np.sqrt(dr**2 + dc**2)
                                if dist <= r_radius:
                                    land_mask[nr, nc] = 0.0
    except Exception:
        pass

    return land_mask


async def create_simulation(lat: float, lon: float, radius_km: float = 2.0) -> dict:
    """Initialise a new fire simulation at (lat, lon)."""
    sim_id = str(uuid.uuid4())[:8]

    # Check if the location is in water
    water = await _is_water(lat, lon)
    if water:
        return {
            "error": True,
            "message": "Cannot start a fire simulation on water/ocean. Please select a land location.",
        }

    # Create initial fire mask — fire starts at centre
    fire_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    r_px = max(1, int(radius_km / CELL_SIZE_KM))
    yy, xx = np.ogrid[-cx:GRID_SIZE - cx, -cy:GRID_SIZE - cy]
    circle = (xx ** 2 + yy ** 2) <= r_px ** 2
    fire_mask[circle] = 1.0

    # Fetch real weather and elevation
    weather = await fetch_weather(lat, lon)
    elevation = await fetch_elevation(lat, lon)
    elevation_grid = np.full((GRID_SIZE, GRID_SIZE), elevation, dtype=np.float32)

    # Build land mask to prevent fire spreading into water
    land_mask = await _build_land_mask(lat, lon)

    # Apply land mask to initial fire (shouldn't start on water)
    fire_mask = fire_mask * land_mask

    _simulations[sim_id] = {
        "id": sim_id,
        "lat": lat,
        "lon": lon,
        "weather": weather,
        "elevation_grid": elevation_grid,
        "land_mask": land_mask,
        "fire_masks": [fire_mask.copy()],
        "hours_simulated": 0,
        "max_hours": 24,
    }

    return _build_response(sim_id)


async def step_simulation(sim_id: str, hours: int = 1) -> Optional[dict]:
    """Advance simulation by `hours` steps."""
    sim = _simulations.get(sim_id)
    if sim is None:
        return None

    for _ in range(hours):
        if sim["hours_simulated"] >= sim["max_hours"]:
            break

        current_fire = sim["fire_masks"][-1]
        weather = sim["weather"]
        elevation_grid = sim["elevation_grid"]

        input_tensor = model_service.build_input_tensor(
            weather, elevation_grid, current_fire,
        )
        pred = model_service.predict(input_tensor)

        # Union with previous fire (fire doesn't un-burn)
        new_fire = np.clip(np.maximum(current_fire, pred), 0, 1)

        # Apply land mask — fire cannot spread into water
        land_mask = sim.get("land_mask")
        if land_mask is not None:
            new_fire = new_fire * land_mask

        sim["fire_masks"].append(new_fire)
        sim["hours_simulated"] += 1

    return _build_response(sim_id)


def get_simulation(sim_id: str) -> Optional[dict]:
    """Retrieve a simulation's full state."""
    return _build_response(sim_id)


def _build_response(sim_id: str) -> Optional[dict]:
    """Build a unified simulation response for the frontend."""
    sim = _simulations.get(sim_id)
    if sim is None:
        return None
    frames = {}
    for h, mask in enumerate(sim["fire_masks"]):
        frames[str(h)] = _mask_to_geojson(mask, sim["lat"], sim["lon"])
    last_mask = sim["fire_masks"][-1]
    return {
        "simulation_id": sim_id,
        "lat": sim["lat"],
        "lon": sim["lon"],
        "current_hour": sim["hours_simulated"],
        "burned_area_km2": float(np.sum(last_mask > 0.5) * CELL_SIZE_KM ** 2),
        "frames": frames,
    }


def get_explainability(sim_id: str) -> Optional[dict]:
    """Return feature importance and analysis for a simulation."""
    sim = _simulations.get(sim_id)
    if sim is None:
        return None

    importance = model_service.get_feature_importance()
    weather = sim["weather"]

    factors = []
    for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
        val = weather.get(feature, None)
        factors.append({
            "feature": feature,
            "importance": score,
            "value": val,
        })

    return {
        "simulation_id": sim_id,
        "model_name": model_service.model_name or "physics_fallback",
        "factors": factors,
        "summary": _generate_summary(weather, importance),
    }


def _mask_to_geojson(mask: np.ndarray, center_lat: float, center_lon: float) -> dict:
    """Convert a 64×64 fire mask to GeoJSON polygons (grid cells) with probability."""
    deg_per_km = 1.0 / 111.0
    cell_deg = CELL_SIZE_KM * deg_per_km
    half = (GRID_SIZE / 2) * cell_deg
    features = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            prob = float(mask[row, col])
            if prob < 0.1:
                continue
            lat = center_lat + (GRID_SIZE / 2 - row) * cell_deg
            lon = center_lon + (col - GRID_SIZE / 2) * cell_deg
            # Create polygon (rectangle) for each grid cell
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon, lat],
                        [lon + cell_deg, lat],
                        [lon + cell_deg, lat - cell_deg],
                        [lon, lat - cell_deg],
                        [lon, lat],
                    ]]
                },
                "properties": {"probability": round(prob, 3)},
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "bbox": [
            center_lon - half, center_lat - half,
            center_lon + half, center_lat + half,
        ],
    }


def _generate_summary(weather: dict, importance: dict) -> str:
    """Generate a human-readable explanation of the prediction factors."""
    lines = []
    wind = weather.get("wind_speed", 0)
    humidity = weather.get("humidity", 0) * 10000
    temp = weather.get("max_temp", 300) - 273.15

    if wind > 20:
        lines.append(f"Strong winds ({wind:.0f} km/h) significantly drive fire spread.")
    elif wind > 10:
        lines.append(f"Moderate winds ({wind:.0f} km/h) contribute to fire spread.")
    else:
        lines.append(f"Light winds ({wind:.0f} km/h) — limited wind-driven spread.")

    if humidity < 30:
        lines.append(f"Low humidity ({humidity:.0f}%) — dry conditions favour fire.")
    elif humidity > 60:
        lines.append(f"High humidity ({humidity:.0f}%) — moisture slows spread.")

    if temp > 35:
        lines.append(f"High temperature ({temp:.0f}°C) increases fire risk.")

    lines.append("Previous fire location is the dominant predictor (80% importance).")
    return " ".join(lines)
