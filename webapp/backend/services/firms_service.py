"""Fetch active fire data from NASA FIRMS."""

import httpx
from typing import Optional
from config import FIRMS_MAP_KEY


FIRMS_CSV_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_OPEN_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"


async def fetch_active_fires(
    west: float, south: float, east: float, north: float,
    days: int = 1, source: str = "VIIRS_SNPP_NRT",
) -> list[dict]:
    """Fetch active fires within bounding box from FIRMS API.

    Falls back to sample data if no API key is configured.
    """
    if FIRMS_MAP_KEY:
        return await _fetch_from_firms(west, south, east, north, days, source)
    return _generate_sample_fires(west, south, east, north)


async def _fetch_from_firms(
    west: float, south: float, east: float, north: float,
    days: int, source: str,
) -> list[dict]:
    area = f"{west},{south},{east},{north}"
    url = f"{FIRMS_CSV_URL}/{FIRMS_MAP_KEY}/{source}/{area}/{days}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = lines[0].split(",")
    fires = []
    for line in lines[1:]:
        values = line.split(",")
        if len(values) < len(headers):
            continue
        row = dict(zip(headers, values))
        try:
            fires.append({
                "latitude": float(row.get("latitude", 0)),
                "longitude": float(row.get("longitude", 0)),
                "brightness": float(row.get("bright_ti4", row.get("brightness", 300))),
                "confidence": row.get("confidence", "nominal"),
                "acq_date": row.get("acq_date", ""),
                "acq_time": row.get("acq_time", ""),
                "frp": float(row.get("frp", 0)),
                "satellite": row.get("satellite", source),
            })
        except (ValueError, KeyError):
            continue
    return fires


def _generate_sample_fires(
    west: float, south: float, east: float, north: float,
) -> list[dict]:
    """Generate realistic sample fires for demo (no API key needed)."""
    import random
    import datetime

    random.seed(42)
    hotspots = [
        (37.0, -119.5), (34.2, -118.2), (40.5, -122.0), (38.5, -120.5),
        (36.0, -112.0), (44.5, -114.0), (47.5, -117.0), (33.8, -117.5),
        (41.0, -124.0), (35.5, -106.0), (39.0, -105.5), (43.7, -110.5),
        (-15.5, -47.5), (-12.9, -49.3), (-8.5, -63.0),
        (-33.8, 151.0), (-37.8, 144.9), (-34.0, 18.5),
        (38.7, 23.5), (39.5, -8.0), (42.5, 2.5),
    ]

    fires = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for lat, lon in hotspots:
        if west <= lon <= east and south <= lat <= north:
            for _ in range(random.randint(3, 12)):
                fires.append({
                    "latitude": lat + random.uniform(-0.5, 0.5),
                    "longitude": lon + random.uniform(-0.5, 0.5),
                    "brightness": random.uniform(300, 450),
                    "confidence": random.choice(["low", "nominal", "high"]),
                    "acq_date": now.strftime("%Y-%m-%d"),
                    "acq_time": now.strftime("%H%M"),
                    "frp": random.uniform(5, 150),
                    "satellite": "VIIRS_SNPP_NRT",
                })
    return fires
