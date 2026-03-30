"""Active fire data & fire declaration endpoints."""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from services.firms_service import fetch_active_fires

router = APIRouter()


class DeclaredFire(BaseModel):
    latitude: float = None
    longitude: float = None
    lat: float = None
    lon: float = None
    radius_km: float = 1.0
    description: str = ""
    reporter: str = "anonymous"

    def get_lat(self):
        return self.latitude if self.latitude is not None else self.lat

    def get_lon(self):
        return self.longitude if self.longitude is not None else self.lon


# In-memory store for declared fires
_declared_fires: list[dict] = []


@router.get("/active")
async def get_active_fires(
    west: float = Query(-180), south: float = Query(-90),
    east: float = Query(180), north: float = Query(90),
    days: int = Query(1, ge=1, le=10),
):
    """Get active fires (from FIRMS or sample data) within bounding box."""
    fires = await fetch_active_fires(west, south, east, north, days)
    return {"count": len(fires), "fires": fires}


@router.post("/declare")
async def declare_fire(fire: DeclaredFire):
    """Declare a fire at a given location (user-submitted report)."""
    entry = fire.model_dump()
    # Normalize lat/lon fields
    entry["latitude"] = fire.get_lat()
    entry["longitude"] = fire.get_lon()
    entry["lat"] = entry["latitude"]
    entry["lon"] = entry["longitude"]
    entry["id"] = len(_declared_fires) + 1
    entry["source"] = "user_report"
    _declared_fires.append(entry)
    return {"status": "received", "fire": entry}


@router.get("/declared")
async def get_declared_fires():
    """Get all user-declared fires."""
    return {"count": len(_declared_fires), "fires": _declared_fires}
