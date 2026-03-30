"""Fire spread simulation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import simulation_service

router = APIRouter()


class SimulationRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 2.0


class StepRequest(BaseModel):
    hours: int = 1


@router.post("/start")
async def start_simulation(req: SimulationRequest):
    """Start a new fire spread simulation at given coordinates."""
    result = await simulation_service.create_simulation(
        req.latitude, req.longitude, req.radius_km,
    )
    if result.get("error"):
        raise HTTPException(400, result["message"])
    return result


@router.post("/from-active")
async def simulate_from_active(req: SimulationRequest):
    """Start simulation from an active detected fire."""
    result = await simulation_service.create_simulation(
        req.latitude, req.longitude, req.radius_km,
    )
    if result.get("error"):
        raise HTTPException(400, result["message"])
    return result


@router.post("/{sim_id}/step")
async def step(sim_id: str, req: StepRequest):
    """Advance simulation by N hours."""
    result = await simulation_service.step_simulation(sim_id, req.hours)
    if result is None:
        raise HTTPException(404, "Simulation not found")
    return result


@router.get("/{sim_id}")
async def get_simulation(sim_id: str):
    """Get full simulation state with all frames."""
    result = simulation_service.get_simulation(sim_id)
    if result is None:
        raise HTTPException(404, "Simulation not found")
    return result
