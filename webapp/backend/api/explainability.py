"""Explainability / interpretability endpoints."""

from fastapi import APIRouter, HTTPException
from services.simulation_service import get_explainability
from services.model_service import model_service

router = APIRouter()


@router.get("/general")
async def explain_general():
    """Return general explainability info (no simulation needed)."""
    importance = model_service.get_feature_importance()
    factors = []
    for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
        factors.append({
            "feature": feature,
            "importance": score,
            "value": None,
        })
    return {
        "simulation_id": None,
        "model_name": model_service.model_name or "physics_fallback",
        "factors": factors,
        "summary": (
            "This panel shows the relative importance of each input feature in the fire spread prediction model. "
            "The previous fire mask (80%) is the dominant predictor — fire spreads from already-burning areas. "
            "Wind speed (8%) and humidity (4%) are the next most influential weather factors. "
            "Temperature, elevation, vegetation (NDVI), and energy release component (ERC) also contribute. "
            "Start a simulation to see context-specific explanations based on real weather data."
        ),
        "is_general": True,
    }


@router.get("/{sim_id}")
async def explain(sim_id: str):
    """Get feature importance and explanation for a simulation."""
    result = get_explainability(sim_id)
    if result is None:
        raise HTTPException(404, "Simulation not found")
    return result
