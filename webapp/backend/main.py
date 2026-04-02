"""FireSpreadNet — Backend API Server."""

import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable for model classes (src.models.*)
# Use append (not insert) so the backend's config.py takes precedence
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import CORS_ORIGINS
from services.model_service import model_service
from api.fires import router as fires_router
from api.simulation import router as simulation_router
from api.explainability import router as explainability_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model in background so the API is available immediately."""
    asyncio.get_event_loop().run_in_executor(None, model_service.load_model)
    yield


app = FastAPI(
    title="FireSpreadNet API",
    description="Wildfire spread prediction & active fire monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fires_router, prefix="/api/fires", tags=["fires"])
app.include_router(simulation_router, prefix="/api/simulation", tags=["simulation"])
app.include_router(explainability_router, prefix="/api/explainability", tags=["explainability"])


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_name": model_service.model_name,
    }
