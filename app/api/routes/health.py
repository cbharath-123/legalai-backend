from fastapi import APIRouter

from app.api.schemas import HealthResponse, ReadinessResponse
from app.core.config import get_settings
from app.db.client import check_db_connection

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def liveness():
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness():
    settings = get_settings()
    db_ok = await check_db_connection()
    return ReadinessResponse(
        status="ok" if db_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        llm_provider=settings.llm_provider,
    )
