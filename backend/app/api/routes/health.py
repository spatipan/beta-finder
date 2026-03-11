"""GET /api/health"""
from fastapi import APIRouter
from backend.app.models.schema import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
