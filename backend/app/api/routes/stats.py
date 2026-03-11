"""GET /api/stats"""
from datetime import datetime, timezone
from fastapi import APIRouter
from backend.app.models.schema import StatsResponse
from ml.vectordb.store import stats as db_stats

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    data = db_stats()
    return StatsResponse(
        total=data["total"],
        by_gym=data["by_gym"],
        by_source=data["by_source"],
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
