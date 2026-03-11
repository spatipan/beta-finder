"""GET /api/gyms"""
from fastapi import APIRouter
from backend.app.models.schema import GymInfo

router = APIRouter()

GYMS = [
    GymInfo(id="all", label="All Gyms", handle="all", color="#F59E0B"),
    GymInfo(id="alpine", label="Alpine Outpost", handle="the_alpine_outpost", color="#4CAF7D"),
    GymInfo(id="mainwall", label="Main Wall CNX", handle="mainwallcnx", color="#5B8DEE"),
    GymInfo(id="progression", label="Progression Vertical", handle="progressionvertical", color="#FF7043"),
]


@router.get("/gyms", response_model=list[GymInfo])
async def get_gyms() -> list[GymInfo]:
    return GYMS
