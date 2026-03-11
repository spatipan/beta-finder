"""FastAPI entry point for BetaFinder CNX backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import health, gym, stats, search, media

app = FastAPI(
    title="BetaFinder CNX",
    version="1.1.0",
    description="Visual beta search engine for Chiang Mai climbing gyms",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(gym.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(media.router, prefix="/api")
