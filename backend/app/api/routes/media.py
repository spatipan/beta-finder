"""GET /api/thumb/:reel_id  and  GET /api/frames/:reel_id/:n"""
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from ml.cache.image_cache import thumb_path, frame_path

router = APIRouter()


@router.get("/thumb/{reel_id}")
async def get_thumbnail(reel_id: str):
    path = thumb_path(reel_id)
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(
        str(path),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=86400"},
    )


@router.get("/frames/{reel_id}/{n}")
async def get_keyframe(reel_id: str, n: int):
    path = frame_path(reel_id, n)
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(
        str(path),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=86400"},
    )
