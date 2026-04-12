from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from backend.core.config import FRONTEND_DIR

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend no encontrado</h1>", status_code=404)
