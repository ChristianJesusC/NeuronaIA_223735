import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.core.config import executor
from backend.core.state import state, progress
from backend.schemas.training import TrainingConfig
from backend.services.streaming_service import sse_generator
from backend.services.training_service import run_training

router = APIRouter()


@router.post("/entrenar")
async def entrenar(config: TrainingConfig):
    if state.df is None:
        raise HTTPException(status_code=400, detail="No hay datos cargados. Sube un CSV primero.")

    progress.clear()
    loop = asyncio.get_running_loop()
    asyncio.ensure_future(loop.run_in_executor(executor, run_training, config))

    return {"status": "iniciado"}


@router.get("/progreso")
async def progreso():
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/resultados")
async def resultados():
    if state.ultimo_resultado is None:
        raise HTTPException(status_code=404, detail="No hay resultados disponibles.")
    return state.ultimo_resultado


@router.get("/comparaciones")
async def comparaciones():
    return state.comparaciones


@router.delete("/comparaciones")
async def limpiar_comparaciones():
    state.comparaciones = []
    return {"message": "Historial de comparaciones limpiado"}
