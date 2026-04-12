from fastapi import APIRouter, HTTPException

from backend.core.state import state
from backend.schemas.prediction import PredictConfig
from backend.services.prediction_service import predict

router = APIRouter()


@router.post("/predecir")
async def predecir(payload: PredictConfig):
    if state.mejor_modelo is None:
        raise HTTPException(status_code=400, detail="No hay modelo entrenado.")
    try:
        return predict(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")
