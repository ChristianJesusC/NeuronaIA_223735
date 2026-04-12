from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.state import state
from backend.services.data_service import parse_csv

router = APIRouter()


@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        state.df = parse_csv(contents)
        return {
            "message": "Archivo cargado exitosamente",
            "num_samples": int(state.df.shape[0]),
            "num_features": int(state.df.shape[1] - 1),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar archivo: {e}")


@router.get("/status")
async def get_status():
    return {
        "data_loaded":     state.df is not None,
        "num_samples":     int(state.df.shape[0]) if state.df is not None else 0,
        "num_features":    int(state.df.shape[1] - 1) if state.df is not None else 0,
        "modelo_entrenado": state.mejor_modelo is not None,
        "mejor_fold":      state.mejor_fold_num,
    }


@router.delete("/clear-data")
async def clear_data():
    state.df              = None
    state.X               = None
    state.yo              = None
    state.mejor_modelo    = None
    state.ultimo_resultado = None
    state.mejor_fold_num  = -1
    return {"message": "Datos limpiados"}
