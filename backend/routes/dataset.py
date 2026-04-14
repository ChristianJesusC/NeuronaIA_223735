from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.core.state import state
from backend.services.data_service import detectar_tipo_salida, get_preview, parse_csv

router = APIRouter()


@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    has_header: bool = Form(False),
):
    try:
        contents = await file.read()
        df, col_mappings = parse_csv(contents, has_header=has_header)
        state.df          = df
        state.col_mappings = col_mappings

        tipo = detectar_tipo_salida(df)
        if tipo == "categorical":
            class_names = sorted(df.iloc[:, -1].unique().tolist())
            state.is_multiclass = True
            state.class_names   = class_names
            state.n_classes     = len(class_names)
        else:
            state.is_multiclass = False
            state.class_names   = []
            state.n_classes     = 1

        preview = get_preview(df)

        return {
            "message":       "Archivo cargado exitosamente",
            "num_samples":   int(df.shape[0]),
            "num_features":  int(df.shape[1] - 1),
            "is_multiclass": state.is_multiclass,
            "n_classes":     state.n_classes,
            "class_names":   state.class_names,
            "preview":       preview,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar archivo: {e}")


@router.get("/status")
async def get_status():
    return {
        "data_loaded":      state.df is not None,
        "num_samples":      int(state.df.shape[0]) if state.df is not None else 0,
        "num_features":     int(state.df.shape[1] - 1) if state.df is not None else 0,
        "modelo_entrenado": state.mejor_modelo is not None,
        "mejor_fold":       state.mejor_fold_num,
        "is_multiclass":    state.is_multiclass,
        "class_names":      state.class_names,
    }


@router.delete("/clear-data")
async def clear_data():
    state.df               = None
    state.X                = None
    state.yo               = None
    state.mejor_modelo     = None
    state.ultimo_resultado  = None
    state.mejor_fold_num   = -1
    state.is_multiclass    = False
    state.class_names      = []
    state.n_classes        = 1
    return {"message": "Datos limpiados"}
