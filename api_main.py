from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import io
import os
import shutil
from datetime import datetime
import traceback
from pathlib import Path
import math

from models.perceptron import Perceptron
from models.cross_validator import CrossValidator

app = FastAPI(title="Perceptron Neural Network API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingConfig(BaseModel):
    tasas: List[float]
    eps: float
    funcion: str
    usar_kfold: bool
    k_folds: int
    normalizar: bool = False
    normalizar_y: bool = False
    max_epochs: int = 1000
    max_intentos: int = 20
    iteraciones_post_minimo: Optional[int] = None

class TrainingState:
    def __init__(self):
        self.X = None
        self.yo = None
        self.df = None
        self.session_dir = None
        self.normalizar = False
        
training_state = TrainingState()

OUTPUT_DIR = "graficas_output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def safe_float(value, allow_large=True):
    try:
        f = float(value)
        if math.isnan(f):
            return 0.0
        if math.isinf(f):
            return 999999.0 if f > 0 else -999999.0
        if allow_large:
            return f
        if abs(f) > 1e10:
            return 999999.0 if f > 0 else -999999.0
        return f
    except:
        return 0.0

def safe_float_list(values, allow_large=True):
    return [safe_float(v, allow_large) for v in values]

def numpy_to_python(obj, allow_large=True):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return safe_float(obj, allow_large)
    elif isinstance(obj, np.ndarray):
        return [numpy_to_python(item, allow_large) for item in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value, allow_large) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item, allow_large) for item in obj]
    elif isinstance(obj, float):
        return safe_float(obj, allow_large)
    else:
        return obj

def crear_session_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"sesion_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def cargar_datos_con_normalizacion(df, normalizar=False, normalizar_y=False):
    from sklearn.preprocessing import StandardScaler
    
    data = df.values
    X_raw = data[:, :-1]
    yo_raw = data[:, -1].reshape(-1, 1)
    
    if normalizar:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        
        if normalizar_y:
            scaler_y = StandardScaler()
            yo_scaled = scaler_y.fit_transform(yo_raw)
            yo = yo_scaled
        else:
            yo = yo_raw
        
        ones_col = np.ones((X_scaled.shape[0], 1))
        X = np.hstack([ones_col, X_scaled])
    else:
        ones_col = np.ones((X_raw.shape[0], 1))
        X = np.hstack([ones_col, X_raw])
        yo = yo_raw
    
    return X, yo

@app.get("/")
def read_root():
    return {"message": "Perceptron Neural Network API", "version": "1.0"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None)
        
        training_state.df = df
        
        print(f"CSV cargado - Shape: {df.shape}")
        
        return {
            "message": "Archivo cargado exitosamente",
            "num_samples": int(df.shape[0]),
            "num_features": int(df.shape[1] - 1),
            "preview": df.head(10).to_dict()
        }
    except Exception as e:
        print(f"Error en upload_csv: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error al procesar archivo: {str(e)}")

@app.post("/train")
async def train(config: TrainingConfig):
    print(f"Recibida peticion de entrenamiento: {config}")
    
    if training_state.df is None:
        print("Error: No hay datos cargados")
        raise HTTPException(status_code=400, detail="No hay datos cargados. Sube un archivo CSV primero.")
    
    try:
        print(f"Iniciando entrenamiento - Mode: {'K-Fold' if config.usar_kfold else 'Normal'}")
        print(f"Normalizacion X: {'Activada' if config.normalizar else 'Desactivada'}")
        print(f"Normalizacion Y: {'Activada' if config.normalizar_y else 'Desactivada'}")
        print(f"Max epochs: {config.max_epochs}, Max intentos: {config.max_intentos}")
        if config.iteraciones_post_minimo:
            print(f"Iteraciones post-minimo: {config.iteraciones_post_minimo}")
        
        X, yo = cargar_datos_con_normalizacion(training_state.df, config.normalizar, config.normalizar_y)
        
        print(f"X shape: {X.shape}, yo shape: {yo.shape}")
        
        training_state.X = X
        training_state.yo = yo
        training_state.normalizar = config.normalizar
        
        training_state.session_dir = crear_session_dir()
        print(f"Session dir creado: {training_state.session_dir}")
        
        if config.usar_kfold:
            print("Entrenando con K-Fold...")
            resultado = entrenar_con_kfold(
                training_state.X,
                training_state.yo,
                config,
                training_state.session_dir
            )
        else:
            print("Entrenando sin K-Fold...")
            resultado = entrenar_sin_kfold(
                training_state.X,
                training_state.yo,
                config,
                training_state.session_dir
            )
        
        resultado["session_dir"] = training_state.session_dir
        resultado["normalizado"] = config.normalizar
        resultado["normalizado_y"] = config.normalizar_y
        resultado["max_epochs"] = config.max_epochs
        resultado["max_intentos"] = config.max_intentos
        resultado_convertido = numpy_to_python(resultado, allow_large=True)
        
        print("Entrenamiento completado exitosamente")
        return resultado_convertido
        
    except Exception as e:
        print(f"Error durante entrenamiento: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {str(e)}")

def entrenar_sin_kfold(X, yo, config, session_dir):
    resultados = []
    logs = []
    
    logs.append("=== CONFIGURACION INICIAL ===")
    logs.append(f"Tasas de aprendizaje: {config.tasas}")
    logs.append(f"Error permitido: {config.eps}")
    logs.append(f"Funcion de activacion: {config.funcion}")
    logs.append(f"Normalizacion X: {'Activada' if config.normalizar else 'Desactivada'}")
    logs.append(f"Normalizacion Y: {'Activada' if config.normalizar_y else 'Desactivada'}")
    logs.append(f"Max epocas: {config.max_epochs}")
    logs.append(f"Max intentos: {config.max_intentos}")
    if config.iteraciones_post_minimo:
        logs.append(f"Iteraciones post-minimo: {config.iteraciones_post_minimo}")
    logs.append("")
    
    for idx, eta in enumerate(config.tasas):
        print(f"Procesando tasa {idx+1}/{len(config.tasas)}: eta={eta}")
        logs.append(f"=== ENTRENAMIENTO {idx+1}/{len(config.tasas)} - eta = {eta} ===")
        
        perceptron = Perceptron(eta=eta, eps=config.eps, funcion_activacion=config.funcion)
        perceptron.inicializar_pesos(X.shape[1])
        
        logs.append(f"Pesos iniciales: {perceptron.w.flatten().tolist()}")
        
        convergio, epoch, razon_parada, mejor_epoca, mejor_error = perceptron.entrenar(
            X, yo, 
            max_epochs=config.max_epochs,
            max_intentos=config.max_intentos,
            iteraciones_post_minimo=config.iteraciones_post_minimo
        )
        
        logs.append(f"Razon de parada: {razon_parada}")
        logs.append(f"Mejor epoca: {mejor_epoca} con error: {mejor_error:.6f}")
        
        if convergio:
            logs.append(f"Convergio en {epoch} epocas")
        else:
            logs.append(f"No convergio - Detenido en epoca {epoch}")
        
        logs.append(f"Pesos finales: {perceptron.w.flatten().tolist()}")
        logs.append(f"Mejor configuracion de pesos: {perceptron.mejor_w.flatten().tolist()}")
        
        error_final = safe_float(perceptron.historial_error[-1], allow_large=True) if len(perceptron.historial_error) > 0 else 0.0
        logs.append(f"Error final: {error_final:.6f}")
        logs.append("")
        
        resultados.append({
            'eta': float(eta),
            'historial_error': safe_float_list(perceptron.historial_error, allow_large=True),
            'historial_w': [[safe_float(w, allow_large=True) for w in weights.flatten()] for weights in perceptron.historial_w],
            'pesos_finales': safe_float_list(perceptron.w.flatten(), allow_large=True),
            'mejor_w': safe_float_list(perceptron.mejor_w.flatten(), allow_large=True),
            'mejor_epoca': int(mejor_epoca),
            'mejor_error': safe_float(mejor_error, allow_large=True),
            'convergio': bool(convergio),
            'epocas': int(epoch),
            'razon_parada': razon_parada
        })
        
        print(f"  Completado - {razon_parada}, Epocas: {epoch}, Mejor error: {mejor_error:.6f}")
    
    return {
        'mode': 'sin_kfold',
        'resultados': resultados,
        'logs': logs
    }

def entrenar_con_kfold(X, yo, config, session_dir):
    print(f"Iniciando K-Fold con k={config.k_folds}")
    validator = CrossValidator(k=config.k_folds)
    resultados_por_eta = []
    logs = []
    
    logs.append("=== CONFIGURACION INICIAL ===")
    logs.append(f"Tasas de aprendizaje: {config.tasas}")
    logs.append(f"Error permitido: {config.eps}")
    logs.append(f"Funcion de activacion: {config.funcion}")
    logs.append(f"K-Fold: {config.k_folds} folds")
    logs.append(f"Normalizacion X: {'Activada' if config.normalizar else 'Desactivada'}")
    logs.append(f"Normalizacion Y: {'Activada' if config.normalizar_y else 'Desactivada'}")
    logs.append(f"Max epocas: {config.max_epochs}")
    logs.append(f"Max intentos: {config.max_intentos}")
    if config.iteraciones_post_minimo:
        logs.append(f"Iteraciones post-minimo: {config.iteraciones_post_minimo}")
    logs.append("")
    
    for idx, eta in enumerate(config.tasas):
        print(f"Procesando tasa {idx+1}/{len(config.tasas)}: eta={eta}")
        logs.append(f"=== VALIDACION CRUZADA {idx+1}/{len(config.tasas)} - eta = {eta} ===")
        
        try:
            resultado_cv = validator.validate(
                X, yo, eta, config.eps, config.funcion, 
                max_epochs=config.max_epochs,
                max_intentos=config.max_intentos,
                iteraciones_post_minimo=config.iteraciones_post_minimo
            )
            print(f"  Validacion completada para eta={eta}")
        except Exception as e:
            print(f"Error en validate: {str(e)}")
            print(traceback.format_exc())
            raise
        
        folds_data = []
        for fold_res in resultado_cv['folds']:
            ec_train = safe_float(fold_res['error_convergencia_train'], allow_large=True)
            ec_test = safe_float(fold_res['error_convergencia_test'], allow_large=True)
            diff_conv = safe_float(fold_res['diferencia_convergencia'], allow_large=True)
            mejor_error = safe_float(fold_res['mejor_error_entrenamiento'], allow_large=True)
            
            logs.append(
                f"Fold {fold_res['fold']}: {fold_res['razon_parada']} | "
                f"Mejor epoca: {fold_res['mejor_epoca_entrenamiento']} | "
                f"Mejor error: {mejor_error:.6f}"
            )
            
            folds_data.append({
                'fold': int(fold_res['fold']),
                'epocas': int(fold_res['epocas']),
                'convergio': bool(fold_res['convergio']),
                'razon_parada': fold_res['razon_parada'],
                'mejor_epoca_entrenamiento': int(fold_res['mejor_epoca_entrenamiento']),
                'mejor_error_entrenamiento': mejor_error,
                'epoca_convergencia': int(fold_res['epoca_convergencia']),
                'error_convergencia_train': ec_train,
                'error_convergencia_test': ec_test,
                'diferencia_convergencia': diff_conv,
                'pesos_convergencia': safe_float_list(fold_res['pesos_convergencia'].flatten(), allow_large=True),
                'mejor_w': safe_float_list(fold_res['mejor_w'].flatten(), allow_large=True),
                'historial_error_train': safe_float_list(fold_res['historial_error_train'], allow_large=True),
                'historial_error_test': safe_float_list(fold_res['historial_error_test'], allow_large=True),
            })
        
        error_train_prom = safe_float(resultado_cv['error_train_promedio'], allow_large=True)
        error_test_prom = safe_float(resultado_cv['error_test_promedio'], allow_large=True)
        std_train = safe_float(resultado_cv['std_train'], allow_large=True)
        std_test = safe_float(resultado_cv['std_test'], allow_large=True)
        
        logs.append(f"Promedios:")
        logs.append(f"  Error Train: {error_train_prom:.6f} ± {std_train:.6f}")
        logs.append(f"  Error Test: {error_test_prom:.6f} ± {std_test:.6f}")
        logs.append(f"  MEJOR FOLD: Fold {resultado_cv['mejor_fold']}")
        logs.append("")
        
        mejor_fold = min(resultado_cv['folds'], key=lambda x: safe_float(x['error_test'], allow_large=True))
        
        resultados_por_eta.append({
            'eta': float(eta),
            'folds': folds_data,
            'error_train_promedio': error_train_prom,
            'error_test_promedio': error_test_prom,
            'std_train': std_train,
            'std_test': std_test,
            'mejor_fold_numero': int(resultado_cv['mejor_fold']),
            'mejor_fold': {
                'historial_error': safe_float_list(mejor_fold['historial_error'], allow_large=True),
                'historial_w': [[safe_float(w, allow_large=True) for w in weights.flatten()] for weights in mejor_fold['historial_w']],
                'pesos_finales': safe_float_list(mejor_fold['pesos_finales'].flatten(), allow_large=True),
                'mejor_w': safe_float_list(mejor_fold['mejor_w'].flatten(), allow_large=True),
            }
        })
        
        print(f"  Completado eta={eta} - Mejor fold: {resultado_cv['mejor_fold']}")
    
    return {
        'mode': 'kfold',
        'k_folds': int(config.k_folds),
        'total_samples': len(X),
        'resultados': resultados_por_eta,
        'logs': logs
    }

@app.get("/download-graphs/{session_id}")
async def download_graphs(session_id: str):
    session_path = os.path.join(OUTPUT_DIR, f"sesion_{session_id}")
    
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    
    zip_path = f"{session_path}.zip"
    shutil.make_archive(session_path, 'zip', session_path)
    
    return FileResponse(zip_path, filename=f"graficas_{session_id}.zip")

@app.get("/list-sessions")
async def list_sessions():
    sessions = []
    if os.path.exists(OUTPUT_DIR):
        for item in os.listdir(OUTPUT_DIR):
            if item.startswith("sesion_") and os.path.isdir(os.path.join(OUTPUT_DIR, item)):
                session_id = item.replace("sesion_", "")
                sessions.append({
                    "id": session_id,
                    "path": item
                })
    return {"sessions": sessions}

@app.get("/status")
async def get_status():
    return {
        "data_loaded": training_state.df is not None,
        "num_samples": int(training_state.df.shape[0]) if training_state.df is not None else 0,
        "num_features": int(training_state.df.shape[1] - 1) if training_state.df is not None else 0,
        "normalizado": training_state.normalizar
    }

@app.delete("/clear-data")
async def clear_data():
    training_state.X = None
    training_state.yo = None
    training_state.df = None
    training_state.session_dir = None
    training_state.normalizar = False
    return {"message": "Datos limpiados exitosamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)