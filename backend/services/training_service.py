import traceback

from backend.core.state import state, progress
from backend.models.cross_validator import CrossValidator
from backend.schemas.training import TrainingConfig
from backend.services.data_service import cargar_y_normalizar
from backend.utils.helpers import safe_float, safe_list


def run_training(config: TrainingConfig) -> None:
    """Ejecuta el entrenamiento completo en un hilo del executor."""
    try:
        progress.add({"tipo": "inicio", "mensaje": "Preparando datos..."})

        X, yo, scaler_X, scaler_y = cargar_y_normalizar(state.df, config.normalizar)
        state.X        = X
        state.yo       = yo
        state.scaler_X = scaler_X
        state.scaler_y = scaler_y
        state.normalizar  = config.normalizar

        progress.add({"tipo": "datos", "n_samples": len(X), "n_features": int(X.shape[1])})

        capas_list = [{"neuronas": c.neuronas, "activacion": c.activacion} for c in config.capas_config]
        validator  = CrossValidator(k=config.k)

        def fold_callback(fold_num, epoch, error_train, error_val):
            progress.add({
                "tipo":        "epoca",
                "fold":        fold_num + 1,
                "epoca":       epoch,
                "error_train": safe_float(error_train),
                "error_val":   safe_float(error_val),
            })

        progress.add({"tipo": "validacion_inicio", "k": config.k})

        resultado_cv = validator.validate(
            X, yo,
            eta=config.eta,
            eps=config.eps,
            capas_config=capas_list,
            neuronas_salida=config.neuronas_salida,
            activacion_salida=config.activacion_salida,
            max_epochs=config.max_epochs,
            callback=fold_callback,
        )

        folds_data = [
            {
                "fold":                    int(f["fold"]),
                "epocas":                  int(f["epocas"]),
                "n_train":                 int(f["n_train"]),
                "n_test":                  int(f["n_test"]),
                "error_train":             safe_float(f["error_train"]),
                "error_test":              safe_float(f["error_test"]),
                "error_total":             safe_float(f["error_total"]),
                "historial_error_train":   safe_list(f["historial_error_train"]),
                "historial_error_test":    safe_list(f["historial_error_test"]),
                "epoca_convergencia":      int(f["epoca_convergencia"]),
                "error_convergencia_train":safe_float(f["error_convergencia_train"]),
                "error_convergencia_test": safe_float(f["error_convergencia_test"]),
                "diferencia_convergencia": safe_float(f["diferencia_convergencia"]),
            }
            for f in resultado_cv["folds"]
        ]

        mejor_idx = resultado_cv["mejor_fold_idx"]
        mejor_raw = resultado_cv["folds"][mejor_idx]

        state.mejor_modelo     = mejor_raw["model"]
        state.activacion_salida = config.activacion_salida
        state.mejor_fold_num   = int(resultado_cv["mejor_fold"])

        resumen = {
            "mode":                    "kfold",
            "k":                       config.k,
            "eta":                     config.eta,
            "eps":                     config.eps,
            "capas_config":            capas_list,
            "neuronas_salida":         config.neuronas_salida,
            "max_epochs":              config.max_epochs,
            "normalizar":              config.normalizar,
            "total_samples":           int(len(X)),
            "folds":                   folds_data,
            "error_train_promedio":    safe_float(resultado_cv["error_train_promedio"]),
            "error_test_promedio":     safe_float(resultado_cv["error_test_promedio"]),
            "error_total_promedio":    safe_float(resultado_cv["error_total_promedio"]),
            "std_train":               safe_float(resultado_cv["std_train"]),
            "std_test":                safe_float(resultado_cv["std_test"]),
            "std_total":               safe_float(resultado_cv["std_total"]),
            "mejor_fold":              int(resultado_cv["mejor_fold"]),
            "mejor_fold_historial_train": safe_list(mejor_raw["historial_error_train"]),
            "mejor_fold_historial_test":  safe_list(mejor_raw["historial_error_test"]),
        }

        state.ultimo_resultado = resumen
        progress.add({"tipo": "fin", "mejor_fold": state.mejor_fold_num, "resultado": resumen})

    except Exception as e:
        progress.add({"tipo": "error", "mensaje": str(e), "detalle": traceback.format_exc()})
    finally:
        progress.finish()
