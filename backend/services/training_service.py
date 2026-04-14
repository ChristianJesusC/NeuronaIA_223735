import traceback

from backend.core.state import state, progress
from backend.models.cross_validator import CrossValidator
from backend.schemas.training import TrainingConfig
from backend.services.data_service import cargar_y_normalizar
from backend.utils.helpers import safe_float, safe_list


def _arquitectura_desc(capas_list: list, neuronas_salida: int, activacion_salida: str) -> str:
    partes = [f"{c['neuronas']}×{c['activacion']}" for c in capas_list]
    partes.append(f"{neuronas_salida}×{activacion_salida}")
    return " → ".join(partes)


def run_training(config: TrainingConfig) -> None:
    """Ejecuta el entrenamiento completo en un hilo del executor."""
    try:
        progress.add({"tipo": "inicio", "mensaje": "Preparando datos..."})

        X, yo, scaler_X, scaler_y = cargar_y_normalizar(
            state.df,
            config.normalizar,
            is_multiclass=state.is_multiclass,
            class_names=state.class_names,
        )
        state.X        = X
        state.yo       = yo
        state.scaler_X = scaler_X
        state.scaler_y = scaler_y
        state.normalizar = config.normalizar

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
            is_multiclass=state.is_multiclass,
            paciencia=config.paciencia,
        )

        folds_data = []
        for f in resultado_cv["folds"]:
            fd = {
                "fold":                     int(f["fold"]),
                "epocas":                   int(f["epocas"]),
                "n_train":                  int(f["n_train"]),
                "n_test":                   int(f["n_test"]),
                "error_train":              safe_float(f["error_train"]),
                "error_test":               safe_float(f["error_test"]),
                "error_total":              safe_float(f["error_total"]),
                "historial_error_train":    safe_list(f["historial_error_train"]),
                "historial_error_test":     safe_list(f["historial_error_test"]),
                "epoca_convergencia":       int(f["epoca_convergencia"]),
                "error_convergencia_train": safe_float(f["error_convergencia_train"]),
                "error_convergencia_test":  safe_float(f["error_convergencia_test"]),
                "error_convergencia_total": safe_float(f["error_convergencia_total"]),
            }
            if state.is_multiclass:
                fd["accuracy_train"] = safe_float(f["accuracy_train"])
                fd["accuracy_test"]  = safe_float(f["accuracy_test"])
            folds_data.append(fd)

        mejor_idx = resultado_cv["mejor_fold_idx"]
        mejor_raw = resultado_cv["folds"][mejor_idx]

        state.mejor_modelo      = mejor_raw["model"]
        state.activacion_salida = config.activacion_salida
        state.mejor_fold_num    = int(resultado_cv["mejor_fold"])

        resumen = {
            "mode":                       "kfold",
            "k":                          config.k,
            "eta":                        config.eta,
            "eps":                        config.eps,
            "capas_config":               capas_list,
            "neuronas_salida":            config.neuronas_salida,
            "activacion_salida":          config.activacion_salida,
            "max_epochs":                 config.max_epochs,
            "normalizar":                 config.normalizar,
            "total_samples":              int(len(X)),
            "is_multiclass":              state.is_multiclass,
            "class_names":                state.class_names,
            "folds":                      folds_data,
            "error_train_promedio":       safe_float(resultado_cv["error_train_promedio"]),
            "error_test_promedio":        safe_float(resultado_cv["error_test_promedio"]),
            "error_total_promedio":       safe_float(resultado_cv["error_total_promedio"]),
            "std_train":                  safe_float(resultado_cv["std_train"]),
            "std_test":                   safe_float(resultado_cv["std_test"]),
            "std_total":                  safe_float(resultado_cv["std_total"]),
            "mejor_fold":                 int(resultado_cv["mejor_fold"]),
            "mejor_fold_historial_train": safe_list(mejor_raw["historial_error_train"]),
            "mejor_fold_historial_test":  safe_list(mejor_raw["historial_error_test"]),
        }

        if state.is_multiclass:
            resumen["accuracy_train_prom"] = safe_float(resultado_cv["accuracy_train_prom"])
            resumen["accuracy_test_prom"]  = safe_float(resultado_cv["accuracy_test_prom"])

        state.ultimo_resultado = resumen

        # Añadir al historial de comparaciones
        entrada_comp = {
            "run":          len(state.comparaciones) + 1,
            "arquitectura": _arquitectura_desc(capas_list, config.neuronas_salida, config.activacion_salida),
            "k":            config.k,
            "error_train":  resumen["error_train_promedio"],
            "error_val":    resumen["error_test_promedio"],
            "error_total":  resumen["error_total_promedio"],
            "mejor_fold":   resumen["mejor_fold"],
            "tipo":         "multiclase" if state.is_multiclass else
                            ("binaria" if config.activacion_salida in ("Binaria", "Sigmoide") else "regresion"),
        }
        if state.is_multiclass:
            entrada_comp["accuracy"] = resumen["accuracy_test_prom"]
        state.comparaciones.append(entrada_comp)

        progress.add({"tipo": "fin", "mejor_fold": state.mejor_fold_num, "resultado": resumen})

    except Exception as e:
        progress.add({"tipo": "error", "mensaje": str(e), "detalle": traceback.format_exc()})
    finally:
        progress.finish()
