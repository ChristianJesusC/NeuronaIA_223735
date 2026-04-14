import numpy as np
from .neural_network import crear_modelo


class CrossValidator:
    def __init__(self, k: int = 5):
        self.k = k

    def split_data(self, X, yo):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        fold_size = len(indices) // self.k
        folds = []

        for i in range(self.k):
            start = i * fold_size
            end = start + fold_size if i < self.k - 1 else len(indices)
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            folds.append({
                "X_train": X[train_idx],
                "yo_train": yo[train_idx],
                "X_test": X[test_idx],
                "yo_test": yo[test_idx],
            })

        return folds

    def validate(
        self,
        X,
        yo,
        eta: float,
        eps: float,
        capas_config: list,
        neuronas_salida: int = 1,
        activacion_salida: str = "Lineal",
        max_epochs: int = 1000,
        callback=None,
        is_multiclass: bool = False,
    ):
        """
        Ejecuta K-Fold Cross Validation con un modelo Keras por fold.

        capas_config: lista de dicts [{"neuronas": int, "activacion": str}, ...]
        Retorna dict con resultados por fold y resumen global.
        """
        folds = self.split_data(X, yo)
        resultados_folds = []

        for fold_num, fold in enumerate(folds):
            n_features = fold["X_train"].shape[1]
            model = crear_modelo(n_features, capas_config, neuronas_salida, activacion_salida, eta)

            hist_train = []
            hist_test = []

            mejor_diff = float("inf")
            best_epoch = 0
            best_ee = 0.0
            best_ev = 0.0
            best_pesos = [w.copy() for w in model.get_weights()]
            epocas_realizadas = 0

            for epoch in range(1, max_epochs + 1):
                epocas_realizadas = epoch

                model.fit(
                    fold["X_train"],
                    fold["yo_train"],
                    epochs=1,
                    verbose=0,
                    batch_size=len(fold["X_train"]),
                )

                yc_train = model.predict(fold["X_train"], verbose=0)
                error_train = float(np.linalg.norm(yc_train - fold["yo_train"]))

                yc_test = model.predict(fold["X_test"], verbose=0)
                error_test = float(np.linalg.norm(yc_test - fold["yo_test"]))

                hist_train.append(error_train)
                hist_test.append(error_test)

                diff = abs(error_train - error_test)
                if diff < mejor_diff:
                    mejor_diff = diff
                    best_epoch = epoch
                    best_ee = error_train
                    best_ev = error_test
                    best_pesos = [w.copy() for w in model.get_weights()]

                if callback:
                    callback(fold_num, epoch, error_train, error_test)

                if error_train <= eps:
                    break

            yc_tr = model.predict(fold["X_train"], verbose=0)
            yc_ts = model.predict(fold["X_test"], verbose=0)

            n_train = len(fold["X_train"])
            n_test  = len(fold["X_test"])
            n_total = n_train + n_test
            e_train = float(np.linalg.norm(yc_tr - fold["yo_train"]))
            e_test  = float(np.linalg.norm(yc_ts - fold["yo_test"]))
            e_total = (e_train * n_train + e_test * n_test) / n_total

            # Accuracy para clasificación multiclase
            if is_multiclass:
                acc_train = float(np.mean(
                    np.argmax(yc_tr, axis=1) == np.argmax(fold["yo_train"], axis=1)
                ))
                acc_test = float(np.mean(
                    np.argmax(yc_ts, axis=1) == np.argmax(fold["yo_test"], axis=1)
                ))
            else:
                acc_train = None
                acc_test  = None

            resultados_folds.append({
                "fold": fold_num + 1,
                "model": model,
                "epocas": epocas_realizadas,
                "n_train": n_train,
                "n_test": n_test,
                "error_train": e_train,
                "error_test": e_test,
                "error_total": e_total,
                "historial_error_train": hist_train,
                "historial_error_test": hist_test,
                "epoca_convergencia": best_epoch,
                "error_convergencia_train": best_ee,
                "error_convergencia_test": best_ev,
                "diferencia_convergencia": float(mejor_diff),
                "pesos_convergencia": best_pesos,
                "accuracy_train": acc_train,
                "accuracy_test":  acc_test,
            })

        n_total_dataset = X.shape[0]
        error_train_prom = float(np.mean([r["error_train"] for r in resultados_folds]))
        error_test_prom  = float(np.mean([r["error_test"]  for r in resultados_folds]))
        std_train = float(np.std([r["error_train"] for r in resultados_folds]))
        std_test  = float(np.std([r["error_test"]  for r in resultados_folds]))

        # Error total ponderado por tamaño de partición (promedio entre folds)
        error_total_prom = float(np.mean([r["error_total"] for r in resultados_folds]))
        std_total = float(np.std([r["error_total"] for r in resultados_folds]))

        mejor_idx = int(np.argmin([r["error_test"] for r in resultados_folds]))

        # Accuracy promedio (solo multiclase)
        if is_multiclass:
            acc_train_prom = float(np.mean([r["accuracy_train"] for r in resultados_folds]))
            acc_test_prom  = float(np.mean([r["accuracy_test"]  for r in resultados_folds]))
        else:
            acc_train_prom = None
            acc_test_prom  = None

        return {
            "folds": resultados_folds,
            "error_train_promedio":  error_train_prom,
            "error_test_promedio":   error_test_prom,
            "error_total_promedio":  error_total_prom,
            "std_train":             std_train,
            "std_test":              std_test,
            "std_total":             std_total,
            "accuracy_train_prom":   acc_train_prom,
            "accuracy_test_prom":    acc_test_prom,
            "epocas_promedio":       float(np.mean([r["epocas"] for r in resultados_folds])),
            "mejor_fold":            resultados_folds[mejor_idx]["fold"],
            "mejor_fold_idx":        mejor_idx,
        }
