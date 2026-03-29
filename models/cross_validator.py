import numpy as np
from models.perceptron import Perceptron

class CrossValidator:
    def __init__(self, k=5):
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
            
            X_train = X[train_idx]
            yo_train = yo[train_idx]
            X_test = X[test_idx]
            yo_test = yo[test_idx]
            
            folds.append({
                'X_train': X_train,
                'yo_train': yo_train,
                'X_test': X_test,
                'yo_test': yo_test,
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            
        return folds
    
    def validate(self, X, yo, eta, eps, funcion, max_epochs=1000, max_intentos=20, iteraciones_post_minimo=None, callback=None):
        folds = self.split_data(X, yo)
        resultados_folds = []
        
        for fold_num, fold in enumerate(folds):
            perceptron = Perceptron(eta=eta, eps=eps, funcion_activacion=funcion)
            perceptron.inicializar_pesos(fold['X_train'].shape[1])
            
            historial_error_test = []
            historial_w_test = []
            
            def fold_callback(epoch, yc, E, dw, w, error_norm, intentos_restantes):
                yc_test = perceptron.predecir(fold['X_test'])
                error_test = np.linalg.norm(yc_test - fold['yo_test'])
                historial_error_test.append(error_test)
                historial_w_test.append(w.copy())
                
                if callback:
                    callback(fold_num, epoch, yc, E, dw, w, error_norm, intentos_restantes)
            
            convergio, epoch, razon_parada, mejor_epoca, mejor_error = perceptron.entrenar(
                fold['X_train'], 
                fold['yo_train'], 
                max_epochs=max_epochs,
                max_intentos=max_intentos,
                iteraciones_post_minimo=iteraciones_post_minimo,
                callback=fold_callback
            )
            
            yc_test = perceptron.predecir(fold['X_test'])
            error_test = np.linalg.norm(yc_test - fold['yo_test'])
            
            yc_train = perceptron.predecir(fold['X_train'])
            error_train = np.linalg.norm(yc_train - fold['yo_train'])
            
            min_diff = float('inf')
            best_epoch = 0
            best_ee = 0
            best_ev = 0
            best_pesos = perceptron.w.copy()
            
            for i, (ee, ev) in enumerate(zip(perceptron.historial_error, historial_error_test)):
                diff = abs(ee - ev)
                if diff < min_diff:
                    min_diff = diff
                    best_epoch = i + 1
                    best_ee = ee
                    best_ev = ev
                    best_pesos = historial_w_test[i].copy()
            
            resultados_folds.append({
                'fold': fold_num + 1,
                'perceptron': perceptron,
                'convergio': convergio,
                'epocas': epoch,
                'razon_parada': razon_parada,
                'mejor_epoca_entrenamiento': mejor_epoca,
                'mejor_error_entrenamiento': mejor_error,
                'error_train': error_train,
                'error_test': error_test,
                'pesos_finales': perceptron.w.copy(),
                'mejor_w': perceptron.mejor_w.copy(),
                'historial_error': perceptron.historial_error.copy(),
                'historial_error_train': perceptron.historial_error.copy(),
                'historial_error_test': historial_error_test.copy(),
                'historial_w': [w.copy() for w in perceptron.historial_w],
                'epoca_convergencia': best_epoch,
                'error_convergencia_train': best_ee,
                'error_convergencia_test': best_ev,
                'diferencia_convergencia': min_diff,
                'pesos_convergencia': best_pesos
            })
            
        error_train_promedio = np.mean([r['error_train'] for r in resultados_folds])
        error_test_promedio = np.mean([r['error_test'] for r in resultados_folds])
        std_train = np.std([r['error_train'] for r in resultados_folds])
        std_test = np.std([r['error_test'] for r in resultados_folds])
        
        mejor_fold_idx = np.argmin([r['error_test'] for r in resultados_folds])
        mejor_fold_numero = resultados_folds[mejor_fold_idx]['fold']
        
        return {
            'folds': resultados_folds,
            'error_train_promedio': error_train_promedio,
            'error_test_promedio': error_test_promedio,
            'std_train': std_train,
            'std_test': std_test,
            'epocas_promedio': np.mean([r['epocas'] for r in resultados_folds]),
            'mejor_fold': mejor_fold_numero,
            'mejor_fold_idx': mejor_fold_idx
        }