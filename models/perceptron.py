import numpy as np

class Perceptron:
    def __init__(self, eta=0.05, eps=0.1, funcion_activacion="Lineal"):
        self.eta = eta
        self.eps = eps
        self.funcion_activacion = funcion_activacion
        self.w = None
        self.historial_w = []
        self.historial_error = []
        self.mejor_w = None
        self.mejor_error = float('inf')
        self.mejor_epoca = 0
        
    def inicializar_pesos(self, n_features):
        self.w = np.round(np.random.uniform(-1, 1, size=(n_features, 1)), 3)
        self.historial_w = [self.w.copy()]
        self.historial_error = []
        self.mejor_w = self.w.copy()
        self.mejor_error = float('inf')
        self.mejor_epoca = 0
        
    def aplicar_activacion(self, u):
        if self.funcion_activacion == "Lineal":
            return u
        elif self.funcion_activacion == "ReLU":
            return np.maximum(0, u)
        elif self.funcion_activacion == "Binaria":
            return np.where(u >= 0, 1, -1)
        elif self.funcion_activacion == "Sigmoide":
            return 1 / (1 + np.exp(-u))
        elif self.funcion_activacion == "Gaussiano":
            return np.exp(-u**2)
    
    def entrenar(self, X, yo, max_epochs=1000, max_intentos=20, iteraciones_post_minimo=None, callback=None):
        epoch = 0
        intentos_sin_mejora = 0
        iteraciones_despues_minimo = 0
        
        usar_limite_post_minimo = iteraciones_post_minimo is not None and iteraciones_post_minimo > 0
        
        while epoch < max_epochs:
            epoch += 1
            
            u = np.dot(X, self.w)
            yc = self.aplicar_activacion(u)
            E = yc - yo
            dw = -self.eta * (np.dot(X.T, E))
            self.w = dw + self.w 
            error_norm = np.linalg.norm(E)
            
            self.historial_error.append(error_norm)
            self.historial_w.append(self.w.copy())
            
            if error_norm < self.mejor_error:
                self.mejor_error = error_norm
                self.mejor_w = self.w.copy()
                self.mejor_epoca = epoch
                intentos_sin_mejora = 0
                iteraciones_despues_minimo = 0
            else:
                intentos_sin_mejora += 1
                if usar_limite_post_minimo:
                    iteraciones_despues_minimo += 1
            
            if callback:
                callback(epoch, yc, E, dw, self.w, error_norm, intentos_sin_mejora)
            
            if error_norm <= self.eps:
                razon_parada = "convergencia_eps"
                break
            
            if intentos_sin_mejora >= max_intentos:
                razon_parada = "max_intentos"
                break
            
            if usar_limite_post_minimo and iteraciones_despues_minimo >= iteraciones_post_minimo:
                razon_parada = "limite_post_minimo"
                break
        else:
            razon_parada = "max_epochs"
        
        return error_norm <= self.eps, epoch, razon_parada, self.mejor_epoca, self.mejor_error
    
    def predecir(self, X):
        u = np.dot(X, self.w)
        return self.aplicar_activacion(u)