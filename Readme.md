# Red Neuronal Multicapa — API + Frontend Web

Sistema de entrenamiento de redes neuronales multicapa con validación cruzada K-Fold, usando FastAPI como backend y una interfaz web en HTML/CSS/JS vanilla.

## Estructura del Proyecto

```
proyecto/
├── api_main.py              # Punto de entrada de la aplicación (2 líneas)
├── run_api.py               # Script para arrancar el servidor
├── requirements.txt         # Dependencias
├── estudiantes.csv          # Dataset de ejemplo (regresión)
├── estudiantes1.csv         # Dataset de ejemplo (clasificación binaria)
│
├── backend/
│   ├── core/
│   │   ├── app.py           # Factory de FastAPI (middleware, rutas, static)
│   │   ├── config.py        # Constantes: FRONTEND_DIR, executor
│   │   └── state.py         # AppState + ProgressTracker (singletons)
│   │
│   ├── schemas/
│   │   ├── training.py      # LayerConfig, TrainingConfig
│   │   └── prediction.py    # PredictConfig
│   │
│   ├── services/
│   │   ├── data_service.py       # Carga y normalización del CSV
│   │   ├── training_service.py   # Lógica de entrenamiento K-Fold
│   │   ├── streaming_service.py  # Generador SSE de progreso
│   │   └── prediction_service.py # Predicción con el mejor modelo
│   │
│   ├── routes/
│   │   ├── frontend.py      # GET /
│   │   ├── dataset.py       # POST /upload-csv, GET /status, DELETE /clear-data
│   │   ├── training.py      # POST /entrenar, GET /progreso, GET /resultados
│   │   └── prediction.py    # POST /predecir
│   │
│   ├── models/
│   │   ├── neural_network.py  # Construcción del modelo Keras Sequential
│   │   └── cross_validator.py # K-Fold Cross Validation
│   │
│   └── utils/
│       ├── helpers.py         # safe_float, safe_list
│       ├── data_loader.py     # Carga CSV sin normalizar
│       └── data_loader_nor.py # Carga CSV con StandardScaler
│
└── frontend/
    ├── index.html           # Estructura HTML de la interfaz
    ├── css/
    │   └── style.css        # Estilos (modo claro y oscuro)
    └── js/
        ├── layers.js        # Tabla de capas + diagrama de red (canvas)
        └── app.js           # Lógica de UI (CSV, entrenamiento, SSE, resultados)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python run_api.py
```

Abre el navegador en `http://localhost:8000`

## Uso

1. **Carga tu CSV** — arrastra o selecciona el archivo. La última columna es la variable a predecir; el resto son características de entrada.
2. **Configura la arquitectura** — define el número de capas ocultas, neuronas por capa y función de activación de cada una.
3. **Configura el entrenamiento** — K folds, learning rate, épsilon y máximo de épocas.
4. **Entrena** — presiona el botón y observa el progreso en tiempo real.
5. **Analiza los resultados** — revisa errores por fold, curvas de aprendizaje y logs detallados por época.
6. **Predice** — introduce nuevos datos para obtener predicciones con el mejor modelo.

## Funciones de Activación

| Función  | Uso recomendado                        |
|----------|----------------------------------------|
| ReLU     | Capas ocultas (uso general)            |
| Sigmoide | Salida de probabilidades (0 a 1)       |
| Lineal   | Salida en tareas de regresión          |
| Binaria  | Salida en clasificación binaria (0/1)  |

## API Endpoints

| Método   | Ruta           | Descripción                              |
|----------|----------------|------------------------------------------|
| GET      | `/`            | Sirve la interfaz web                    |
| POST     | `/upload-csv`  | Carga el dataset CSV                     |
| GET      | `/status`      | Estado actual del servidor               |
| POST     | `/entrenar`    | Lanza el entrenamiento en segundo plano  |
| GET      | `/progreso`    | Stream SSE con el progreso en tiempo real|
| GET      | `/resultados`  | Resultados del último entrenamiento      |
| POST     | `/predecir`    | Predicción con el mejor modelo           |
| DELETE   | `/clear-data`  | Limpia los datos y el modelo en memoria  |

## Características

- Red neuronal multicapa configurable por capa (neuronas y función de activación)
- K-Fold Cross Validation con detección automática de época de convergencia
- Error total ponderado por tamaño de partición: `(E_train×N_train + E_val×N_val) / N_total`
- Diagrama visual de la red en tiempo real con conteo de pesos entrenables
- Progreso de entrenamiento por streaming SSE (sin polling)
- Curvas de aprendizaje del mejor fold con Chart.js
- Logs detallados por época y por fold con época de convergencia resaltada
- Normalización opcional con StandardScaler
- Modo oscuro / claro con persistencia en localStorage
- Descarga del diagrama de red como PNG

## Tecnologías

| Capa      | Tecnología                          |
|-----------|-------------------------------------|
| Backend   | FastAPI, Uvicorn                    |
| ML        | TensorFlow 2.18 / Keras, scikit-learn |
| Datos     | Pandas, NumPy                       |
| Frontend  | HTML, CSS, JS vanilla, Chart.js     |
