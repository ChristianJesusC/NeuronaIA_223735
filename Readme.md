# Perceptron Neural Network - API + Streamlit

Sistema de entrenamiento de perceptrones con validacion cruzada K-Fold, usando arquitectura API REST (FastAPI) y frontend moderno (Streamlit).

## Estructura del Proyecto

```
proyecto/
├── api_main.py              # API FastAPI
├── streamlit_app.py         # Frontend Streamlit
├── run_api.py              # Script para iniciar API
├── requirements.txt         # Dependencias
├── models/
│   ├── perceptron.py
│   └── cross_validator.py
└── utils/
    └── data_loader.py
```

## Instalacion

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecucion

### Opcion 1: Usar scripts separados

**Terminal 1 - Iniciar API:**
```bash
python run_api.py
```

**Terminal 2 - Iniciar Streamlit:**
```bash
streamlit run streamlit_app.py
```

### Opcion 2: Comando directo

**API:**
```bash
uvicorn api_main:app --reload --host 0.0.0.0 --port 8000
```

**Streamlit:**
```bash
streamlit run streamlit_app.py
```

## Uso

1. Abre el navegador en `http://localhost:8501` (Streamlit)
2. En el panel lateral, carga tu archivo CSV
3. Configura los parametros:
   - Tasas de aprendizaje
   - Error permitido
   - Funcion de activacion
   - K-Fold (activar/desactivar y numero de folds)
4. Presiona "Iniciar Entrenamiento"
5. Visualiza los resultados en las pestanas:
   - Resumen: Metricas generales
   - Convergencia: Graficas de error
   - Pesos: Evolucion de pesos
   - Logs: Registro detallado

## API Endpoints

- `POST /upload-csv` - Cargar archivo CSV
- `POST /train` - Entrenar modelo
- `GET /list-sessions` - Listar sesiones
- `GET /download-graphs/{session_id}` - Descargar graficas
- `DELETE /clear-data` - Limpiar datos

## Caracteristicas

- Entrenamiento con/sin K-Fold Cross Validation
- Comparacion de multiples tasas de aprendizaje
- Visualizacion interactiva con Plotly
- Deteccion automatica de punto de convergencia
- Exportacion de graficas en alta resolucion
- Logs detallados de entrenamiento
- Interfaz moderna y responsive

## Notas

- La API corre en `http://localhost:8000`
- Streamlit corre en `http://localhost:8501`
- Las graficas se guardan en `graficas_output/`
- Cada sesion crea una carpeta timestamped