import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Perceptron Neural Network",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a26;
    --accent: #7c6af7;
    --accent2: #f7716a;
    --text: #e8e6ff;
    --muted: #6b6880;
    --border: #2a2840;
    --train: #4ecdc4;
    --val: #ff6b6b;
    --sweet: #4dffd2;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Syne', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: var(--text) !important;
}

.block-container {
    padding-top: 2rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5a4dd4) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124, 106, 247, 0.4) !important;
}

.stNumberInput input, .stTextInput input {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stFileUploader {
    background-color: var(--surface2) !important;
    border: 1px dashed var(--accent) !important;
    border-radius: 12px !important;
}

.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}

.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

.metric-value.accent { color: var(--accent); }
.metric-value.accent2 { color: var(--accent2); }
.metric-value.sweet { color: var(--sweet); }

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
}

.section-tag {
    display: inline-block;
    background: rgba(124, 106, 247, 0.12);
    border: 1px solid rgba(124, 106, 247, 0.3);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 4px;
    margin-bottom: 0.8rem;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
}

.stTabs [aria-selected="true"] {
    background-color: var(--surface2) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stCheckbox {
    font-family: 'JetBrains Mono', monospace !important;
}

.warning-box {
    background: rgba(247, 113, 106, 0.1);
    border: 1px solid rgba(247, 113, 106, 0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent2);
}

.info-box {
    background: rgba(124, 106, 247, 0.1);
    border: 1px solid rgba(124, 106, 247, 0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
}
</style>
""", unsafe_allow_html=True)

if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1.5rem;">
    <div class="hero-title" style="font-size:1.6rem;">PerceptronCV</div>
    <div class="hero-sub">K-Fold Cross Validation</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-tag">Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Cargar CSV", type=['csv'])

if uploaded_file is not None:
    if st.sidebar.button("Cargar Datos", type="primary"):
        with st.spinner("Cargando datos..."):
            files = {'file': uploaded_file}
            response = requests.post(f"{API_URL}/upload-csv", files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.data_loaded = True
                st.sidebar.success(f"Cargado: {data['num_samples']} muestras")
            else:
                st.sidebar.error("Error al cargar archivo")

st.sidebar.markdown('<div style="margin-top:1rem;" class="section-tag">Preprocesamiento</div>', unsafe_allow_html=True)

normalizar = st.sidebar.checkbox("Normalizar X (features)", value=False)

normalizar_y = st.sidebar.checkbox("Normalizar Y (target)", value=False, disabled=not normalizar)

if normalizar and not normalizar_y:
    st.sidebar.markdown("""
    <div class="info-box">
    Solo se normalizaran las features (X). El target (Y) permanecera en su escala original para mejor interpretabilidad.
    </div>
    """, unsafe_allow_html=True)
elif normalizar and normalizar_y:
    st.sidebar.markdown("""
    <div class="warning-box">
    Se normalizaran tanto X como Y. Los errores estaran en escala normalizada. Util para convergencia mas rapida.
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<div style="margin-top:1rem;" class="section-tag">Criterios de Parada</div>', unsafe_allow_html=True)

max_epochs = st.sidebar.number_input("Iteraciones maximas", min_value=1, max_value=100000, value=1000, step=100)

max_intentos = st.sidebar.number_input("Intentos sin mejora", min_value=1, max_value=1000, value=20, step=1)

usar_limite_post_minimo = st.sidebar.checkbox("Limitar iteraciones despues del minimo", value=False)

iteraciones_post_minimo = None
if usar_limite_post_minimo:
    iteraciones_post_minimo = st.sidebar.number_input(
        "Iteraciones tras encontrar minimo", 
        min_value=1, 
        max_value=1000, 
        value=50, 
        step=10
    )

st.sidebar.markdown("""
<div class="info-box">
El entrenamiento se detiene cuando:
1. Error < ξ (convergencia)
2. Se alcanzan las iteraciones maximas
3. Error no mejora por N intentos
4. (Opcional) Se excede limite post-minimo
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="margin-top:1rem;" class="section-tag">Tasas de Aprendizaje</div>', unsafe_allow_html=True)

num_tasas = st.sidebar.number_input("Cantidad de tasas", min_value=1, max_value=10, value=3)

tasas = []
default_values = [0.001, 0.01, 0.1] if not normalizar else [0.01, 0.05, 0.1]

for i in range(num_tasas):
    default = default_values[i] if i < len(default_values) else 0.01
    tasa = st.sidebar.number_input(
        f"Tasa {i+1}", 
        min_value=0.0,
        value=default, 
        step=0.001, 
        format="%.6f",
        key=f"tasa_{i}"
    )
    tasas.append(tasa)

st.sidebar.markdown('<div style="margin-top:1rem;" class="section-tag">Parametros del Modelo</div>', unsafe_allow_html=True)

eps = st.sidebar.number_input("Error permitido (ξ)", min_value=0.0, value=0.1, step=0.01, format="%.6f")
funcion = st.sidebar.selectbox("Funcion de activacion", ["Lineal", "ReLU", "Binaria", "Sigmoide", "Gaussiano"])

st.sidebar.markdown('<div style="margin-top:1rem;" class="section-tag">Validacion Cruzada</div>', unsafe_allow_html=True)

usar_kfold = st.sidebar.checkbox("Activar K-Fold", value=True)
k_folds = st.sidebar.number_input("Numero de folds (K)", min_value=2, max_value=10, value=5, disabled=not usar_kfold)

st.sidebar.markdown("---")

if st.sidebar.button("Iniciar Entrenamiento", type="primary", disabled=not st.session_state.data_loaded):
    config = {
        "tasas": tasas,
        "eps": eps,
        "funcion": funcion,
        "usar_kfold": usar_kfold,
        "k_folds": k_folds,
        "normalizar": normalizar,
        "normalizar_y": normalizar_y,
        "max_epochs": max_epochs,
        "max_intentos": max_intentos,
        "iteraciones_post_minimo": iteraciones_post_minimo
    }
    
    with st.spinner("Entrenando modelo..."):
        try:
            response = requests.post(f"{API_URL}/train", json=config, timeout=600)
            
            if response.status_code == 200:
                st.session_state.training_results = response.json()
                st.sidebar.success("Entrenamiento completado")
            else:
                st.sidebar.error(f"Error: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="hero-title">Perceptron Neural Network</div>
    <div class="hero-title" style="font-size:1.8rem; -webkit-text-fill-color: var(--muted); background: none; color: var(--muted);">con Validacion Cruzada</div>
    <div class="hero-sub" style="margin-top:0.5rem;">Gradient Descent con criterio de parada optimizado</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.training_results:
    st.markdown("""
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 2.5rem; text-align: center; margin-top: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">▲</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem;">Listo para entrenar</div>
        <div style="font-size: 0.85rem; color: var(--muted); font-family: 'JetBrains Mono', monospace;">
            1. Carga un CSV<br>
            2. Configura iteraciones e intentos<br>
            3. Presiona <strong style="color:var(--accent)">Iniciar Entrenamiento</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

results = st.session_state.training_results

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Iteraciones maximas</div>
        <div class="metric-value accent">{results.get('max_epochs', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Intentos sin mejora</div>
        <div class="metric-value accent2">{results.get('max_intentos', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    norm_x = 'Si' if results.get('normalizado') else 'No'
    norm_y = 'Si' if results.get('normalizado_y') else 'No'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Normalizacion</div>
        <div class="metric-value sweet">X: {norm_x} | Y: {norm_y}</div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Convergencia", "Pesos", "Logs"])

with tab1:
    st.markdown('<div class="section-tag">Detalles por Fold</div>', unsafe_allow_html=True)
    
    if results['mode'] == 'kfold':
        for eta_result in results['resultados']:
            with st.expander(f"η = {eta_result['eta']}", expanded=True):
                mejor_fold_idx = None
                menor_error_total = float('inf')
                
                total_samples = results.get('total_samples', 150)
                
                k = results['k_folds']
                fold_base_size = total_samples // k
                remainder = total_samples % k
                
                fold_sizes = []
                for i in range(k):
                    if i < remainder:
                        fold_sizes.append(fold_base_size + 1)
                    else:
                        fold_sizes.append(fold_base_size)
                
                for idx, fold in enumerate(eta_result['folds']):
                    n_test = fold_sizes[idx]
                    n_train = total_samples - n_test
                    
                    ee_min = fold['mejor_error_entrenamiento']
                    ev_min = fold['error_convergencia_test']
                    
                    error_total = (n_train * ee_min + n_test * ev_min) / total_samples
                    
                    fold['n_train_calc'] = n_train
                    fold['n_test_calc'] = n_test
                    fold['error_total_calc'] = error_total
                    
                    if error_total < menor_error_total:
                        menor_error_total = error_total
                        mejor_fold_idx = fold['fold']
                        mejor_n_train = n_train
                        mejor_n_test = n_test
                        mejor_ee = ee_min
                        mejor_ev = ev_min
                
                fold_details = []
                for fold in eta_result['folds']:
                    es_mejor = fold['fold'] == mejor_fold_idx
                    marca = "⭐ " if es_mejor else ""
                    
                    num_pesos = len(fold['mejor_w'])
                    if num_pesos <= 3:
                        pesos_mejor = ", ".join([f"{float(p):.3f}" for p in fold['mejor_w']])
                        pesos_display = f"[{pesos_mejor}]"
                    else:
                        primeros_dos = ", ".join([f"{float(p):.3f}" for p in fold['mejor_w'][:2]])
                        pesos_display = f"[{primeros_dos}, ...+{num_pesos-2}]"
                    
                    fold_details.append({
                        'Fold': str(f"{marca}Fold {int(fold['fold'])}"),
                        'Error Train': float(fold['mejor_error_entrenamiento']),
                        'Error Test': float(fold['error_convergencia_test']),
                        'Error Total': float(fold['error_total_calc']),
                        'Pesos': str(pesos_display),
                        'Epoca Min': int(fold['mejor_epoca_entrenamiento']),
                        'Epocas': int(fold['epocas']),
                        'N Train': int(fold['n_train_calc']),
                        'N Test': int(fold['n_test_calc'])
                    })
                
                df_folds = pd.DataFrame(fold_details)
                
                df_folds['Error Train'] = df_folds['Error Train'].apply(lambda x: f"{x:.6f}")
                df_folds['Error Test'] = df_folds['Error Test'].apply(lambda x: f"{x:.6f}")
                df_folds['Error Total'] = df_folds['Error Total'].apply(lambda x: f"{x:.6f}")
                
                st.dataframe(df_folds, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                for fold in eta_result['folds']:
                    if fold['fold'] == mejor_fold_idx:
                        st.markdown(f"### ⭐ Mejor Modelo: Fold {fold['fold']}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            **Errores:**
                            - Error Train: `{fold['mejor_error_entrenamiento']:.6f}`
                            - Error Test: `{fold['error_convergencia_test']:.6f}`
                            - Error Total: `{fold['error_total_calc']:.6f}`
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **Épocas:**
                            - Época mínimo: `{fold['mejor_epoca_entrenamiento']}`
                            - Épocas totales: `{fold['epocas']}`
                            """)
                        
                        with col3:
                            st.markdown(f"""
                            **Observaciones:**
                            - Train: `{fold['n_train_calc']}`
                            - Test: `{fold['n_test_calc']}`
                            - Total: `{total_samples}`
                            """)
                        
                        st.markdown(f"""
                        **Fórmula Error Total:**  
                        `({mejor_n_train} × {mejor_ee:.6f} + {mejor_n_test} × {mejor_ev:.6f}) / {total_samples} = {menor_error_total:.6f}`
                        """)
                        
                        st.markdown("**Pesos en época de error mínimo:**")
                        pesos_str = ", ".join([f"W{i}={float(p):.6f}" for i, p in enumerate(fold['mejor_w'])])
                        st.code(pesos_str, language=None)
                        
                        break
    else:
        mejor_idx = None
        mejor_error = float('inf')
        
        for idx, resultado in enumerate(results['resultados']):
            error_final = resultado['historial_error'][-1] if resultado['historial_error'] else float('inf')
            if error_final < mejor_error:
                mejor_error = error_final
                mejor_idx = idx
        
        summary_data = []
        for idx, resultado in enumerate(results['resultados']):
            es_mejor = idx == mejor_idx
            eta_display = f"{resultado['eta']:.6e}" if resultado['eta'] < 0.001 else f"{resultado['eta']:.6f}"
            
            summary_data.append({
                'Tasa (η)': f"{'⭐ ' if es_mejor else ''}{eta_display}",
                'Epoca Menor Error': resultado['mejor_epoca'],
                'Menor Error': f"{resultado['mejor_error']:.6f}" + (" MEJOR" if es_mejor else ""),
                'Epocas Total': resultado['epocas'],
                'Convergio': 'Si' if resultado['convergio'] else 'No'
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

with tab2:
    st.markdown('<div class="section-tag">Analisis de Convergencia</div>', unsafe_allow_html=True)
    
    if results['mode'] == 'kfold':
        for eta_result in results['resultados']:
            eta_display = f"{eta_result['eta']:.6e}" if eta_result['eta'] < 0.001 else f"{eta_result['eta']:.6f}"
            st.markdown(f"### η = {eta_display}")
            
            num_folds = len(eta_result['folds'])
            cols_per_row = 2
            num_rows = (num_folds + cols_per_row - 1) // cols_per_row
            
            fig = make_subplots(
                rows=num_rows, 
                cols=cols_per_row,
                subplot_titles=[f"Fold {f['fold']}" for f in eta_result['folds']],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            for idx, fold in enumerate(eta_result['folds']):
                row = idx // cols_per_row + 1
                col = idx % cols_per_row + 1
                
                epochs_train = list(range(1, len(fold['historial_error_train']) + 1))
                epochs_test = list(range(1, len(fold['historial_error_test']) + 1))
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs_train, 
                        y=fold['historial_error_train'],
                        mode='lines',
                        name=f"Ee Fold {fold['fold']}",
                        line=dict(color='#4ecdc4', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs_test, 
                        y=fold['historial_error_test'],
                        mode='lines',
                        name=f"Ev Fold {fold['fold']}",
                        line=dict(color='#ff6b6b', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
                
                fig.add_vline(
                    x=fold['mejor_epoca_entrenamiento'], 
                    line_dash="dash",
                    line_color="#ffd700", 
                    line_width=2,
                    row=row, col=col
                )
            
            fig.update_layout(
                height=400 * num_rows,
                showlegend=True,
                paper_bgcolor='#111118',
                plot_bgcolor='#0f0f1a',
                font=dict(family="JetBrains Mono", color="#e8e6ff"),
                legend=dict(
                    bgcolor="#1a1a26",
                    bordercolor="#2a2840",
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(
                title_text="Epocas",
                gridcolor="#1e1e2e",
                linecolor="#2a2840"
            )
            
            fig.update_yaxes(
                title_text="Error",
                gridcolor="#1e1e2e",
                linecolor="#2a2840"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    else:
        fig = go.Figure()
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        
        for idx, resultado in enumerate(results['resultados']):
            epochs = list(range(1, len(resultado['historial_error']) + 1))
            eta_display = f"{resultado['eta']:.6e}" if resultado['eta'] < 0.001 else f"{resultado['eta']:.6f}"
            fig.add_trace(go.Scatter(
                x=epochs,
                y=resultado['historial_error'],
                mode='lines+markers',
                name=f"η = {eta_display}",
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
            
            fig.add_vline(
                x=resultado['mejor_epoca'],
                line_dash="dash",
                line_color="#ffd700",
                annotation_text=f"Mejor: {resultado['mejor_epoca']}",
                annotation_position="top"
            )
        
        fig.update_layout(
            title="Comparacion de Tasas de Aprendizaje",
            xaxis_title="Epocas",
            yaxis_title="Error (Norma)",
            paper_bgcolor='#111118',
            plot_bgcolor='#0f0f1a',
            font=dict(family="JetBrains Mono", color="#e8e6ff"),
            legend=dict(bgcolor="#1a1a26", bordercolor="#2a2840", borderwidth=1),
            height=500
        )
        
        fig.update_xaxes(gridcolor="#1e1e2e", linecolor="#2a2840")
        fig.update_yaxes(gridcolor="#1e1e2e", linecolor="#2a2840")
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-tag">Evolucion de Pesos</div>', unsafe_allow_html=True)
    
    if results['mode'] == 'kfold':
        for eta_result in results['resultados']:
            eta_display = f"{eta_result['eta']:.6e}" if eta_result['eta'] < 0.001 else f"{eta_result['eta']:.6f}"
            st.markdown(f"### η = {eta_display}")
            
            historial_w = eta_result['mejor_fold']['historial_w']
            
            if not historial_w:
                st.info("No hay datos de pesos disponibles")
                continue
            
            fig = go.Figure()
            
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
            
            num_pesos = len(historial_w[0])
            for i in range(num_pesos):
                pesos_i = [w[i] for w in historial_w]
                epochs = list(range(len(pesos_i)))
                
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=pesos_i,
                    mode='lines+markers',
                    name=f"W{i}",
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title=f"Evolucion de Pesos (η = {eta_display})",
                xaxis_title="Epocas",
                yaxis_title="Valor del Peso",
                paper_bgcolor='#111118',
                plot_bgcolor='#0f0f1a',
                font=dict(family="JetBrains Mono", color="#e8e6ff"),
                legend=dict(bgcolor="#1a1a26", bordercolor="#2a2840", borderwidth=1),
                height=400
            )
            
            fig.update_xaxes(gridcolor="#1e1e2e", linecolor="#2a2840")
            fig.update_yaxes(gridcolor="#1e1e2e", linecolor="#2a2840")
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        for resultado in results['resultados']:
            eta_display = f"{resultado['eta']:.6e}" if resultado['eta'] < 0.001 else f"{resultado['eta']:.6f}"
            st.markdown(f"### η = {eta_display}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pesos Finales**")
                pesos_df = pd.DataFrame({
                    'Peso': [f"W{i}" for i in range(len(resultado['pesos_finales']))],
                    'Valor': [f"{p:.6f}" for p in resultado['pesos_finales']]
                })
                st.dataframe(pesos_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Mejor Configuracion**")
                mejor_pesos_df = pd.DataFrame({
                    'Peso': [f"W{i}" for i in range(len(resultado['mejor_w']))],
                    'Valor': [f"{p:.6f}" for p in resultado['mejor_w']]
                })
                st.dataframe(mejor_pesos_df, use_container_width=True, hide_index=True)

with tab4:
    st.markdown('<div class="section-tag">Registro de Entrenamiento</div>', unsafe_allow_html=True)
    
    logs_text = "\n".join(results['logs'])
    st.code(logs_text, language="text")