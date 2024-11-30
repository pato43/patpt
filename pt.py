import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Configuración inicial
st.set_page_config(
    page_title="Demo Avanzado de Machine Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema de colores
st.markdown("""
<style>
    .block-container { padding: 1.5rem 2rem; }
    h1, h2, h3 { color: #205375; }
    .stButton>button { background-color: #205375; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #1E90FF; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🤖 Dashboard Interactivo de Machine Learning")
st.subheader("Explora y Aprende sobre Modelos de Machine Learning")
st.markdown("""
¡Bienvenido! Aquí podrás aprender cómo funcionan algunos modelos de Machine Learning a través de visualizaciones impactantes y explicaciones sencillas.
""")

# Interacción con el usuario
with st.sidebar:
    st.header("🔧 Opciones Interactivas")
    show_decision_tree = st.checkbox("Mostrar modelo: Árbol de Decisión")
    show_random_forest = st.checkbox("Mostrar modelo: Random Forest")
    show_svm = st.checkbox("Mostrar modelo: Máquina de Vectores de Soporte (SVM)")
    custom_data_points = st.slider("Tamaño de los datos simulados", min_value=50, max_value=500, step=50, value=300)
    st.markdown("---")
    st.button("Actualizar Gráficos")

# Generación de datos simulados
@st.cache_data
def simulate_data(n_points):
    np.random.seed(42)
    X, y = make_classification(
        n_samples=n_points, n_features=2, n_classes=2, n_clusters_per_class=1, flip_y=0.03, random_state=42
    )
    data = pd.DataFrame(X, columns=["Característica 1", "Característica 2"])
    data["Clase"] = y
    return data

data = simulate_data(custom_data_points)

# Gráficos demostrativos de modelos de Machine Learning
st.markdown("## 📈 Modelos de Machine Learning")

# --- Gráfico: Árbol de Decisión ---
if show_decision_tree:
    st.markdown("### 🌳 Árbol de Decisión")
    st.markdown("""
    Un **Árbol de Decisión** funciona como una serie de preguntas **SÍ/NO** para clasificar datos.  
    Imagina un árbol donde cada rama es una pregunta: ¿Es mayor a X? Sí, ve por aquí; No, ve por allá.  
    Es como un juego de "20 preguntas" para llegar a una respuesta.
    """)

    # Gráfico del árbol con datos simulados
    fig_tree = px.scatter(
        data, x="Característica 1", y="Característica 2", color="Clase",
        title="Visualización del Árbol de Decisión (Demostrativo)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# --- Gráfico: Random Forest ---
if show_random_forest:
    st.markdown("### 🌲🌲 Bosque Aleatorio (Random Forest)")
    st.markdown("""
    Un **Bosque Aleatorio** combina **muchos árboles de decisión** para hacer predicciones más confiables.  
    Es como preguntar la opinión a un grupo grande de personas y tomar la decisión más popular.
    """)

    # Datos de importancia de características simulados
    importance = pd.DataFrame({
        "Característica": ["Característica 1", "Característica 2"],
        "Importancia": [0.6, 0.4]
    })

    # Gráfico de importancia
    fig_rf = px.bar(
        importance, x="Característica", y="Importancia", color="Característica",
        title="Importancia de las Características en el Bosque Aleatorio",
        template="plotly_dark"
    )
    st.plotly_chart(fig_rf, use_container_width=True)

# --- Gráfico: SVM ---
if show_svm:
    st.markdown("### 💻 Máquina de Vectores de Soporte (SVM)")
    st.markdown("""
    Una **Máquina de Vectores de Soporte (SVM)** encuentra la **línea que separa mejor** dos grupos de datos.  
    Es como un árbitro que divide dos equipos en un partido, asegurándose de que haya la mayor distancia posible entre ellos.
    """)

    # Simulación de separación con SVM
    fig_svm = go.Figure()
    fig_svm.add_trace(go.Scatter(
        x=data["Característica 1"], y=data["Característica 2"],
        mode='markers', marker=dict(color=data["Clase"], colorscale="Viridis"),
        name="Puntos de Datos"
    ))
    fig_svm.add_shape(
        type="line", x0=-3, y0=-3, x1=3, y1=3, line=dict(color="magenta", dash="dash"),
        name="Hiperplano (SVM)"
    )
    fig_svm.update_layout(
        title="Máquina de Vectores de Soporte (SVM) - Hiperplano Separador",
        template="plotly_dark"
    )
    st.plotly_chart(fig_svm, use_container_width=True)

# --- Más interactividad ---
st.markdown("## 🎮 Más Interacciones")
st.markdown("""
### 🎨 Cambia el estilo de visualización:
Selecciona un tema:
""")
theme = st.radio("Selecciona un estilo:", ["Oscuro", "Claro"], index=0)
if theme == "Oscuro":
    st.markdown("""
    <style>
        .css-18e3th9 { background-color: #1E1E1E; color: #E0E0E0; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .css-18e3th9 { background-color: #FFFFFF; color: #000000; }
    </style>
    """, unsafe_allow_html=True)
