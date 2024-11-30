import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Configuración inicial
st.set_page_config(
    page_title="Análisis de Recursos y Machine Learning",
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
st.title("📊 Dashboard de Análisis y Machine Learning")
st.subheader("Automatización y Eficiencia Financiera para Recursos del Partido")

# Carga de datos simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    categories = ["Salarios", "Administración", "Gastos Médicos", "Limpieza", "Propaganda", "Capacitación"]
    data = {
        "Categoría": np.random.choice(categories, 300),
        "Mes": np.random.choice(range(1, 13), 300),
        "Gasto ($)": np.random.randint(5000, 60000, 300),
        "Año": np.random.choice([2022, 2023, 2024], 300),
    }
    return pd.DataFrame(data)

data = load_data()

# Barra lateral
with st.sidebar:
    st.header("Opciones de Filtro")
    filtro_categoria = st.multiselect("Seleccionar Categorías", data["Categoría"].unique(), default=data["Categoría"].unique())
    filtro_año = st.multiselect("Seleccionar Años", data["Año"].unique(), default=data["Año"].unique())

# Filtrar datos
data_filtrada = data[data["Categoría"].isin(filtro_categoria) & data["Año"].isin(filtro_año)]

# Verificar si hay datos filtrados
if data_filtrada.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    # --- Gráfico inicial ---
    st.header("📊 Análisis de Recursos")
    fig1 = px.bar(
        data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
        x="Categoría", y="Gasto ($)", color="Categoría",
        title="Gasto Total por Categoría"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Gráficos y modelos de Machine Learning ---
    st.header("🤖 Modelos de Machine Learning y Explicaciones")

    # Datos para gráficos demostrativos
    @st.cache_data
    def generate_ml_data(n_points=300):
        np.random.seed(42)
        X, y = make_classification(
            n_samples=n_points, n_features=2, n_classes=2,
            n_clusters_per_class=1, flip_y=0.03, random_state=42
        )
        return pd.DataFrame(X, columns=["Característica 1", "Característica 2"]), y

    ml_data, ml_labels = generate_ml_data()

    # --- Árbol de Decisión ---
    st.subheader("🌳 Árbol de Decisión")
    st.markdown("""
    Un **Árbol de Decisión** es como un juego de "20 preguntas". Se divide en ramas según respuestas 
    "SÍ" o "NO" a preguntas sobre los datos. Matemáticamente, busca maximizar la **ganancia de información** o 
    reducir la **impureza de Gini** en cada división. Es ideal para decisiones rápidas y visualización.
    """)

    fig_tree = px.scatter(
        ml_data, x="Característica 1", y="Característica 2",
        color=ml_labels.astype(str), title="Clasificación Simulada - Árbol de Decisión"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- Bosque Aleatorio ---
    st.subheader("🌲 Bosque Aleatorio")
    st.markdown("""
    Un **Bosque Aleatorio** combina múltiples Árboles de Decisión, cada uno entrenado con una muestra diferente 
    del conjunto de datos. Esto reduce el riesgo de **sobreajuste**. Matemáticamente, usa promedios o votación 
    para decidir una clasificación.
    """)

    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_classifier.fit(ml_data, ml_labels)
    rf_feature_importance = pd.DataFrame({
        "Característica": ["Característica 1", "Característica 2"],
        "Importancia": rf_classifier.feature_importances_
    })

    fig_rf = px.bar(
        rf_feature_importance, x="Característica", y="Importancia",
        title="Importancia de Características en el Bosque Aleatorio"
    )
    st.plotly_chart(fig_rf, use_container_width=True)

    # --- K-Means ---
    st.subheader("📦 Clustering con K-Means")
    st.markdown("""
    **K-Means** agrupa datos en **K grupos**, buscando minimizar la distancia entre los puntos y el centroide del grupo.  
    Es útil para segmentación y descubrimiento de patrones en datos. Matemáticamente, usa el algoritmo de optimización 
    de Lloyd para minimizar la **suma de distancias cuadradas**.
    """)

    kmeans = KMeans(n_clusters=2, random_state=42)
    ml_data["Cluster"] = kmeans.fit_predict(ml_data)
    fig_kmeans = px.scatter(
        ml_data, x="Característica 1", y="Característica 2",
        color=ml_data["Cluster"].astype(str), title="Clustering con K-Means"
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

    # --- PCA ---
    st.subheader("🌐 Análisis de Componentes Principales (PCA)")
    st.markdown("""
    El **PCA** reduce la dimensionalidad de los datos al proyectarlos en un nuevo espacio con menor número de 
    dimensiones. Encuentra combinaciones lineales de características que retienen la mayor **varianza** posible.
    """)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(ml_data[["Característica 1", "Característica 2"]])
    pca_df = pd.DataFrame(pca_result, columns=["Componente Principal 1", "Componente Principal 2"])
    pca_df["Clase"] = ml_labels

    fig_pca = px.scatter(
        pca_df, x="Componente Principal 1", y="Componente Principal 2",
        color=pca_df["Clase"].astype(str), title="Reducción de Dimensiones con PCA"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# Elementos interactivos
st.markdown("## 🎮 Más Interacciones")
st.markdown("### Cambia el número de datos:")
data_size = st.slider("Cantidad de datos simulados:", min_value=100, max_value=500, step=50, value=300)
st.button("Actualizar datos")
