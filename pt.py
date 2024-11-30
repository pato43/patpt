import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error

# Configuración inicial
st.set_page_config(
    page_title="Optimización de Recursos 2027",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema de colores
st.markdown("""
<style>
    .block-container { padding: 1.5rem 2rem; }
    h1, h2, h3 { color: #205375; font-weight: bold; }
    .stButton>button { background-color: #205375; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #1E90FF; }
    .css-18e3th9 { background-color: #F4F4F4; }
    .css-1d391kg { color: #333333; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎛️ Dashboard de Optimización de Recursos para 2027")
st.subheader("Automatización, Eficiencia y Detección de Anomalías")
st.markdown("""
**Objetivo:** Reducir pérdidas, detectar desviaciones en los fondos y optimizar los recursos financieros y de inventarios.  
Esta herramienta está diseñada para maximizar el impacto de los recursos en campañas electorales y mejorar la transparencia.
""")

# Carga de datos simulados con opciones interactivas
@st.cache_data
def load_data(custom_size):
    np.random.seed(42)
    categories = [
        "Salarios", "Administración", "Gastos Médicos", 
        "Limpieza", "Propaganda", "Capacitación"
    ]
    data = {
        "Categoría": np.random.choice(categories, custom_size),
        "Mes": np.random.choice(range(1, 13), custom_size),
        "Gasto ($)": np.random.randint(5000, 60000, custom_size),
        "Año": np.random.choice([2022, 2023, 2024], custom_size),
    }
    return pd.DataFrame(data)

# Elementos interactivos: Slider para el tamaño de los datos
with st.sidebar:
    st.header("🔧 Configuración")
    data_size = st.slider("Tamaño de los datos simulados:", min_value=100, max_value=1000, step=100, value=500)
    filtro_categoria = st.multiselect("Seleccionar Categorías", ["Salarios", "Administración", "Gastos Médicos", "Limpieza", "Propaganda", "Capacitación"], default=["Salarios", "Administración"])
    filtro_año = st.multiselect("Seleccionar Años", [2022, 2023, 2024], default=[2022, 2023])

# Carga de datos y filtros
data = load_data(data_size)
data_filtrada = data[data["Categoría"].isin(filtro_categoria) & data["Año"].isin(filtro_año)]

# Verificar si hay datos filtrados
if data_filtrada.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    tabs = st.tabs([
        "📊 Análisis General", 
        "🔎 Detección de Anomalías", 
        "📦 Optimización de Inventarios", 
        "📚 Predicciones de Gasto", 
        "🌐 Minería de Procesos",
        "🤖 Explicación de Modelos ML"
    ])

    # --- Pestaña 1: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General de Recursos")
        fig1 = px.bar(
            data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
            x="Categoría", y="Gasto ($)", color="Categoría",
            title="Gasto Total por Categoría", color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(
            data_filtrada.groupby("Mes")["Gasto ($)"].sum().reset_index(),
            x="Mes", y="Gasto ($)", title="Tendencia Mensual del Gasto",
            markers=True, line_shape="spline", color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña 2: Detección de Anomalías ---
    with tabs[1]:
        st.header("🔎 Detección de Anomalías")
        iforest = IsolationForest(contamination=0.05, random_state=42)
        if not data_filtrada[["Gasto ($)"]].empty:
            data_filtrada["Anomalía"] = iforest.fit_predict(data_filtrada[["Gasto ($)"]])
            anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]
            st.write(f"Se detectaron {len(anomalías)} anomalías:")
            st.dataframe(anomalías)

            fig_anomalías = px.scatter(
                anomalías, x="Mes", y="Gasto ($)", color="Categoría",
                title="Transacciones Sospechosas Detectadas",
                size="Gasto ($)", color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_anomalías, use_container_width=True)

    # --- Pestaña 3: Optimización de Inventarios ---
    with tabs[2]:
        st.header("📦 Optimización de Inventarios")
        kmeans = KMeans(n_clusters=3, random_state=42)
        data_filtrada["Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])
        fig_kmeans = px.scatter(
            data_filtrada, x="Mes", y="Gasto ($)", color="Cluster",
            title="Agrupamiento de Gastos por Inventario",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_kmeans, use_container_width=True)

    # --- Pestaña 4: Predicciones de Gasto ---
    with tabs[3]:
        st.header("📚 Predicciones de Gasto")
        lr = LinearRegression()
        X = data_filtrada[["Mes"]]
        y = data_filtrada["Gasto ($)"]
        if not X.empty:
            lr.fit(X, y)
            data_filtrada["Predicción ($)"] = lr.predict(X)
            fig_pred = px.line(
                data_filtrada, x="Mes", y="Predicción ($)", color="Categoría",
                title="Predicciones de Gasto Mensual",
                markers=True, line_shape="spline", color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_pred, use_container_width=True)

    # --- Pestaña 5: Minería de Procesos ---
    with tabs[4]:
        st.header("🌐 Minería de Procesos")
        st.image(
            "https://miro.medium.com/max/1400/1*X47Jl9zwmDRQz-Z1knG0gg.png",
            caption="Diagrama de Minería de Procesos", use_column_width=True
        )

    # --- Pestaña 6: Explicación de Modelos ML ---
    with tabs[5]:
        st.header("🤖 Modelos de Machine Learning: Explicación")
        st.markdown("""
        **Modelos utilizados:**
        1. **Isolation Forest:** Detecta anomalías mediante árboles de decisión que identifican puntos atípicos en los datos.
        2. **KMeans:** Agrupa datos en clústeres según similitudes, ideal para análisis de inventarios.
        3. **Regresión Lineal:** Predice tendencias futuras en gastos basado en datos históricos.
        """)
