import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Configuración inicial
st.set_page_config(
    page_title="Demo de Dashboard para Optimización de Recursos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema de colores
st.markdown("""
<style>
    .css-18e3th9 { background-color: #1E1E1E; }
    .block-container { padding: 1.5rem 2rem; }
    h1, h2, h3 { color: #E0E0E0; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #333333;
        color: #E0E0E0;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        background-color: #444444;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background-color: #205375;
        color: #FFFFFF;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎛️ Demo de Dashboard para Optimización de Recursos")
st.subheader("Automatización y Eficiencia Financiera para Competitividad Electoral 2027")
st.markdown("""
**Objetivo:** Este dashboard permite detectar anomalías, predecir tendencias y optimizar recursos. Diseñado para reducir fugas de dinero, identificar patrones financieros y maximizar el impacto de los recursos en campañas electorales.
""")

# Propuesta de valor
st.markdown("""
### 💰 Ganancias Potenciales al Implementar Este Sistema
1. **Reducción de Pérdidas:** Con una detección oportuna de fugas de dinero, las instituciones pueden ahorrar entre un 15% y 30% de su presupuesto anual.
2. **Mayor Transparencia:** La automatización y visualización transparente aumentan la confianza de los votantes.
3. **Eficiencia Comercial:** Este sistema puede comercializarse a partidos políticos, ONGs e instituciones gubernamentales a un costo estimado de $50,000 a $100,000 USD por implementación, generando ingresos recurrentes por mantenimiento.
""")

# Carga de datos simulados
@st.cache_data
def load_data():
    np.random.seed(42)
    categories = [
        "Salarios", "Administración", "Gastos Médicos", 
        "Limpieza", "Propaganda", "Capacitación"
    ]
    data = {
        "Categoría": np.random.choice(categories, 300),  # Datos iniciales reducidos para pruebas
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
    # Pestañas principales
    tabs = st.tabs([
        "📊 Análisis General", 
        "🔎 Transacciones Sospechosas (Isolation Forest)", 
        "📦 Clustering de Inventarios (K-Means)", 
        "📚 Predicciones de Gasto (Regresión Lineal)"
    ])

    # --- Pestaña 1: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General de Recursos")
        col1, col2 = st.columns(2)
        
        # Gráfico de gasto por categoría
        fig1 = px.bar(
            data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
            x="Categoría", y="Gasto ($)", color="Categoría",
            title="Gasto Total por Categoría"
        )
        col1.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico de gasto mensual
        fig2 = px.line(
            data_filtrada.groupby("Mes")["Gasto ($)"].sum().reset_index(),
            x="Mes", y="Gasto ($)", title="Gasto Mensual"
        )
        col2.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña 2: Transacciones Sospechosas ---
    with tabs[1]:
        st.header("🔎 Transacciones Sospechosas (Isolation Forest)")
        iforest = IsolationForest(contamination=0.05, random_state=42)
        if not data_filtrada[["Gasto ($)"]].empty:
            data_filtrada.loc[:, "Anomalía"] = iforest.fit_predict(data_filtrada[["Gasto ($)"]])
            anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]
            st.write("Transacciones sospechosas detectadas:", anomalías)
            fig3 = px.scatter(
                anomalías, x="Mes", y="Gasto ($)", color="Categoría",
                title="Transacciones Sospechosas Detectadas"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para ejecutar el modelo de IsolationForest.")

    # --- Pestaña 3: Clustering de Inventarios ---
    with tabs[2]:
        st.header("📦 Clustering de Inventarios (K-Means)")
        kmeans = KMeans(n_clusters=3, random_state=42)
        if not data_filtrada[["Gasto ($)"]].empty:
            data_filtrada.loc[:, "Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])
            fig4 = px.scatter(
                data_filtrada, x="Mes", y="Gasto ($)", color="Cluster",
                title="Clustering de Gasto por Inventarios"
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para ejecutar el modelo K-Means.")

    # --- Pestaña 4: Predicciones de Gasto ---
    with tabs[3]:
        st.header("📚 Predicciones de Gasto (Regresión Lineal)")
        lr = LinearRegression()
        X = data_filtrada[["Mes"]]
        y = data_filtrada["Gasto ($)"]
        if not X.empty:
            lr.fit(X, y)
            predicciones = lr.predict(X)
            data_filtrada.loc[:, "Predicción ($)"] = predicciones
            
            fig5 = px.line(
                data_filtrada, x="Mes", y="Predicción ($)", color="Categoría",
                title="Predicciones de Gasto con Regresión Lineal"
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para entrenar el modelo de regresión lineal.")
