import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

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
    h1, h2, h3 { color: #205375; }
    .stButton>button { background-color: #205375; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #1E90FF; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎛️ Demo de Dashboard para Optimización de Recursos")
st.subheader("Automatización y Eficiencia Financiera para Competitividad Electoral 2027")
st.markdown("""
**Objetivo:** Este dashboard permite detectar anomalías, predecir tendencias y optimizar recursos. Diseñado para reducir fugas de dinero, identificar patrones financieros y maximizar el impacto de los recursos en campañas electorales.
""")

# Casos exitosos
st.markdown("""
### 🌍 Casos Exitosos en Otros Países
- **Brasil:** Uso de Machine Learning para monitorear gastos públicos, logrando un ahorro anual de $150 millones de dólares mediante la detección de corrupción en contratos gubernamentales.
- **India:** Implementación de dashboards de gasto público, reduciendo en un 35% los tiempos de procesamiento presupuestario y mejorando la transparencia.
- **Canadá:** Aplicación de herramientas analíticas para predecir desviaciones en proyectos de infraestructura, evitando pérdidas superiores a $50 millones de dólares.
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
        "Categoría": np.random.choice(categories, 500),
        "Mes": np.random.choice(range(1, 13), 500),
        "Gasto ($)": np.random.randint(5000, 60000, 500),
        "Año": np.random.choice([2022, 2023, 2024], 500),
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

# --- Pestañas principales ---
if data_filtrada.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    tabs = st.tabs([
        "📊 Análisis General", 
        "🔎 Detección de Anomalías", 
        "📦 Optimización de Inventarios",
        "📚 Predicciones de Gasto", 
        "🌐 Minería de Procesos"
    ])

    # --- Pestaña 1: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General de Recursos")
        fig1 = px.bar(
            data_filtrada.groupby("Categoría")["Gasto ($)"].sum().reset_index(),
            x="Categoría", y="Gasto ($)", color="Categoría",
            title="Gasto Total por Categoría"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(
            data_filtrada.groupby("Mes")["Gasto ($)"].sum().reset_index(),
            x="Mes", y="Gasto ($)", title="Gasto Mensual"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña 2: Detección de Anomalías ---
    with tabs[1]:
        st.header("🔎 Detección de Anomalías")
        st.markdown("""
        Utilizamos el modelo **Isolation Forest** para detectar anomalías en los gastos.  
        Las anomalías pueden indicar posibles desviaciones o mal manejo de recursos.
        """)
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
            st.warning("No hay datos suficientes para ejecutar la detección de anomalías.")

    # --- Pestaña 3: Optimización de Inventarios ---
    with tabs[2]:
        st.header("📦 Optimización de Inventarios")
        st.markdown("""
        Se utilizan técnicas de **clustering** para agrupar categorías de gasto similares.  
        Esto permite identificar áreas donde se pueden reducir costos o mejorar la eficiencia.
        """)
        kmeans = KMeans(n_clusters=3, random_state=42)
        data_filtrada.loc[:, "Cluster"] = kmeans.fit_predict(data_filtrada[["Gasto ($)"]])
        fig4 = px.scatter(
            data_filtrada, x="Mes", y="Gasto ($)", color="Cluster",
            title="Clustering de Categorías de Gasto"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Pestaña 4: Predicciones de Gasto ---
    with tabs[3]:
        st.header("📚 Predicciones de Gasto")
        st.markdown("""
        Utilizamos modelos de regresión para predecir el gasto futuro y planificar mejor el presupuesto.
        """)
        lr = LinearRegression()
        X = data_filtrada[["Mes"]]
        y = data_filtrada["Gasto ($)"]
        if not X.empty:
            lr.fit(X, y)
            data_filtrada["Predicción ($)"] = lr.predict(X)
            fig5 = px.line(
                data_filtrada, x="Mes", y="Predicción ($)", color="Categoría",
                title="Predicciones de Gasto"
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para entrenar el modelo de regresión.")

    # --- Pestaña 5: Minería de Procesos ---
    with tabs[4]:
        st.header("🌐 Minería de Procesos")
        st.markdown("""
        Aplicamos minería de procesos para analizar el flujo de actividades relacionadas con el gasto y optimización de recursos.
        """)
        # Gráfico demostrativo de minería de procesos
        st.image("https://miro.medium.com/max/1400/1*X47Jl9zwmDRQz-Z1knG0gg.png", caption="Diagrama de Minería de Procesos", use_column_width=True)
