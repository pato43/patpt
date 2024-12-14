import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Configuración inicial
st.set_page_config(
    page_title="DEMO - Optimización Industrial Holman",
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
st.title("💡 Dashboard de Optimización Industrial - Grupo Holman")
st.subheader("Automatización, Eficiencia y Análisis de Datos Operativos")
st.markdown("""
**Objetivo:** Esta plataforma tiene como misión transformar los procesos industriales mediante la digitalización y optimización avanzada. 

Con herramientas de Machine Learning, visualización interactiva y simulación de procesos, se busca:
- Maximizar la eficiencia operativa.
- Identificar anomalías en tiempo real.
- Reducir costos innecesarios.
- Optimizar el uso de inventarios y recursos.

**¿Por qué este dashboard es relevante?**
Nuestro enfoque se centra en la integración de análisis predictivo y soluciones prácticas para enfrentar desafíos clave del sector industrial.
""")

# Carga de datos simulados con opciones interactivas
@st.cache_data
def load_data(custom_size):
    np.random.seed(42)
    categories = [
        "Producción", "Mantenimiento", "Energía",
        "Logística", "Inventarios", "Calidad"
    ]
    data = {
        "Categoría": np.random.choice(categories, custom_size),
        "Mes": np.random.choice(range(1, 13), custom_size),
        "Costo ($)": np.random.randint(5000, 60000, custom_size),
        "Año": np.random.choice([2022, 2023, 2024], custom_size),
    }
    return pd.DataFrame(data)

# Elementos interactivos: Slider para el tamaño de los datos
with st.sidebar:
    st.header("🔧 Configuración")
    data_size = st.slider("Tamaño de los datos simulados:", min_value=100, max_value=1000, step=100, value=500)
    filtro_categoria = st.multiselect("Seleccionar Categorías", ["Producción", "Mantenimiento", "Energía", "Logística", "Inventarios", "Calidad"], default=["Producción", "Logística"])
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
        "🛋 Optimización de Inventarios", 
        "📈 Predicciones de Costos",
        "🌐 Simulación de Procesos",
        "📚 Recomendaciones Personalizadas"
    ])

    # --- Pestaña 1: Análisis General ---
    with tabs[0]:
        st.header("📊 Análisis General de Operaciones")
        st.markdown("""
        **Enfoque:** Esta sección presenta un panorama general de los costos operativos para identificar patrones y áreas clave de oportunidad.
        """)
        
        fig1 = px.bar(
            data_filtrada.groupby("Categoría")["Costo ($)"].sum().reset_index(),
            x="Categoría", y="Costo ($)", color="Categoría",
            title="Costo Total por Categoría", color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(
            data_filtrada.groupby("Mes")["Costo ($)"].sum().reset_index(),
            x="Mes", y="Costo ($)", title="Tendencia Mensual de Costos",
            markers=True, line_shape="spline", color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Pestaña 2: Detección de Anomalías ---
    with tabs[1]:
        st.header("🔎 Detección de Anomalías")
        st.markdown("""
        **Propósito:** Identificar transacciones inusuales que podrían indicar errores o áreas de mejora.
        """)
        
        iforest = IsolationForest(contamination=0.05, random_state=42)
        if not data_filtrada[["Costo ($)"]].empty:
            data_filtrada["Anomalía"] = iforest.fit_predict(data_filtrada[["Costo ($)"]])
            anomalías = data_filtrada[data_filtrada["Anomalía"] == -1]
            st.write(f"Se detectaron {len(anomalías)} anomalías:")
            st.dataframe(anomalías)

            fig_anomalías = px.scatter(
                anomalías, x="Mes", y="Costo ($)", color="Categoría",
                title="Transacciones Sospechosas Detectadas",
                size="Costo ($)", color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_anomalías, use_container_width=True)

    # --- Pestaña 3: Optimización de Inventarios ---
    with tabs[2]:
        st.header("🛋 Optimización de Inventarios")
        st.markdown("""
        **Objetivo:** Agrupar costos asociados a inventarios para facilitar la toma de decisiones estratégicas.
        """)
        kmeans = KMeans(n_clusters=3, random_state=42)
        data_filtrada["Cluster"] = kmeans.fit_predict(data_filtrada[["Costo ($)"]])
        fig_kmeans = px.scatter(
            data_filtrada, x="Mes", y="Costo ($)", color="Cluster",
            title="Agrupamiento de Costos por Inventario",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_kmeans, use_container_width=True)

    # --- Pestaña 4: Predicciones de Costos ---
    with tabs[3]:
        st.header("📈 Predicciones de Costos")
        st.markdown("""
        **Análisis Predictivo:** Proyectar costos futuros con base en patrones históricos para anticiparse a posibles excesos.
        """)
        lr = LinearRegression()
        X = data_filtrada[["Mes"]]
        y = data_filtrada["Costo ($)"]
        if not X.empty:
            lr.fit(X, y)
            data_filtrada["Predicción ($)"] = lr.predict(X)
            fig_pred = px.line(
                data_filtrada, x="Mes", y="Predicción ($)", color="Categoría",
                title="Predicciones de Costos Mensuales",
                markers=True, line_shape="spline", color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_pred, use_container_width=True)

    # --- Pestaña 5: Simulación de Procesos ---
    with tabs[4]:
        st.header("🌐 Simulación de Procesos")
        st.markdown("""
        **Descripción:** Modelar y analizar flujos de trabajo clave para identificar posibles cuellos de botella y mejorar la asignación de recursos.
        """)
        simulated_data = pd.DataFrame({
            "Tarea": ["Planificación", "Producción", "Distribución", "Entrega"],
            "Duración (horas)": np.random.randint(2, 8, 4),
            "Recursos Utilizados": np.random.randint(50, 200, 4)
        })
        fig_simulation = px.bar(
            simulated_data, x="Tarea", y="Duración (horas)",
            title="Duración Estimada por Tarea",
            color="Recursos Utilizados", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_simulation, use_container_width=True)

        st.markdown("**Optimización sugerida:**")
        st.write(simulated_data)

    # --- Pestaña 6: Recomendaciones Personalizadas ---
    with tabs[5]:
        st.header("📚 Recomendaciones Personalizadas")
        st.markdown("""
        Basándonos en los datos analizados, sugerimos las siguientes acciones para maximizar la eficiencia:
        - **Automatizar tareas recurrentes** en las áreas de Producción y Logística.
        - **Implementar controles preventivos** en los puntos identificados como anómalos.
        - **Monitorear y ajustar** inventarios según los patrones de demanda proyectados.
        - **Evaluar opciones de inversión** en tecnología para reducir costos energéticos.

        Estas recomendaciones están diseñadas para adaptarse dinámicamente a los resultados del análisis.
        """)
