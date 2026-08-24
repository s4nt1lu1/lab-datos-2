import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="EDA Penguins Dashboard",
    page_icon="🐧",
    layout="wide"
)

# Colores consistentes para las especies solicitados por el usuario
color_map = {'Adelie': '#FF8C00', 'Chinstrap': '#9932CC', 'Gentoo': '#008B8B'}

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🐧 Dashboard Interactivo: Palmer Penguins EDA")
st.markdown("""
Este panel interactivo resume los hallazgos del Análisis Exploratorio de Datos (EDA).
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    return sns.load_dataset("penguins")

df_raw = load_data()
df_clean = df_raw.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']).copy()
columnas_fisicas = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
df_numeric = df_clean[columnas_fisicas]

# --- GRILLA 2x2 ---
st.markdown("---")
row1_col1, row1_col2 = st.columns(2)
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

# ==========================================
# CUADRANTE 1: HEATMAP DE CORRELACIÓN
# ==========================================
with row1_col1:
    st.subheader("1. Correlación de Variables")
    st.markdown("Seleccione para ver la correlación global o segmentada por especie.")
    
    opciones_hm = ['Global', 'Adelie', 'Chinstrap', 'Gentoo']
    seleccion_hm = st.selectbox("Datos para Heatmap:", opciones_hm)
    
    if seleccion_hm == 'Global':
        corr_matrix = df_numeric.corr()
    else:
        corr_matrix = df_clean[df_clean['species'] == seleccion_hm][columnas_fisicas].corr()
        
    fig1 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f"Heatmap de Correlación ({seleccion_hm})"
    )
    fig1.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# CUADRANTE 2: OUTLIERS (BOXPLOTS)
# ==========================================
with row1_col2:
    st.subheader("2. Outliers (Anomalías Estadísticas)")
    
    opciones = {
        'bill_length_mm': 'Largo Pico (mm)',
        'bill_depth_mm': 'Profundidad Pico (mm)',
        'flipper_length_mm': 'Largo Aleta (mm)',
        'body_mass_g': 'Masa Corporal (g)'
    }
    variable_seleccionada = st.selectbox("Selecciona la variable a analizar:", list(opciones.keys()), format_func=lambda x: opciones[x])
    
    st.markdown(f"Distribución y valores atípicos para **{opciones[variable_seleccionada]}**.")
    
    fig2 = px.box(
        df_clean, x="species", y=variable_seleccionada, color="species",
        color_discrete_map=color_map, points="all",
        hover_data=["island", "sex"]
    )
    fig2.update_layout(margin=dict(t=10, b=20, l=20, r=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# CUADRANTE 3: REGRESIÓN (SCATTERPLOT + OLS) INTERACTIVO
# ==========================================
with row2_col1:
    st.subheader("3. Longitud vs Profundidad del Pico")
    st.markdown("Tendencia global vs Tendencias por especie.")
    
    r_global, p_global = pearsonr(df_clean['bill_length_mm'], df_clean['bill_depth_mm'])
    
    # Creamos el scatter interactivo base con Plotly
    fig3 = px.scatter(
        df_clean, x="bill_length_mm", y="bill_depth_mm", color="species",
        color_discrete_map=color_map, hover_data=["island", "sex"],
        labels={'bill_length_mm': 'Longitud del pico (mm)', 'bill_depth_mm': 'Profundidad del pico (mm)'}
    )
    
    # Calculamos y agregamos las líneas de regresión manualmente usando numpy
    # 1. Por especie
    for especie, color in color_map.items():
        sub = df_clean[df_clean['species'] == especie]
        if len(sub) > 1:
            m, b = np.polyfit(sub['bill_length_mm'], sub['bill_depth_mm'], 1)
            x_range = np.linspace(sub['bill_length_mm'].min(), sub['bill_length_mm'].max(), 50)
            fig3.add_trace(go.Scatter(
                x=x_range, y=m*x_range + b, mode='lines', 
                line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'
            ))

    # 2. Global
    m_glob, b_glob = np.polyfit(df_clean['bill_length_mm'], df_clean['bill_depth_mm'], 1)
    x_range_glob = np.linspace(df_clean['bill_length_mm'].min(), df_clean['bill_length_mm'].max(), 50)
    fig3.add_trace(go.Scatter(
        x=x_range_glob, y=m_glob*x_range_glob + b_glob, mode='lines', 
        name=f'Global (r={r_global:.2f})', 
        line=dict(color='white', dash='dash', width=2)
    ))

    fig3.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# CUADRANTE 4: COORDENADAS PARALELAS
# ==========================================
with row2_col2:
    st.subheader("4. Coordenadas Paralelas")
    st.markdown("Relación multivariada entre todas las características físicas.")
    
    # Mapeo numérico para Plotly parallel_coordinates
    df_paralelo = df_clean.copy()
    especie_a_numero = {'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}
    df_paralelo['species_id'] = df_paralelo['species'].map(especie_a_numero)
    
    # Escala de colores basada en los valores min(1), mid(2) y max(3)
    colores_escala = [
        (0.0, color_map['Adelie']),
        (0.5, color_map['Chinstrap']),
        (1.0, color_map['Gentoo'])
    ]
    
    fig4 = px.parallel_coordinates(
        df_paralelo, 
        color="species_id", 
        dimensions=columnas_fisicas,
        color_continuous_scale=colores_escala,
        labels={
            'bill_length_mm': 'L. Pico',
            'bill_depth_mm': 'P. Pico',
            'flipper_length_mm': 'L. Aleta',
            'body_mass_g': 'Masa',
            'species_id': 'Especie'
        }
    )
    
    # Ampliamos los márgenes izquierdo y derecho para que no se corten los textos
    fig4.update_layout(
        coloraxis_showscale=False, 
        margin=dict(t=50, b=40, l=90, r=90)
    )
    st.plotly_chart(fig4, use_container_width=True)
