import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="EDA Penguins Dashboard (Interactive)",
    page_icon="🐧",
    layout="wide"
)

# Colores consistentes para las especies
color_map = {'Adelie': '#e41a1c', 'Chinstrap': '#377eb8', 'Gentoo': '#4daf4a'}

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🐧 Dashboard Interactivo: Palmer Penguins EDA")
st.markdown("""
Este panel interactivo resume los hallazgos del Análisis Exploratorio de Datos (EDA).
**Ahora 100% interactivo**: Pasa el ratón sobre los gráficos, haz zoom y usa los selectores.
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    return sns.load_dataset("penguins")

df_raw = load_data()
df_clean = df_raw.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']).copy()
columnas_fisicas = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# --- GRILLA 2x2 ---
st.markdown("---")
row1_col1, row1_col2 = st.columns(2)
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

# ==========================================
# CUADRANTE 1: VALORES FALTANTES
# ==========================================
with row1_col1:
    st.subheader("1. Valores Faltantes")
    st.markdown("Gráfico interactivo de cantidad de datos nulos por columna.")
    
    # Calcular nulos
    nulos = df_raw.isnull().sum().reset_index()
    nulos.columns = ['Variable', 'Cantidad de Nulos']
    
    # Crear gráfico Plotly
    fig1 = px.bar(
        nulos, x='Variable', y='Cantidad de Nulos', 
        text='Cantidad de Nulos',
        color='Cantidad de Nulos',
        color_continuous_scale=px.colors.sequential.Reds,
        title="Nulos en el Dataset Original"
    )
    fig1.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# CUADRANTE 2: OUTLIERS (BOXPLOTS)
# ==========================================
with row1_col2:
    st.subheader("2. Outliers (Anomalías Estadísticas)")
    
    # Menú desplegable interactivo
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
    fig2.update_layout(margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# CUADRANTE 3: PATRONES MORFOLÓGICOS (RADAR)
# ==========================================
with row2_col1:
    st.subheader("3. Patrones y Fenotipos (Radar)")
    st.markdown("Firma geométrica promedio de cada especie. (Haz clic en la leyenda para apagar/prender especies).")
    
    # Normalizar para el radar
    scaler = MinMaxScaler()
    df_norm = df_clean.copy()
    df_norm[columnas_fisicas] = scaler.fit_transform(df_clean[columnas_fisicas])
    promedios = df_norm.groupby('species')[columnas_fisicas].mean()
    
    etiquetas = ['Largo Pico', 'Prof. Pico', 'Largo Aleta', 'Masa Corporal']
    
    fig3 = go.Figure()
    for especie in promedios.index:
        valores = promedios.loc[especie].values.tolist()
        valores += valores[:1]  # Cerrar el polígono
        etiquetas_cerradas = etiquetas + [etiquetas[0]]
        
        fig3.add_trace(go.Scatterpolar(
            r=valores,
            theta=etiquetas_cerradas,
            fill='toself',
            name=especie,
            line_color=color_map[especie]
        ))
        
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# CUADRANTE 4: SEPARABILIDAD (PCA 2D)
# ==========================================
with row2_col2:
    st.subheader("4. Alta Separabilidad de Clases (PCA)")
    st.markdown("Pasa el ratón por los puntos para ver la información real del pingüino comprimido.")
    
    # Calcular PCA
    Xz = StandardScaler().fit_transform(df_clean[columnas_fisicas])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xz)
    var_exp = pca.explained_variance_ratio_
    
    # Crear un DataFrame para Plotly
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['species'] = df_clean['species'].values
    df_pca['island'] = df_clean['island'].values
    df_pca['sex'] = df_clean['sex'].values
    for col in columnas_fisicas:
        df_pca[col] = df_clean[col].values
        
    fig4 = px.scatter(
        df_pca, x='PC1', y='PC2', color='species',
        color_discrete_map=color_map,
        hover_data=['island', 'sex', 'body_mass_g', 'bill_length_mm'],
        labels={
            'PC1': f"PC1 (Tamaño) - {var_exp[0]:.1%} var",
            'PC2': f"PC2 (Pico) - {var_exp[1]:.1%} var"
        }
    )
    fig4.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig4.update_layout(margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig4, use_container_width=True)
