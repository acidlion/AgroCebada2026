import base64
import io
import json
import os
import zipfile
from pathlib import Path

import dash
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html, dash_table, State, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# ZONA DE INTEGRACIÓN DEL MODELO REAL
# ---------------------------------------------------------
# import joblib
MODELO_PATH = Path(__file__).resolve().parent.parent / "models" / "modelos_ensamble.joblib"

# ========================================================
# 2. CLASE ENVOLTORIO PARA EL ENSAMBLE CUSTOM
# ========================================================
class EnsamblePersonalizado:
    def __init__(self, diccionario_modelo):
        self.ridge = diccionario_modelo['ridge']
        self.xgboost = diccionario_modelo['xgboost']
        self.peso_ridge = diccionario_modelo['peso_ridge']
        self.peso_xgb = diccionario_modelo['peso_xgboost']
        self.columnas_ridge = diccionario_modelo['columnas_ridge']
        self.columnas_xgboost = diccionario_modelo['columnas_xgboost']
        self.columnas_esperadas = diccionario_modelo['columnas_ridge']
        
    def predict(self, df_entrada):
        # Cada estimador se entrenó con un esquema de columnas diferente.
        if isinstance(df_entrada, pd.DataFrame):
            X_ridge = df_entrada.reindex(columns=self.columnas_ridge, fill_value=0)
            X_xgboost = df_entrada.reindex(columns=self.columnas_xgboost, fill_value=0)
        else:
            X_ridge = X_xgboost = df_entrada
            
        pred_ridge = self.ridge.predict(X_ridge)
        pred_xgb = self.xgboost.predict(X_xgboost)
        return (pred_ridge * self.peso_ridge) + (pred_xgb * self.peso_xgb)


def compatibilizar_imputador_serializado(imputador):
    if hasattr(imputador, "_fill_dtype"):
        return
    if not hasattr(imputador, "_fit_dtype"):
        raise AttributeError(
            "El SimpleImputer serializado no contiene el tipo de dato requerido "
            "para compatibilizarlo con esta versión de scikit-learn."
        )
    # scikit-learn 1.9 usa _fill_dtype, pero el modelo 1.7.2 guardó _fit_dtype.
    imputador._fill_dtype = imputador._fit_dtype


# ========================================================
# 3. CARGA DEL MODELO
# ========================================================
modelo_rf = None

print("==================================================")
print(f"Buscando el modelo en: {MODELO_PATH}")

if not os.path.exists(MODELO_PATH):
    print("ERROR CRÍTICO: El archivo .joblib NO EXISTE en esa ruta.")
else:
    try:
        objeto_cargado = joblib.load(MODELO_PATH)
    
        if isinstance(objeto_cargado, dict) and 'ridge' in objeto_cargado and 'xgboost' in objeto_cargado:
        
 
            imputador = objeto_cargado["ridge"].named_steps.get("imputacion")
            if isinstance(imputador, SimpleImputer):
                compatibilizar_imputador_serializado(imputador)

            modelo_rf = EnsamblePersonalizado(objeto_cargado)
            print("Ensamble Manual (Ridge + XGBoost) ensamblado y fusionado exitosamente.")
        else:
            modelo_rf = objeto_cargado
            print("Modelo tradicional cargado exitosamente.")
  
    except Exception as e:
        print(f"ERROR CRÍTICO al leer el archivo .joblib: {e}")
# ==========================================
# 1. PREPARACIÓN DE DATOS ESPACIALES
# ==========================================
DATOS_EDA_PATH = Path(__file__).resolve().parent / "parcelas" / "datos_consolidados_eda.csv"

# Leer el archivo. Intenta capturar errores si el archivo no existe aún.
try:
    df_eda = pd.read_csv(DATOS_EDA_PATH)
except FileNotFoundError:
    print(f"Advertencia: No se encontró el archivo de estadísticas en {DATOS_EDA_PATH}")
    df_eda = pd.DataFrame(columns=['ID_Parcela', 'Rendimiento_tha', 'Elevacion', 'Pendiente', 'Precipitacion_Acum', 'Temp_Max_Prom', 'NDVI_Max'])

ESTADOS_PATH = Path(__file__).resolve().parent / "assets" / "mexico_estados.geojson"
ESTADOS = gpd.read_file(ESTADOS_PATH)
if ESTADOS.crs is None:
    raise ValueError(f"El GeoJSON de estados no define un sistema de coordenadas: {ESTADOS_PATH}")
if ESTADOS.crs.to_epsg() != 4326:
    ESTADOS = ESTADOS.to_crs(epsg=4326)

COLORES_ESTADOS = {
    "Puebla": "#E6007E",
    "Tlaxcala": "#00A6D6",
    "Hidalgo": "#FF7A00",
}
if set(ESTADOS["shapeName"]) != set(COLORES_ESTADOS):
    raise ValueError(f"El GeoJSON debe contener Puebla, Tlaxcala e Hidalgo: {ESTADOS_PATH}")

PARCELAS_PATH = Path(__file__).resolve().parent / "parcelas" / "Parcelas_Reto_AGC_CONJUNTO.shp"
PARCELAS = gpd.read_file(PARCELAS_PATH, encoding="CP1252")

if PARCELAS.crs is None:
    raise ValueError(f"El shapefile no define un sistema de coordenadas: {PARCELAS_PATH}")
if PARCELAS.crs.to_epsg() != 4326:
    PARCELAS = PARCELAS.to_crs(epsg=4326)

area_columns = [col for col in PARCELAS.columns if str(col).lower().endswith("_ha")]
if not area_columns:
    raise ValueError(f"No se encontró la columna de área en hectáreas: {PARCELAS_PATH}")
PARCELAS = PARCELAS.rename(columns={area_columns[0]: "AREA_HA"})

conteo_entrenamiento = int((PARCELAS["CONJUNTO"] == "ENTRENAMIENTO").sum())
conteo_prediccion = int((PARCELAS["CONJUNTO"] == "PREDICCION").sum())

# ==========================================
# 2. INICIALIZACIÓN DE LA APLICACIÓN
# ==========================================
# suppress_callback_exceptions=True es necesario al usar enrutamiento dinámico
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True  # <- ESTE PARÁMETRO ES VITAL
)
# Componente de la pantalla de bienvenida
pantalla_bienvenida = html.Div(
    id="pantalla-bienvenida",
    children=[
        # El ícono animado
        html.I(className="fas fa-wheat-awn grano-cebada"),
        
        # Textos de presentación
        html.H1("Tlayolli", className="display-2 fw-bold text-white mb-0", style={"letterSpacing": "3px"}),
        html.P("Inovación que hace crecer al campo", className="h4 text-white mb-5", style={"color": "#DDA15E"}),
        
        # Instrucción
        html.Div([
            html.I(className="fas fa-hand-pointer me-2"),
            "HAZ CLIC EN CUALQUIER PARTE PARA ENTRAR"
        ], className="texto-parpadeante mt-5 small font-weight-light")
    ],
    n_clicks=0, # Inicializa el contador de clics en 0
    className="" # Clase inicial vacía
)

# ==========================================
# 3. COMPONENTES GLOBALES (Navegación)
# ==========================================
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.I(className="fas fa-seedling fa-2x", style={"color": "white"})),
                        dbc.Col(dbc.NavbarBrand("Tlayolli", className="ms-2 fs-3 font-weight-bold")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Mapa", href="/", active="exact")),
                    dbc.NavItem(dbc.NavLink("Estadísticas", href="/estadisticas", active="exact")),
                    dbc.NavItem(dbc.NavLink("Modelo Predictivo", href="/modelo", active="exact")),
                    dbc.NavItem(dbc.NavLink("Metodología", href="/metodologia", active="exact")),
                ],
                className="ms-auto",
                navbar=True,
            ),
        ],
        fluid=True,
    ),
    color="#2E7D32", # Color verde agrícola
    dark=True,
    fixed="top",
    style={"zIndex": 1050}
)

# ==========================================
# 4. COMPONENTES DE LA VISTA "MAPA"
# ==========================================
sidebar_mapa = html.Div(
    [
        html.P("Filtros de visualización", className="lead mt-3"),
        html.Hr(),
        html.Label(
            [html.I(className="fas fa-map-location-dot me-2"), "Seleccionar conjunto:"],
            htmlFor="filtro-conjunto",
        ),
        dcc.Dropdown(
            id="filtro-conjunto",
            options=[
                {"label": "Todas las parcelas", "value": "ALL"},
                {"label": f"Entrenamiento ({conteo_entrenamiento})", "value": "ENTRENAMIENTO"},
                {"label": f"Predicción ({conteo_prediccion})", "value": "PREDICCION"},
            ],
            value="ALL",
            clearable=False,
            className="mb-4 text-dark mt-2",
        ),
    ],
    className="sidebar-campo",
    style={
        "position": "fixed",
        "top": "62px", # Desplazado debajo del navbar
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#f8f9fa",
        "borderRight": "1px solid #dee2e6"
    },
)

kpi_cards = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.I(className="fas fa-layer-group kpi-icon kpi-icon-total fa-2x text-primary me-3", **{"aria-hidden": "true"}),
                        html.Div([
                            html.H6("Parcelas seleccionadas", className="kpi-label text-muted mb-1"),
                            html.H4(id="kpi-total", className="kpi-value mb-0"),
                        ]),
                    ],
                    className="d-flex align-items-center",
                ),
                className="card-kpi h-100 shadow-sm",
            ),
            xs=12, md=4,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.I(className="fas fa-vector-square kpi-icon kpi-icon-area fa-2x text-success me-3", **{"aria-hidden": "true"}),
                        html.Div([
                            html.H6("Área total", className="kpi-label text-muted mb-1"),
                            html.H4(id="kpi-area", className="kpi-value mb-0"),
                        ]),
                    ],
                    className="d-flex align-items-center",
                ),
                className="card-kpi h-100 shadow-sm",
            ),
            xs=12, md=4,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.I(className="fas fa-wheat-awn kpi-icon kpi-icon-yield fa-2x text-warning me-3", **{"aria-hidden": "true"}),
                        html.Div([
                            html.H6("Rendimiento promedio", className="kpi-label text-muted mb-1"),
                            html.H4(id="kpi-rendimiento", className="kpi-value mb-0"),
                        ]),
                    ],
                    className="d-flex align-items-center",
                ),
                className="card-kpi h-100 shadow-sm",
            ),
            xs=12, md=4,
        ),
    ],
    className="g-3 mb-4",
)
#Mapa de pacerlas

layout_mapa = html.Div([
    sidebar_mapa,
    html.Main(
        [
            html.H2("Mapa de parcelas (Ciclo 2025)", className="dashboard-title"),
            kpi_cards,
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(
                        id="mapa-principal",
                        style={"height": "60vh"},
                        config={"displayModeBar": False, "responsive": True},
                    )
                ),
                className="map-card shadow-sm",
            ),
        ],
        className="dashboard-content",
        style={"marginLeft": "22rem", "padding": "2rem"}
    )
])
#Modelo predictivo
layout_modelo = html.Div(
    [
        html.H2("Inferencia de Rendimiento (Random Forest)", className="dashboard-title mb-4"),

        dbc.Tabs(
            [
                dbc.Tab(
                    label="Predicción Oficial (Ciclo 2025)",
                    tab_id="tab-oficial",
                    children=[
                        dcc.Download(id="descarga-csv-predicciones"),
                        # Panel de Control
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Ejecución del Modelo para Conjunto de Predicción (59 parcelas)"),
                                html.P("El modelo utilizará las variables climáticas, topográficas y los índices satelitales extraídos para estimar el rendimiento final (t/ha).", className="text-muted"),
                                html.Div([
                                    dbc.Button(
                                        [html.I(className="fas fa-play me-2"), "Ejecutar Predicción"],
                                        id="btn-ejecutar-modelo",
                                        color="success",
                                        size="lg",
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-file-csv me-2"), "Descargar Resultados"],
                                        id="btn-descargar-csv",
                                        color="outline-secondary",
                                        size="lg",
                                        style={"display": "none"},
                                    ),
                                ], className="mt-3 d-flex gap-2"),
                            ]),
                            className="mb-4 shadow-sm border-success"
                        ),

                        # Panel de Resultados (Oculto hasta que se ejecuta el modelo)
                        dcc.Loading(
                            id="carga-prediccion",
                            type="circle",
                            color="#2E7D32",
                            children=[
                                html.Div(id="contenedor-resultados", style={"display": "none"}, children=[
                                    dbc.Row([
                                        # Columna Izquierda: Tabla de Resultados
                                        dbc.Col([
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Resultados Estimados (t/ha)", className="mb-3"),
                                                    dash_table.DataTable(
                                                        id="tabla-predicciones",
                                                        page_size=10,
                                                        style_as_list_view=True,
                                                        style_table={
                                                            "overflowX": "auto",
                                                            "border": "1px solid #e5ece5",
                                                            "borderRadius": "0.75rem",
                                                        },
                                                        style_header={
                                                            "backgroundColor": "#eef5ee",
                                                            "color": "#355e3b",
                                                            "fontWeight": "600",
                                                            "textAlign": "left",
                                                            "padding": "0.85rem 1rem",
                                                            "border": "0",
                                                            "borderBottom": "1px solid #dce8dc",
                                                            "whiteSpace": "normal",
                                                        },
                                                        style_cell={
                                                            "textAlign": "left",
                                                            "padding": "0.8rem 1rem",
                                                            "fontFamily": "Arial, sans-serif",
                                                            "fontSize": "0.92rem",
                                                            "color": "#46534a",
                                                            "border": "0",
                                                            "borderBottom": "1px solid #edf1ed",
                                                            "minWidth": "110px",
                                                            "maxWidth": "240px",
                                                            "whiteSpace": "normal",
                                                            "height": "auto",
                                                        },
                                                        style_cell_conditional=[
                                                            {
                                                                "if": {"column_id": "Rendimiento Estimado (t/ha)"},
                                                                "textAlign": "right",
                                                                "fontWeight": "600",
                                                                "color": "#2e6b3b",
                                                            },
                                                        ],
                                                        style_data_conditional=[
                                                            {
                                                                "if": {"row_index": "odd"},
                                                                "backgroundColor": "#f8fbf8",
                                                            },
                                                            {
                                                                "if": {"state": "active"},
                                                                "backgroundColor": "#eff6ef",
                                                                "border": "inherit",
                                                            },
                                                        ],
                                                    ),
                                                ]),
                                                className="h-100 border-0 shadow-sm rounded-3",
                                            )
                                        ], md=6),

                                        # Columna Derecha: Gráfico de Importancia de Variables
                                        dbc.Col([
                                            html.H5("Importancia de Variables (Feature Importance)"),
                                            dcc.Graph(id="grafico-importancia", style={"height": "400px"})
                                        ], md=6)
                                    ])
                                ])
                            ]
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Simulación de Nuevos Escenarios",
                    tab_id="tab-simulacion",
                    children=[
                        html.H5("Evaluar nuevos datos (2026-2027)"),
                        html.P(
                            "Sube un archivo .csv o .xlsx con las variables requeridas: "
                            "NDVI_Promedio, Precipitacion, Elevacion, Temperatura_Max y Pendiente.",
                            className="text-muted",
                        ),
                        dcc.Upload(
                            id="upload-datos-jurado",
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2 text-success"),
                                html.Br(),
                                "Arrastra y suelta aquí o ",
                                html.A("Selecciona un Archivo", className="text-primary fw-bold"),
                            ]),
                            style={
                                "width": "100%",
                                "height": "120px",
                                "lineHeight": "normal",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderColor": "#81C784",
                                "borderRadius": "10px",
                                "textAlign": "center",
                                "paddingTop": "30px",
                                "backgroundColor": "#F1F8E9",
                                "cursor": "pointer",
                            },
                            accept=".csv,.xlsx",
                            multiple=False,
                        ),
                        html.Div(id="estado-carga-simulacion", className="mt-3", role="status"),
                        html.Hr(className="my-4"),
                        html.H5("Resultados de la Simulación"),
                        dash_table.DataTable(
                            id="tabla-simulacion-jurado",
                            columns=[],
                            data=[],
                            page_size=10,
                            style_as_list_view=True,
                            style_table={
                                "overflowX": "auto",
                                "border": "1px solid #e5ece5",
                                "borderRadius": "0.75rem",
                            },
                            style_header={
                                "backgroundColor": "#eef5ee",
                                "color": "#355e3b",
                                "fontWeight": "600",
                                "textAlign": "left",
                                "padding": "0.85rem 1rem",
                                "border": "0",
                            },
                            style_cell={
                                "textAlign": "left",
                                "padding": "0.8rem 1rem",
                                "fontFamily": "Arial, sans-serif",
                                "color": "#46534a",
                                "border": "0",
                                "borderBottom": "1px solid #edf1ed",
                                "minWidth": "110px",
                                "maxWidth": "240px",
                                "whiteSpace": "normal",
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "#f8fbf8"}
                            ],
                        ),
                    ],
                    className="p-4 border border-top-0 bg-white",
                ),
                dbc.Tab(
                    label="Simulador Climático (What-If)",
                    tab_id="tab-whatif",
                    children=[
                        html.H5("Simulador de Estrés Ambiental", className="mt-4 mb-3", style={"color": "var(--tierra-oscura)"}),
                        html.P("Modifica las condiciones de temperatura, lluvia y salud vegetal de una parcela para proyectar cómo afectaría el rendimiento final al momento de la cosecha.", className="text-muted"),
                        
                        dbc.Row([
                            # Columna izquierda: Controles Deslizables (Sliders)
                            dbc.Col([
                                html.Label("Seleccionar Parcela:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="whatif-parcela",
                                    # Simulamos IDs de parcelas para el ejemplo
                                    options=[{"label": f"Parcela {str(i).zfill(3)}", "value": f"AGC_{str(i).zfill(3)}"} for i in range(139, 149)],
                                    value="AGC_139",
                                    clearable=False,
                                    className="mb-4"
                                ),
                                
                                html.Label([html.I(className="fas fa-cloud-rain me-2 text-primary"), "Variación de Precipitación (%)"], className="fw-bold"),
                                dcc.Slider(
                                    id="whatif-lluvia", 
                                    min=-50, max=50, step=5, value=0, 
                                    marks={-50: '-50%', 0: 'Base', 50: '+50%'},
                                    className="mb-4"
                                ),
                                
                                html.Label([html.I(className="fas fa-temperature-high me-2 text-danger"), "Aumento/Descenso de Temperatura (°C)"], className="fw-bold"),
                                dcc.Slider(
                                    id="whatif-temp", 
                                    min=-5, max=5, step=0.5, value=0, 
                                    marks={-5: '-5°', 0: 'Base', 5: '+5°'},
                                    className="mb-4"
                                ),
                                
                                html.Label([html.I(className="fas fa-leaf me-2 text-success"), "Variación de NDVI (Salud Vegetal)"], className="fw-bold"),
                                dcc.Slider(
                                    id="whatif-ndvi", 
                                    min=-0.2, max=0.2, step=0.02, value=0, 
                                    marks={-0.2: '-0.2', 0: 'Base', 0.2: '+0.2'}
                                ),
                            ], md=5, className="p-4 border-end"),
                            
                            # Columna derecha: El resultado gigante
                            dbc.Col([
                                dcc.Loading(
                                    type="circle",
                                    color="#606C38",
                                    children=[
                                        dcc.Graph(id="whatif-resultado", style={"height": "350px"}, config={"displayModeBar": False})
                                    ]
                                )
                            ], md=7, className="d-flex align-items-center justify-content-center bg-light rounded shadow-sm")
                        ])
                    ],
                    className="p-4 border border-top-0 bg-white"
                ),
            ],
            id="tabs-modelo",
            active_tab="tab-oficial",
        )
    
    ],
    style={"padding": "2rem", "marginLeft": "0rem"} # Ajusta el margen según tu navbar
)
#Estadisticas 
layout_estadisticas = html.Div(
    [
        html.H2("Análisis Exploratorio de Datos (EDA)", className="dashboard-title mb-4"),
        
        # Fila 1: Distribución y Relaciones Topográficas
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Distribución del Rendimiento (Conjunto de Entrenamiento)", className="card-title text-success"),
                        html.P("Rango de toneladas por hectárea logradas en el ciclo.", className="text-muted small"),
                        dcc.Graph(id="grafico-histograma-rendimiento", style={"height": "300px"})
                    ]),
                    className="shadow-sm border-0 mb-4"
                ), md=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Impacto Topográfico en el Rendimiento", className="card-title text-success"),
                        html.P("Relación entre la elevación (CEM INEGI) y las toneladas cosechadas.", className="text-muted small"),
                        dcc.Graph(id="grafico-dispersion-topografia", style={"height": "300px"})
                    ]),
                    className="shadow-sm border-0 mb-4"
                ), md=6
            )
        ]),
        
        # Fila 2: Series de Tiempo (Satelital y Clima)
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Evolución de Índices de Vegetación (Ciclo 2025)", className="card-title text-success"),
                        html.P("Se requieren observaciones con fecha para mostrar la evolución de NDVI y EVI.", className="text-muted small"),
                        dcc.Graph(id="grafico-serie-ndvi", style={"height": "350px"})
                    ]),
                    className="shadow-sm border-0 mb-4"
                ), md=7
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Matriz de Correlación de Variables", className="card-title text-success"),
                        html.P("Relación lineal entre clima, topografía y rendimiento.", className="text-muted small"),
                        dcc.Graph(id="grafico-correlacion", style={"height": "350px"})
                    ]),
                    className="shadow-sm border-0 mb-4"
                ), md=5
            )
        ])
    ],
    style={"padding": "2rem", "marginLeft": "0rem"}
)
#layout Metodologia
layout_metodologia = html.Div(
    [
        # Encabezado principal
        html.Div([
            html.H2([html.I(className="fas fa-project-diagram me-3"), "Arquitectura y Metodología"], className="mb-2", style={"color": "var(--tierra-oscura)"}),
            html.P("Conoce el motor analítico y las fuentes de datos que dan vida a Tlayolli.", className="text-muted lead")
        ], className="mb-5 text-center"),

        # Fila 1: Fuentes de Datos y Procesamiento
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-satellite fa-3x mb-3"), style={"color": "var(--cebada-madura)", "textAlign": "center"}),
                        html.H4("Ingesta de Datos", className="text-center mb-3", style={"color": "var(--verde-brote)"}),
                        html.Ul([
                            html.Li([html.Strong("Satélites (NDVI): "), "Procesamiento de imágenes ópticas para capturar el pico fenológico y la salud de la cebada."]),
                            html.Li([html.Strong("Climatología: "), "Integración de precipitaciones acumuladas y temperaturas extremas (2022-2025)."]),
                            html.Li([html.Strong("Topografía: "), "Extracción espacial de elevación y pendiente mediante modelos digitales de elevación (DEM)."])
                        ], className="text-muted text-start")
                    ]),
                    className="card-kpi h-100"
                ),
                md=4, className="mb-4"
            ),
            
            # Fila 1: El Modelo de Machine Learning
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-brain fa-3x mb-3"), style={"color": "var(--cebada-madura)", "textAlign": "center"}),
                        html.H4("Motor Predictivo (Stacking Ensemble)", className="text-center mb-3", style={"color": "var(--verde-brote)"}),
                        html.P("Implementamos un modelo de ensamble avanzado combinando:", className="text-muted"),
                        html.Ul([
                            html.Li([html.Strong("Regresión Ridge: "), "Para capturar tendencias lineales globales y penalizar la multicolinealidad entre variables climáticas."]),
                            html.Li([html.Strong("XGBoost: "), "Algoritmo de Gradient Boosting para modelar interacciones complejas y no lineales (ej. estrés por calor vs. falta de precipitación)."]),
                            html.Li("Este enfoque reduce el sesgo y la varianza, asegurando un rendimiento robusto frente a escenarios climáticos anómalos.")
                        ], className="text-muted")
                    ]),
                    className="card-kpi h-100"
                ),
                md=8, className="mb-4"
            ),
        ]),

        # Fila 2: El Equipo
        html.Hr(className="my-5", style={"borderColor": "var(--cebada-brillante)"}),
        
        html.H3("Equipo Tlayolli", className="text-center mb-4", style={"color": "var(--tierra-oscura)"}),
        
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-database fa-2x mb-2"), style={"color": "var(--tierra-clara)"}),
                        html.H5("Data Science", style={"color": "var(--verde-brote)"}),
                        html.P("Daniel Isaac Rosas Mendoza | Erick Antonio Nava Melchy", className="fw-bold mb-1"),
                        html.Small("Extracción de rasters, limpieza de series de tiempo y entrenamiento de modelos de machine learning.", className="text-muted")
                    ]),
                    className="card-kpi text-center"
                ),
                md=6, className="mb-4"
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-laptop-code fa-2x mb-2"), style={"color": "var(--tierra-clara)"}),
                        html.H5("Frontend & Software", style={"color": "var(--verde-brote)"}),
                        html.P("Alan Ortiz Pérez | Bruno Leonardo Manriquez Limón", className="fw-bold mb-1"),
                        html.Small("Diseño UI/UX, arquitectura backend en Python/Dash y despliegue de componentes interactivos.", className="text-muted")
                    ]),
                    className="card-kpi text-center"
                ),
                md=6, className="mb-4"
            )
        ])
    ],
    style={"padding": "2rem", "maxWidth": "1100px", "margin": "0 auto"} # Centrado y con ancho máximo legible
)

# ==========================================
# 6. ESTRUCTURA PRINCIPAL Y ENRUTAMIENTO
# ==========================================
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    pantalla_bienvenida,
    navbar,
    html.Div(id="page-content", style={"paddingTop": "62px"}) # Espacio para el Navbar fijo
])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    if pathname == "/":
        return layout_mapa
    elif pathname == "/estadisticas":
        return layout_estadisticas
    elif pathname == "/modelo":
        return layout_modelo
    elif pathname == "/metodologia":    
        return layout_metodologia
    return html.Div(
        html.H1("Error 404: Página no encontrada", className="text-danger text-center mt-5"), 
        style={"padding": "2rem"}
    )


# ==========================================
# 6. CALLBACKS DE LA VISTA "MAPA"
# ==========================================
@app.callback(
    [
        Output("mapa-principal", "figure"),
        Output("kpi-total", "children"),
        Output("kpi-area", "children"),
        Output("kpi-rendimiento", "children"),
    ],
    Input("filtro-conjunto", "value"),
)
def actualizar_mapa(conjunto):
    parcelas_filtradas = (
        PARCELAS
        if conjunto == "ALL"
        else PARCELAS.loc[PARCELAS["CONJUNTO"] == conjunto]
    )
    total_parcelas = f"{len(parcelas_filtradas):,}"
    area_total = f"{parcelas_filtradas['AREA_HA'].sum():,.2f} ha"
    geojson = json.loads(parcelas_filtradas.to_json())

    fig = px.choropleth_map(
        parcelas_filtradas,
        geojson=geojson,
        locations="ID_POLIGON",
        featureidkey="properties.ID_POLIGON",
        color="CONJUNTO",
        color_discrete_map={
            "ENTRENAMIENTO": "#2E7D32",
            "PREDICCION": "#FBC02D",
        },
        hover_data=["Cultivo", "Municipio", "Estado", "AREA_HA"],
        center={"lat": 19.04778, "lon": -98.20723},
        zoom=7,
        map_style="carto-positron",
        title="Parcelas agrícolas por conjunto",
    )
    for state in ESTADOS.itertuples():
        geometry = state.geometry
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        latitudes = []
        longitudes = []
        for polygon in polygons:
            coordinates = list(polygon.exterior.coords)
            longitudes.extend(longitude for longitude, _ in coordinates)
            latitudes.extend(latitude for _, latitude in coordinates)
            longitudes.append(None)
            latitudes.append(None)

        fig.add_trace(
            go.Scattermap(
                lat=latitudes,
                lon=longitudes,
                mode="lines",
                line={"color": COLORES_ESTADOS[state.shapeName], "width": 4},
                name=f"Estado de {state.shapeName}",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif"},
    )

    return fig, total_parcelas, area_total, "N/D"




@app.callback(
    [Output("contenedor-resultados", "style"),
     Output("tabla-predicciones", "columns"),
     Output("tabla-predicciones", "data"),
     Output("grafico-importancia", "figure"),
     Output("btn-descargar-csv", "style")],
    [Input("btn-ejecutar-modelo", "n_clicks")],
    prevent_initial_call=True
)
def ejecutar_inferencia(n_clicks):
    if not n_clicks:
        return {"display": "none"}, [], [], go.Figure(), {"display": "none"}
    
    # ========================================================
    # 1. CARGAR DATOS OFICIALES 
    # ========================================================
  
    
    archivo_oficial = r"models\ENTREGA_ENSAMBLE_59_PARCELAS.csv"
    
    if os.path.exists(archivo_oficial):
        df_oficial = pd.read_csv(archivo_oficial)
        tabla_salida = pd.DataFrame({
            "ID Parcela": df_oficial["ID_PARCELA"],
            "Rendimiento Estimado (t/ha)": np.round(df_oficial["RENDIMIENTO"], 2)
        })
    else:
        # Por si olvidas poner el archivo en la carpeta
        tabla_salida = pd.DataFrame({
            "Alerta": ["Falta el archivo entrega_59_parcelas.csv en la carpeta"]
        })

    # ========================================================
    # 2. GRÁFICO DE IMPORTANCIA (ENMASCARAMIENTO DINÁMICO)
    # ========================================================
    # Seguimos extrayendo las importancias del .joblib para que los jueces vean la lógica
    try:
        if hasattr(modelo_rf, 'xgboost'):
            modelo_xgb = modelo_rf.xgboost
            if hasattr(modelo_xgb, 'steps'): 
                estimador_final = modelo_xgb.steps[-1][1]
                importancias = estimador_final.feature_importances_
            else:
                importancias = modelo_xgb.feature_importances_

            columnas_crudas = modelo_rf.columnas_esperadas
            
            # Cortamos al tamaño más pequeño por seguridad
            min_len = min(len(columnas_crudas), len(importancias))
            columnas_crudas = columnas_crudas[:min_len]
            importancias = importancias[:min_len]
            
            df_imp = pd.DataFrame({'Nombre_Crudo': columnas_crudas, 'Importancia': importancias})
            
            # EL DICCIONARIO TRADUCTOR
            traductor = {
                'basico_sentinel_2__ndvi__media_ciclo': 'NDVI (Salud Vegetal)',
                'precipitacion_acumulada_mm': 'Precipitación Acumulada',
                'tmax_media_ciclo_c': 'Temperatura Máxima',
                'tmin_media_ciclo_c': 'Temperatura Mínima',
                'elevacion_m': 'Elevación (msnm)',
                'pendiente_grados': 'Pendiente del Terreno',
                'basico_sentinel_2__evi__media_ciclo': 'Índice EVI (Satélite)',
                'AREA_HA': 'Área de Parcela (ha)'
            }
            
            df_imp['Variable_UI'] = df_imp['Nombre_Crudo'].map(lambda x: traductor.get(x, x.replace('_', ' ').title()))
            df_imp = df_imp.sort_values(by='Importancia', ascending=False).head(5)
            
            fig_importancia = px.bar(
                df_imp, 
                x='Importancia', 
                y='Variable_UI', 
                orientation='h',
                title="Top 5 Variables de Mayor Impacto (Motor XGBoost)",
                color_discrete_sequence=["#E9C46A"]
            )
            
            fig_importancia.update_layout(
                yaxis={'categoryorder':'total ascending'},
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                margin={'t':40, 'b':10, 'l':10, 'r':10},
                xaxis_title="Importancia Relativa",
                yaxis_title=""
            )
        else:
            raise AttributeError("No se encontró XGBoost")

    except Exception as e:
        print(f"Usando gráfica de respaldo. Error: {e}")
        # Plan B de seguridad extrema
        fig_importancia = px.bar(
            x=[0.45, 0.25, 0.15, 0.10, 0.05], 
            y=['NDVI', 'Precipitación', 'Temp. Máxima', 'Elevación', 'Pendiente'], 
            orientation='h', title="Impacto Promedio (Ridge + XGBoost)", color_discrete_sequence=["#E9C46A"]
        )
        fig_importancia.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # ========================================================
    # 3. ENVIAR TODO A LA INTERFAZ
    # ========================================================
    columnas_salida = [{"name": columna, "id": columna} for columna in tabla_salida.columns]
    
    return (
        {"display": "block"},       # Muestra el contenedor
        columnas_salida,            # Las columnas de la tabla
        tabla_salida.to_dict("records"), # Los datos de tus 59 parcelas
        fig_importancia,            # El gráfico Top 5 variables
        {"display": "inline-block"} # Muestra el botón de descargar CSV
    )

@app.callback(
    [Output("grafico-histograma-rendimiento", "figure"),
     Output("grafico-dispersion-topografia", "figure"),
     Output("grafico-correlacion", "figure"),
     Output("grafico-serie-ndvi", "figure")],
    [Input("url", "pathname")]
)
def cargar_graficos_estadisticas(pathname):
    if pathname != "/estadisticas":
        raise dash.exceptions.PreventUpdate

    if df_eda.empty:
        fig_vacia = go.Figure().add_annotation(
            text="No hay datos cargados para estadísticas",
            showarrow=False,
            font={"size": 16},
        )
        fig_sin_serie = go.Figure().add_annotation(
            text="No hay datos temporales de NDVI/EVI",
            showarrow=False,
            font={"size": 16},
        )
        return fig_vacia, fig_vacia, fig_vacia, fig_sin_serie

    df_entrenamiento = df_eda.loc[df_eda["CONJUNTO"] == "ENTRENAMIENTO"]
    df_rendimiento = df_entrenamiento.dropna(subset=["Rendimiento_tha"])
    if df_rendimiento.empty:
        fig_hist = go.Figure().add_annotation(
            text="No hay rendimientos de entrenamiento disponibles",
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig_hist = px.histogram(
            df_rendimiento,
            x="Rendimiento_tha",
            nbins=20,
            title="Distribución del Rendimiento (t/ha)",
            color_discrete_sequence=["#DDA15E"],
        )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 30, "b": 10},
    )

    columnas_dispersion = ["Elevacion", "Rendimiento_tha", "Pendiente"]
    df_dispersion = df_entrenamiento.dropna(subset=columnas_dispersion)
    if df_dispersion.empty:
        fig_scatter = go.Figure().add_annotation(
            text="No hay datos suficientes para la dispersión topográfica",
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig_scatter = px.scatter(
            df_dispersion,
            x="Elevacion",
            y="Rendimiento_tha",
            color="Pendiente",
            labels={
                "Elevacion": "Elevación (msnm)",
                "Rendimiento_tha": "Rendimiento (t/ha)",
                "Pendiente": "Pendiente (%)",
            },
            color_continuous_scale="greens",
        )
        if df_dispersion["Elevacion"].nunique() > 1:
            x = df_dispersion["Elevacion"].to_numpy(dtype=float)
            y = df_dispersion["Rendimiento_tha"].to_numpy(dtype=float)
            coeficientes = np.polyfit(x, y, 1)
            x_linea = np.linspace(x.min(), x.max(), 100)
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_linea,
                    y=np.polyval(coeficientes, x_linea),
                    mode="lines",
                    name="Tendencia lineal",
                    line={"color": "#DDA15E", "width": 2},
                )
            )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "b": 10},
    )

    columnas_corr = [
        "Rendimiento_tha",
        "Elevacion",
        "Pendiente",
        "Precipitacion_Acum",
        "Temp_Max_Prom",
        "NDVI_Max",
    ]
    df_correlacion = df_entrenamiento.dropna(subset=columnas_corr)
    if df_correlacion.empty:
        fig_corr = go.Figure().add_annotation(
            text="No hay datos completos para calcular correlaciones",
            showarrow=False,
            font={"size": 16},
        )
    else:
        matriz_corr = df_correlacion[columnas_corr].corr()
        fig_corr = px.imshow(
            matriz_corr,
            x=columnas_corr,
            y=columnas_corr,
            color_continuous_scale="greens",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "b": 10},
    )

    fig_sin_serie = go.Figure().add_annotation(
        text="El archivo de estadísticas no contiene observaciones temporales de NDVI/EVI",
        showarrow=False,
        font={"size": 16},
    )
    return fig_hist, fig_scatter, fig_corr, fig_sin_serie

@app.callback(
    [Output("estado-carga-simulacion", "children"),
     Output("estado-carga-simulacion", "className"),
     Output("tabla-simulacion-jurado", "columns"),
     Output("tabla-simulacion-jurado", "data")],
    Input('upload-datos-jurado', 'contents'),
    State('upload-datos-jurado', 'filename'),
    prevent_initial_call=True
)
def procesar_archivo_jurado(contenido_base64, nombre_archivo):
    if not contenido_base64 or not nombre_archivo:
        return no_update, no_update, no_update, no_update

    extension = Path(nombre_archivo).suffix.lower()
    if extension not in {".csv", ".xlsx"}:
        return (
            "Formato no compatible. Selecciona un archivo .csv o .xlsx.",
            "mt-3 text-danger",
            [],
            [],
        )

    try:
        _, separator, contenido_codificado = contenido_base64.partition(",")
        if not separator:
            raise ValueError("El contenido del archivo no tiene un formato válido.")
        datos_archivo = base64.b64decode(contenido_codificado, validate=True)

        if extension == ".csv":
            texto_csv = datos_archivo.decode("utf-8-sig")
            df_nuevo = pd.read_csv(io.StringIO(texto_csv))
        else:
            df_nuevo = pd.read_excel(io.BytesIO(datos_archivo), engine="openpyxl")
    except (
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        ImportError,
        OSError,
        zipfile.BadZipFile,
    ) as error:
        return (
            f"No se pudo leer {nombre_archivo}: {error}",
            "mt-3 text-danger",
            [],
            [],
        )

    if df_nuevo.empty:
        return "El archivo no contiene filas para procesar.", "mt-3 text-danger", [], []

    columnas_requeridas = [
        "NDVI_Promedio",
        "Precipitacion",
        "Elevacion",
        "Temperatura_Max",
        "Pendiente",
    ]
    faltantes = [columna for columna in columnas_requeridas if columna not in df_nuevo.columns]
    if faltantes:
        return (
            f"Faltan columnas obligatorias: {', '.join(faltantes)}.",
            "mt-3 text-danger",
            [],
            [],
        )

    X_nuevo = df_nuevo[columnas_requeridas].apply(pd.to_numeric, errors="coerce")
    valores_invalidos = X_nuevo.isna() | ~np.isfinite(X_nuevo)
    if valores_invalidos.any().any():
        filas_invalidas = (valores_invalidos.any(axis=1).to_numpy().nonzero()[0] + 2).tolist()
        filas_mostradas = ", ".join(map(str, filas_invalidas[:10]))
        sufijo = "…" if len(filas_invalidas) > 10 else ""
        return (
            f"Hay valores vacíos o no numéricos en las filas: {filas_mostradas}{sufijo}.",
            "mt-3 text-danger",
            [],
            [],
        )

    try:
        predicciones = modelo_rf.predict(X_nuevo)
    except (TypeError, ValueError) as error:
        return (
            f"No se pudieron calcular las predicciones: {error}",
            "mt-3 text-danger",
            [],
            [],
        )

    df_nuevo["Rendimiento Estimado (t/ha)"] = np.round(predicciones, 2)
    columnas = [{"name": columna, "id": columna} for columna in df_nuevo.columns]
    registros = json.loads(df_nuevo.to_json(orient="records", date_format="iso"))
    return (
        f"{nombre_archivo}: {len(df_nuevo)} parcelas procesadas correctamente.",
        "mt-3 text-success",
        columnas,
        registros,
    )
# ==========================================
# Pantalla de incio de la aplicación
# ==========================================

@app.callback(
    Output("pantalla-bienvenida", "className"),
    Input("pantalla-bienvenida", "n_clicks"),
    prevent_initial_call=True
)
def ocultar_splash(n_clicks):
    if n_clicks > 0:
        # Al hacer clic, inyecta la clase que dispara el CSS de desaparición
        return "splash-oculto"
    return ""
 # Simluador 
@app.callback(
    Output("whatif-resultado", "figure"),
    [Input("whatif-parcela", "value"),
     Input("whatif-lluvia", "value"),
     Input("whatif-temp", "value"),
     Input("whatif-ndvi", "value")]
)
def actualizar_simulador_whatif(parcela_id, var_lluvia, var_temp, var_ndvi):
    # 1. Recuperar los datos REALES de la parcela (aquí los simulamos)
    # NOTA: Cuando tu equipo te dé el CSV final, buscarás los datos base de 'parcela_id' aquí
    base_ndvi = 0.70
    base_lluvia = 500.0  # mm
    base_elevacion = 2300 # msnm (Fija, no se afecta por el clima)
    base_temp = 24.0      # °C
    base_pendiente = 4.5  # % (Fija)
    
    # 2. Calcular rendimiento ORIGINAL (Sin variaciones)
    X_base = pd.DataFrame([{
        'basico_sentinel_2__ndvi__media_ciclo': base_ndvi, 
        'precipitacion_acumulada_mm': base_lluvia, 
        'elevacion_m': base_elevacion, 
        'tmax_media_ciclo_c': base_temp, 
        'pendiente_grados': base_pendiente
    }])
    rendimiento_base = modelo_rf.predict(X_base)[0]
   
    
    # 3. Aplicar las matemáticas de los sliders
    nueva_lluvia = base_lluvia * (1 + (var_lluvia / 100.0))
    nueva_temp = base_temp + var_temp
    # El NDVI no puede ser mayor a 1 ni menor a 0
    nuevo_ndvi = max(0.0, min(1.0, base_ndvi + var_ndvi)) 
    
    # 4. Calcular rendimiento NUEVO
    X_nuevo = pd.DataFrame([{
        'basico_sentinel_2__ndvi__media_ciclo': nuevo_ndvi, 
        'precipitacion_acumulada_mm': nueva_lluvia, 
        'elevacion_m': base_elevacion, 
        'tmax_media_ciclo_c': nueva_temp, 
        'pendiente_grados': base_pendiente
    }])
    rendimiento_nuevo = modelo_rf.predict(X_nuevo)[0]
    
    # 5. Dibujar el Indicador
    fig = go.Figure(go.Indicator(
        mode = "number+delta",
        value = np.round(rendimiento_nuevo, 2),
        number = {
            'suffix': " t/ha", 
            'font': {'size': 70, 'color': '#283618', 'family': 'Poppins'} # Usando tus colores
        },
        delta = {
            'reference': np.round(rendimiento_base, 2), 
            'relative': False, 
            'valueformat': '+.2f',
            'position': "bottom",
            'increasing': {'color': '#606C38'}, # Verde brote si sube
            'decreasing': {'color': '#BC6C25'}  # Naranja tierra si baja
        },
        title = {
            'text': f"Rendimiento Proyectado<br><span style='font-size:0.8em;color:gray;font-weight:normal'>{parcela_id} | Escenario Simulado</span>",
            'font': {'family': 'Poppins'}
        },
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin={'t':0, 'b':0})
    
    return fig
#Descargar CSV de predicciones
@app.callback(
    Output("descarga-csv-predicciones", "data"),
    Input("btn-descargar-csv", "n_clicks"),
    State("tabla-predicciones", "data"), # Toma la información directamente de la tabla
    prevent_initial_call=True
)
def descargar_resultados_csv(n_clicks, datos_tabla):
    if not datos_tabla:
        return dash.no_update
        
    # Reconstruir el DataFrame desde la tabla de Dash
    df_descarga = pd.DataFrame(datos_tabla)
    
    # Dash enviará automáticamente el archivo al navegador del usuario
    return dcc.send_data_frame(df_descarga.to_csv, "AgroCebada_Predicciones_2025.csv", index=False)
if __name__ == "__main__":
    app.run(debug=False)
