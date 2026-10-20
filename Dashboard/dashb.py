import base64
import io
import json
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

# ==========================================
# 1. PREPARACIÓN DE DATOS ESPACIALES
# ==========================================
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
# 1.B CARGA DE DATOS PARQUET (SATELITAL, CLIMA Y TOPOGRAFÍA)
# ==========================================
PARQUET_DIR = Path(__file__).resolve().parent / "Parquet"

DF_TOPO = pd.DataFrame()
DF_SERIES = pd.DataFrame()

for pq_file in PARQUET_DIR.glob("*.parquet"):
    df_temp = pd.read_parquet(pq_file)
    if "time" in df_temp.columns:
        DF_SERIES = df_temp.copy()
    elif {"Elevacion", "Pendiente"}.issubset(df_temp.columns):
        DF_TOPO = df_temp.copy()

if DF_SERIES.empty:
    raise ValueError(f"No se encontró el archivo Parquet con la columna 'time' en: {PARQUET_DIR}")

# Procesar fechas y años para agrupación
DF_SERIES["time"] = pd.to_datetime(DF_SERIES["time"])
DF_SERIES["ANIO"] = DF_SERIES["time"].dt.year
DF_SERIES["FECHA_STR"] = DF_SERIES["time"].dt.strftime("%Y-%m-%d")

# Unir Elevación y Pendiente al GeoDataFrame base de PARCELAS si está disponible
if not DF_TOPO.empty:
    PARCELAS = PARCELAS.merge(
        DF_TOPO[["ID_POLIGON", "Elevacion", "Pendiente"]],
        on="ID_POLIGON",
        how="left",
    )

ANIOS_DISPONIBLES = sorted(DF_SERIES["ANIO"].dropna().unique().tolist())

# Diccionario de variables disponibles en tus Parquet para el Mapa
VARIABLES_MAPA = {
    "CONJUNTO": {"label": "Conjunto (Entrenamiento / Predicción)", "tipo": "discreto"},
    # Índices Satelitales (Parquet temporal)
    "ndvi_promedio": {"label": "Satelital: NDVI Promedio", "tipo": "temporal", "escala": "YlGn"},
    "ndvi_promedio_pro": {"label": "Satelital: NDVI Suavizado (Pro)", "tipo": "temporal", "escala": "YlGn"},
    "evi_promedio": {"label": "Satelital: EVI Promedio", "tipo": "temporal", "escala": "YlGn"},
    "evi_promedio_pro": {"label": "Satelital: EVI Suavizado (Pro)", "tipo": "temporal", "escala": "YlGn"},
    "savi_promedio": {"label": "Satelital: SAVI Promedio", "tipo": "temporal", "escala": "YlGn"},
    "ndwi_promedio": {"label": "Satelital: NDWI (Humedad)", "tipo": "temporal", "escala": "GnBu"},
    "lai_promedio": {"label": "Satelital: LAI (Área Foliar)", "tipo": "temporal", "escala": "Greens"},
    "fapar_promedio": {"label": "Satelital: FAPAR", "tipo": "temporal", "escala": "YlGn"},
    "msi_promedio": {"label": "Satelital: MSI (Estrés Hídrico)", "tipo": "temporal", "escala": "YlOrBr"},
    # Clima (Parquet temporal)
    "PREC": {"label": "Clima: Precipitación (mm)", "tipo": "temporal", "escala": "Blues"},
    "Tmax": {"label": "Clima: Temperatura Máxima (°C)", "tipo": "temporal", "escala": "OrRd"},
    "Tmin": {"label": "Clima: Temperatura Mínima (°C)", "tipo": "temporal", "escala": "PuBu"},
    # Topografía (Parquet estático)
    "Elevacion": {"label": "Topografía: Elevación (m s.n.m.)", "tipo": "estatico", "escala": "Viridis"},
    "Pendiente": {"label": "Topografía: Pendiente (grados)", "tipo": "estatico", "escala": "Magma"},
}

# ==========================================
# 2. INICIALIZACIÓN DE LA APLICACIÓN
# ==========================================
# suppress_callback_exceptions=True es necesario al usar enrutamiento dinámico
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True  # <- ESTE PARÁMETRO ES VITAL
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
            className="mb-3 text-dark mt-1",
        ),
        html.Label(
            [html.I(className="fas fa-satellite me-2"), "Capa / Índice a visualizar:"],
            htmlFor="filtro-variable",
        ),
        dcc.Dropdown(
            id="filtro-variable",
            options=[{"label": info["label"], "value": col} for col, info in VARIABLES_MAPA.items()],
            value="ndvi_promedio",
            clearable=False,
            className="mb-3 text-dark mt-1",
        ),
        # Contenedor que se muestra solo cuando la variable es temporal (Satelital / Clima)
        html.Div(
            id="contenedor-tiempo",
            children=[
                html.Hr(),
                html.Label(
                    [html.I(className="fas fa-calendar-alt me-2"), "Agrupar por Año:"],
                    htmlFor="filtro-anio",
                ),
                dcc.Dropdown(
                    id="filtro-anio",
                    options=[{"label": str(anio), "value": anio} for anio in ANIOS_DISPONIBLES],
                    value=ANIOS_DISPONIBLES[-1] if ANIOS_DISPONIBLES else None,
                    clearable=False,
                    className="mb-3 text-dark mt-1",
                ),
                html.Label(
                    [html.I(className="fas fa-clock me-2"), "Fecha de captura:"],
                    htmlFor="filtro-fecha",
                ),
                dcc.Dropdown(
                    id="filtro-fecha",
                    options=[],
                    value=None,
                    clearable=False,
                    className="mb-2 text-dark mt-1",
                ),
                html.Small(
                    id="resumen-fechas-anio",
                    className="text-muted d-block mb-3",
                ),
            ],
        ),
    ],
    className="sidebar-campo",
    style={
        "position": "fixed",
        "top": "62px",
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "1.5rem 1rem",
        "backgroundColor": "#f8f9fa",
        "borderRight": "1px solid #dee2e6",
        "overflowY": "auto",
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
                    label="Parcelas actuales",
                    tab_id="tab-oficial",
                    children=[
                        # Panel de Control
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Ejecución del Modelo para Conjunto de Predicción (59 parcelas)"),
                                html.P("El modelo utilizará las variables climáticas, topográficas y los índices satelitales extraídos para estimar el rendimiento final (t/ha).", className="text-muted"),
                                dbc.Button(
                                    [html.I(className="fas fa-play me-2"), "Ejecutar Predicción"],
                                    id="btn-ejecutar-modelo",
                                    color="success",
                                    size="lg",
                                    className="mt-2"
                                ),
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
            ],
            id="tabs-modelo",
            active_tab="tab-oficial",
        )
    
    ],
    style={"padding": "2rem", "marginLeft": "0rem"} # Ajusta el margen según tu navbar
)
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
                        html.P("Curva promedio de NDVI y EVI extraída de Planet / Sentinel-2.", className="text-muted small"),
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

# ==========================================
# 6. ESTRUCTURA PRINCIPAL Y ENRUTAMIENTO
# ==========================================
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
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
    return html.Div(
        html.H1("Error 404: Página no encontrada", className="text-danger text-center mt-5"), 
        style={"padding": "2rem"}
    )


# ==========================================
# 6. CALLBACKS DE LA VISTA "MAPA"
# ==========================================
# ==========================================
# 6. CALLBACKS DE LA VISTA "MAPA"
# ==========================================

@app.callback(
    [
        Output("filtro-fecha", "options"),
        Output("filtro-fecha", "value"),
        Output("resumen-fechas-anio", "children"),
        Output("contenedor-tiempo", "style"),
    ],
    [
        Input("filtro-anio", "value"),
        Input("filtro-variable", "value"),
    ],
)
def actualizar_fechas_por_anio(anio, variable):
    tipo_var = VARIABLES_MAPA.get(variable, {}).get("tipo", "discreto")
    estilo_contenedor = {"display": "block"} if tipo_var == "temporal" else {"display": "none"}

    if anio is None or DF_SERIES.empty:
        return [], None, "Sin fechas disponibles", estilo_contenedor

    # Filtrar fechas del año seleccionado donde la variable tenga datos válidos
    df_anio = DF_SERIES[DF_SERIES["ANIO"] == anio]
    if variable in df_anio.columns:
        df_anio = df_anio[df_anio[variable].notna()]

    fechas_ordenadas = sorted(df_anio["FECHA_STR"].unique().tolist())
    opciones = [{"label": f, "value": f} for f in fechas_ordenadas]
    fecha_defecto = fechas_ordenadas[-1] if fechas_ordenadas else None
    resumen = f"{len(fechas_ordenadas)} capturas disponibles en {anio}"

    return opciones, fecha_defecto, resumen, estilo_contenedor


@app.callback(
    [
        Output("mapa-principal", "figure"),
        Output("kpi-total", "children"),
        Output("kpi-area", "children"),
        Output("kpi-rendimiento", "children"),
    ],
    [
        Input("filtro-conjunto", "value"),
        Input("filtro-variable", "value"),
        Input("filtro-anio", "value"),
        Input("filtro-fecha", "value"),
    ],
)
def actualizar_mapa(conjunto, variable, anio, fecha_str):
    parcelas_filtradas = (
        PARCELAS.copy()
        if conjunto == "ALL"
        else PARCELAS.loc[PARCELAS["CONJUNTO"] == conjunto].copy()
    )

    total_parcelas = f"{len(parcelas_filtradas):,}"
    area_total = f"{parcelas_filtradas['AREA_HA'].sum():,.2f} ha"
    info_var = VARIABLES_MAPA.get(variable, VARIABLES_MAPA["CONJUNTO"])
    tipo_var = info_var["tipo"]
    kpi_valor = "N/D"

    # Caso 1: Variable temporal del Parquet (Satelital o Clima)
    if tipo_var == "temporal" and fecha_str and variable in DF_SERIES.columns:
        df_corte = DF_SERIES[
            (DF_SERIES["ANIO"] == anio) & (DF_SERIES["FECHA_STR"] == fecha_str)
        ][["ID_POLIGON", variable, "PREC", "Tmax", "Tmin", "ndvi_promedio", "evi_promedio"]].drop_duplicates("ID_POLIGON")

        parcelas_filtradas = parcelas_filtradas.merge(df_corte, on="ID_POLIGON", how="left")
        promedio = parcelas_filtradas[variable].mean()
        kpi_valor = f"{promedio:.2f}" if pd.notna(promedio) else "Sin datos"

        geojson = json.loads(parcelas_filtradas.to_json())
        fig = px.choropleth_map(
            parcelas_filtradas,
            geojson=geojson,
            locations="ID_POLIGON",
            featureidkey="properties.ID_POLIGON",
            color=variable,
            color_continuous_scale=info_var.get("escala", "YlGn"),
            hover_data=["ID_POLIGON", "Municipio", "Estado", "AREA_HA", variable],
            center={"lat": 19.04778, "lon": -98.20723},
            zoom=7,
            map_style="carto-positron",
            title=f"{info_var['label']} — Fecha: {fecha_str} (Año {anio})",
        )

    # Caso 2: Variable estática del Parquet (Elevación o Pendiente)
    elif tipo_var == "estatico" and variable in parcelas_filtradas.columns:
        promedio = parcelas_filtradas[variable].mean()
        kpi_valor = f"{promedio:.1f}" if pd.notna(promedio) else "N/D"

        geojson = json.loads(parcelas_filtradas.to_json())
        fig = px.choropleth_map(
            parcelas_filtradas,
            geojson=geojson,
            locations="ID_POLIGON",
            featureidkey="properties.ID_POLIGON",
            color=variable,
            color_continuous_scale=info_var.get("escala", "Viridis"),
            hover_data=["ID_POLIGON", "Municipio", "Estado", "AREA_HA", variable],
            center={"lat": 19.04778, "lon": -98.20723},
            zoom=7,
            map_style="carto-positron",
            title=f"{info_var['label']} por parcela",
        )

    # Caso 3: Vista por Conjunto (Entrenamiento / Predicción)
    else:
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
            hover_data=["ID_POLIGON", "Cultivo", "Municipio", "Estado", "AREA_HA"],
            center={"lat": 19.04778, "lon": -98.20723},
            zoom=7,
            map_style="carto-positron",
            title="Parcelas agrícolas por conjunto",
        )

    # Contornos estatales
    for state in ESTADOS.itertuples():
        geometry = state.geometry
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        latitudes, longitudes = [], []
        for polygon in polygons:
            coordinates = list(polygon.exterior.coords)
            longitudes.extend(lon for lon, _ in coordinates)
            latitudes.extend(lat for _, lat in coordinates)
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

    return fig, total_parcelas, area_total, kpi_valor


    # ---------------------------------------------------------
# ZONA DE INTEGRACIÓN DEL MODELO REAL
# ---------------------------------------------------------
# CUANDO EL MODELO ESTÉ LISTO, BORRA LA CLASE MOCK Y USA:
# import joblib
# modelo_rf = joblib.load("random_forest_cebada.joblib")

class MockRandomForest:
    def predict(self, X):
        # Simula rendimientos entre 2.5 y 4.5 toneladas por hectárea
        return np.random.uniform(2.5, 4.5, len(X))
    
    @property
    def feature_importances_(self):
        # Simula los pesos que el RF le dio a cada variable
        pesos = np.random.rand(5)
        return pesos / pesos.sum()

modelo_rf = MockRandomForest()
# ---------------------------------------------------------

@app.callback(
    [Output("contenedor-resultados", "style"),
     Output("tabla-predicciones", "columns"),
     Output("tabla-predicciones", "data"),
     Output("grafico-importancia", "figure")],
    [Input("btn-ejecutar-modelo", "n_clicks")],
    prevent_initial_call=True
)
def ejecutar_inferencia(n_clicks):
    if not n_clicks:
        return {"display": "none"}, [], [], go.Figure()
    
    # 1. Obtener los IDs de las parcelas de predicción
    df_prediccion = PARCELAS[PARCELAS["CONJUNTO"] == "PREDICCION"].copy()
    
    # 2. AQUÍ CONECTARÁS TUS VARIABLES REALES (X)
    # Ejemplo: X_pred = df_caracteristicas[df_caracteristicas['ID'].isin(df_prediccion['ID_POLIGON'])]
    X_simulada = np.zeros((len(df_prediccion), 5)) # Matriz vacía solo para simular tamaño
    
    # 3. Ejecutar el modelo
    predicciones = modelo_rf.predict(X_simulada)
    
    # 4. Preparar tabla de salida
    tabla_salida = pd.DataFrame({
        "ID Parcela": df_prediccion["ID_POLIGON"],
        "Rendimiento Estimado (t/ha)": np.round(predicciones, 2)
    })
    
    # 5. Preparar gráfico de importancia de variables
    # Estas son las columnas que el equipo de DS usará para entrenar
    nombres_variables = ['NDVI Promedio', 'Precipitación Acum.', 'Elevación', 'Temp Máxima', 'Pendiente']
    importancia = modelo_rf.feature_importances_
    
    df_importancia = pd.DataFrame({'Variable': nombres_variables, 'Importancia': importancia})
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=True)
    
    fig_importancia = px.bar(
        df_importancia, 
        x='Importancia', 
        y='Variable', 
        orientation='h',
        color='Importancia',
        color_continuous_scale='Greens',
        title="¿Qué variables influyeron más en el modelo?"
    )
    fig_importancia.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    
    # Mostrar el contenedor y enviar los datos
    columnas_salida = [{"name": columna, "id": columna} for columna in tabla_salida.columns]
    return {"display": "block"}, columnas_salida, tabla_salida.to_dict("records"), fig_importancia

@app.callback(
    [
        Output("grafico-dispersion-topografia", "figure"),
        Output("grafico-correlacion", "figure"),
        Output("grafico-serie-ndvi", "figure"),
    ],
    [Input("url", "pathname")],
)
def cargar_graficos_estadisticas(pathname):
    if pathname != "/estadisticas":
        raise dash.exceptions.PreventUpdate

    # 1. Resumen por parcela combinando Topografía y Series de Tiempo
    resumen_parcela = (
        DF_SERIES.groupby("ID_POLIGON", as_index=False)
        .agg(
            NDVI_Max=("ndvi_promedio", "max"),
            NDVI_Mean=("ndvi_promedio", "mean"),
            Precipitacion_Acum=("PREC", "sum"),
            Tmax_Mean=("Tmax", "mean"),
        )
    )
    if not DF_TOPO.empty:
        resumen_parcela = resumen_parcela.merge(
            DF_TOPO[["ID_POLIGON", "Elevacion", "Pendiente", "Estado"]],
            on="ID_POLIGON",
            how="left",
        )

    # Gráfico 1: Dispersión Elevación vs. NDVI Máximo (coloreado por Pendiente)
    fig_scatter = px.scatter(
        resumen_parcela,
        x="Elevacion",
        y="NDVI_Max",
        color="Pendiente",
        hover_data=["ID_POLIGON", "Estado"],
        labels={
            "Elevacion": "Elevación (msnm)",
            "NDVI_Max": "NDVI Máximo alcanzado",
            "Pendiente": "Pendiente (°)",
        },
        color_continuous_scale="Viridis",
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "b": 10},
    )

    # Gráfico 2: Matriz de correlación real
    cols_corr = ["NDVI_Max", "Elevacion", "Pendiente", "Precipitacion_Acum", "Tmax_Mean"]
    cols_validas = [c for c in cols_corr if c in resumen_parcela.columns]
    corr_matrix = resumen_parcela[cols_validas].corr().round(2)

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="Greens",
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

    # Gráfico 3: Serie de tiempo de NDVI y EVI agrupada por Año y Fecha
    df_tiempo = (
        DF_SERIES.groupby(["ANIO", "time"], as_index=False)[["ndvi_promedio", "evi_promedio"]]
        .mean()
        .rename(columns={"ndvi_promedio": "NDVI", "evi_promedio": "EVI"})
        .melt(
            id_vars=["ANIO", "time"],
            value_vars=["NDVI", "EVI"],
            var_name="Índice",
            value_name="Valor",
        )
    )
    df_tiempo["ANIO"] = df_tiempo["ANIO"].astype(str)

    fig_ndvi = px.line(
        df_tiempo,
        x="time",
        y="Valor",
        color="ANIO",
        line_dash="Índice",
        labels={"time": "Fecha", "Valor": "Índice Promedio", "ANIO": "Año"},
    )
    fig_ndvi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "b": 10},
        hovermode="x unified",
    )

    return fig_scatter, fig_corr, fig_ndvi

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

if __name__ == "__main__":
    app.run(debug=False)