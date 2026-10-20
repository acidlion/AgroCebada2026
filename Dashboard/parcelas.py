import geopandas as gpd
import plotly.express as px

# 1. Función para cargar y preparar las geometrías (Ejecutar al inicio del archivo, fuera de los callbacks)
def cargar_parcelas(ruta_shp):
    # Leer el shapefile con GeoPandas
    gdf = gpd.read_file(ruta_shp)
    
    # FORZAR REPROYECCIÓN: Convertir al estándar GPS (WGS84) para que Plotly lo lea correctamente
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
        
    return gdf

# Cargar las parcelas en memoria global
# Asegúrate de cambiar 'ruta_al_archivo/parcelas.shp' por la ruta real de tus datos
gdf_parcelas = cargar_parcelas('parcelas.shp')

# 2. Modificación del Callback en Dash para usar GeoPandas
@app.callback(
    [Output('mapa-principal', 'figure'),
     Output('kpi-total', 'children'),
     Output('kpi-area', 'children')],
    [Input('filtro-conjunto', 'value'),
     Input('filtro-variable', 'value')]
)
def actualizar_mapa(conjunto, variable):
    # Filtrar el GeoDataFrame según el dropdown
    if conjunto == 'ALL':
        gdf_filtrado = gdf_parcelas
    else:
        gdf_filtrado = gdf_parcelas[gdf_parcelas['CONJUNTO'] == conjunto]
    
    # Calcular KPIs reales basados en los datos espaciales
    total_parcelas = str(len(gdf_filtrado))
    # Asumiendo que la columna de área se llama 'AREA' en el shapefile
    area_total = f"{gdf_filtrado['AREA'].sum():,.2f} ha" if 'AREA' in gdf_filtrado.columns else "N/A"
    
    # 3. Dibujar los polígonos en el mapa
    fig = px.choropleth_mapbox(
        gdf_filtrado,
        geojson=gdf_filtrado.geometry, # Extrae automáticamente las coordenadas de los polígonos
        locations=gdf_filtrado.index,  # Índice de referencia interno
        color="CONJUNTO",              # Colorear según Entrenamiento o Predicción
        color_discrete_map={
            "ENTRENAMIENTO": "#2E7D32", # Verde Vegetación (Tu paleta CSS)
            "PREDICCION": "#FBC02D"     # Dorado Cebada
        },
        mapbox_style="carto-positron",
        center={"lat": 19.3, "lon": -98.2}, # Centro aproximado (Hidalgo, Puebla, Tlaxcala)
        zoom=7.5,
        opacity=0.7,
        hover_data=["ID", "AREA", "CONJUNTO"] # Datos a mostrar al pasar el cursor (Ajustar nombres de columnas)
    )
    
    # Aplicar el tema para que coincida con tu CSS
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="#F1F8E9", # Fondo neutro
        plot_bgcolor="#F1F8E9",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig, total_parcelas, area_total