import geopandas as gpd
from rasterstats import zonal_stats

# 1. Cargar tus parcelas
gdf_parcelas = gpd.read_file('parcelas.shp')

# 2. Definir la ruta de tu archivo raster (.tif)
ruta_raster_elevacion = 'elevacion_inegi.tif'

# 3. Extraer las estadísticas (promedio, máximo, mínimo) de los píxeles 
# que caen dentro de los polígonos de tus parcelas
estadisticas = zonal_stats(
    gdf_parcelas,             # Tus polígonos
    ruta_raster_elevacion,    # Tu raster
    stats="mean min max",     # Qué cálculos quieres hacer
    geojson_out=True          # Devuelve la estructura para reconstruir el GeoDataFrame
)

# 4. Convertir el resultado de vuelta a un GeoDataFrame de Pandas
gdf_enriquecido = gpd.GeoDataFrame.from_features(estadisticas)

# Ahora tus parcelas tienen nuevas columnas con la información del TIF
print(gdf_enriquecido[['ID', 'mean', 'max']].head())