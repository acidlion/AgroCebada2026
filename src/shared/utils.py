"""
Funciones compartidas de soporte geoespacial, validación por bloques,
métricas Out-Of-Fold y carga de matrices para el ensamble final.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def localizar_raiz_proyecto() -> Path:
    """Ubica automáticamente la raíz del repositorio AgroCebada2026."""
    origenes = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for origen in origenes:
        for carpeta in [origen, *origen.parents]:
            if (carpeta / "data").is_dir() and (carpeta / "src").is_dir():
                return carpeta
    return Path.cwd().resolve()


def buscar_archivo(nombre_archivo: str, carpetas_preferidas: list[Path] | None = None) -> Path:
    """Busca un archivo priorizando una lista de directorios y luego en todo el proyecto."""
    raiz = localizar_raiz_proyecto()
    if carpetas_preferidas:
        for carpeta in carpetas_preferidas:
            candidato = carpeta / nombre_archivo
            if candidato.is_file():
                return candidato
    coincidencias = sorted(raiz.rglob(nombre_archivo))
    if coincidencias:
        return coincidencias[0]
    raise FileNotFoundError(f"No se encontró el archivo requerido: {nombre_archivo}")


def pooled_std(stds: np.ndarray, means: np.ndarray, n: int = 24) -> float:
    """Desviación estándar combinada (mantenida por compatibilidad con exp_001 y exp_002)."""
    stds = np.asarray(stds, dtype=float)
    means = np.asarray(means, dtype=float)
    mask = np.isfinite(stds) & np.isfinite(means)
    if not mask.any():
        return np.nan
    s, m = stds[mask], means[mask]
    k = len(s)
    if k == 1:
        return float(s[0])
    mean_total = np.mean(m)
    var_within = np.sum((n - 1) * (s ** 2))
    var_between = np.sum(n * ((m - mean_total) ** 2))
    denom = k * n - 1
    return float(np.sqrt((var_within + var_between) / denom)) if denom > 0 else np.nan


def calcular_metricas(observado, predicho) -> dict[str, float]:
    """Calcula MAE, RMSE, R2 y sesgo (predicho - observado) sobre un vector de predicciones."""
    obs = np.asarray(observado, dtype=float)
    pred = np.asarray(predicho, dtype=float)
    return {
        "MAE": float(mean_absolute_error(obs, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(obs, pred))),
        "R2": float(r2_score(obs, pred)),
        "sesgo_pred_menos_real": float(np.mean(pred - obs)),
    }


def crear_particiones_espaciales(
    gdf_parcelas_original: gpd.GeoDataFrame,
    ids_etiquetados: pd.Index,
    n_bloques: int = 10,
    n_splits: int = 5,
    semilla: int = 0,
):
    """
    Reproduce el Bloque 38: proyecta a UTM, obtiene centroides, agrupa en 10 bloques
    con KMeans(n_clusters=10, n_init=10, random_state=0) y genera GroupKFold(n_splits=5).
    """
    col_id_shp = "ID_POLIGON" if "ID_POLIGON" in gdf_parcelas_original.columns else "ID_POLIGONO"
    geometrias = (
        gdf_parcelas_original[[col_id_shp, "geometry"]]
        .rename(columns={col_id_shp: "ID_POLIGONO"})
        .copy()
    )
    geometrias["ID_POLIGONO"] = geometrias["ID_POLIGONO"].astype(str).str.strip()
    geometrias = geometrias.set_index("ID_POLIGONO").loc[ids_etiquetados].copy()

    crs_utm = geometrias.estimate_utm_crs()
    geometrias_utm = geometrias.to_crs(crs_utm)
    centroides = geometrias_utm.geometry.centroid

    coordenadas = pd.DataFrame(
        {"coord_x_m": centroides.x, "coord_y_m": centroides.y},
        index=ids_etiquetados,
    )

    agrupador = KMeans(n_clusters=n_bloques, n_init=10, random_state=semilla)
    grupos_espaciales = pd.Series(
        agrupador.fit_predict(coordenadas),
        index=ids_etiquetados,
        name="bloque_espacial",
    )

    cv = GroupKFold(n_splits=n_splits)
    particiones = list(cv.split(coordenadas, groups=grupos_espaciales))
    return grupos_espaciales, particiones, coordenadas, crs_utm


def cargar_matrices_ensamble(raiz: Path | None = None):
    """
    Carga y alinea las matrices de las 197 parcelas (138 entrenamiento y 59 predicción):
      - X_ridge (33 variables de resumen del ciclo abril-octubre 2025)
      - X_xgb (303 variables mensuales de enero-octubre 2025)
      - y_train (138 etiquetas de RENDIMIENTO_T_HA)
    """
    if raiz is None:
        raiz = localizar_raiz_proyecto()

    carpeta_proc = raiz / "data" / "processed"
    carpeta_raw = raiz / "data" / "raw" / "DATASET_RETO_AGRO_2026"

    # 1. Tabla maestra de rendimientos (197 parcelas)
    ruta_y = buscar_archivo("ID_area_rendimiento_70_30_Reto_AgroCebada.csv", [carpeta_raw])
    df_rend = pd.read_csv(ruta_y).drop(columns=["Unnamed: 4"], errors="ignore")
    df_rend["ID_POLIGONO"] = df_rend["ID_POLIGONO"].astype(str).str.strip()
    df_rend["CONJUNTO"] = df_rend["CONJUNTO"].astype(str).str.strip().str.upper()
    df_rend["RENDIMIENTO_T_HA"] = pd.to_numeric(df_rend["RENDIMIENTO_T_HA"], errors="raise")

    ids_train = pd.Index(
        sorted(df_rend.loc[df_rend["CONJUNTO"].eq("ENTRENAMIENTO"), "ID_POLIGONO"]),
        name="ID_POLIGONO",
    )
    # Orden exacto del maestro para las 59 parcelas de predicción (Bloque 73 FINAL)
    ids_pred = pd.Index(
        df_rend.loc[df_rend["CONJUNTO"].eq("PREDICCION"), "ID_POLIGONO"],
        name="ID_POLIGONO",
    )

    maestro = df_rend.set_index("ID_POLIGONO")
    y_train = maestro.loc[ids_train, "RENDIMIENTO_T_HA"].astype(float)

    # 2. Tabla del ciclo (33 variables para Ridge)
    ruta_ciclo = buscar_archivo(
        "tabla_integrada_parcela_ciclo_2025.csv",
        [carpeta_proc, raiz / "notebooks" / "01_eda_erick", raiz / "notebooks"],
    )
    df_ciclo = pd.read_csv(ruta_ciclo)
    df_ciclo["ID_POLIGONO"] = df_ciclo["ID_POLIGONO"].astype(str).str.strip()
    df_ciclo = df_ciclo.set_index("ID_POLIGONO")

    cols_ridge = [
        c for c in df_ciclo.columns
        if c not in ("ID_POLIGONO", "RENDIMIENTO_T_HA", "CONJUNTO")
        and not c.endswith("__meses_disponibles")
    ]
    X_ridge_all = df_ciclo[cols_ridge].apply(pd.to_numeric, errors="raise").astype(float)

    # 3. Tabla mensual ampliada con enero-marzo 2025 (303 variables para XGBoost)
    ruta_xgb_303 = carpeta_proc / "tabla_mensual_303_xgb_2025.csv"
    if ruta_xgb_303.is_file():
        df_xgb = pd.read_csv(ruta_xgb_303)
        df_xgb["ID_POLIGONO"] = df_xgb["ID_POLIGONO"].astype(str).str.strip()
        X_xgb_all = df_xgb.set_index("ID_POLIGONO").apply(pd.to_numeric, errors="raise").astype(float)
    else:
        # Buscar en los artefactos exportados por el Bloque 57 en reports/resultados/exp_004/
        candidatos_e4 = sorted(raiz.rglob("e4_mensual_3m/X_197_parcelas.csv"))
        if candidatos_e4:
            df_xgb = pd.read_csv(candidatos_e4[-1])
            df_xgb["ID_POLIGONO"] = df_xgb["ID_POLIGONO"].astype(str).str.strip()
            X_xgb_all = df_xgb.set_index("ID_POLIGONO").apply(pd.to_numeric, errors="raise").astype(float)
            carpeta_proc.mkdir(parents=True, exist_ok=True)
            X_xgb_all.reset_index().to_csv(ruta_xgb_303, index=False, encoding="utf-8-sig")
        else:
            raise FileNotFoundError(
                "No se encontró tabla_mensual_303_xgb_2025.csv ni e4_mensual_3m/X_197_parcelas.csv. "
                "Ejecuta primero: python -m src.data.pipeline"
            )

    # 4. Shapefile para particiones espaciales
    ruta_shp = buscar_archivo("Parcelas_Reto_AGC_CONJUNTO.shp", [carpeta_raw])
    gdf_parcelas = gpd.read_file(ruta_shp)

    return {
        "maestro": maestro,
        "y_train": y_train,
        "ids_train": ids_train,
        "ids_pred": ids_pred,
        "Xtr_ridge": X_ridge_all.loc[ids_train].copy(),
        "Xpr_ridge": X_ridge_all.loc[ids_pred].copy(),
        "Xtr_xgb": X_xgb_all.loc[ids_train].copy(),
        "Xpr_xgb": X_xgb_all.loc[ids_pred].copy(),
        "gdf_parcelas": gdf_parcelas,
    }