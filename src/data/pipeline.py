"""
Pipeline de preparación de datos para el modelo final (Ensamble Ridge + XGBoost 50/50).
Reproduce los Bloques 1 al 33.4 (EDA) y el Bloque 57 (Antecedentes Ene-Mar 2025):
  1. Limpieza de BÁSICO y PRO (exclusión de filas vacías e imputaciones puntuales fuera de 2025).
  2. Extracción ponderada por cobertura de píxel (exactextract) de clima (144 TIFs) y topografía (2 TIFs).
  3. Construcción de la tabla de resumen del ciclo 2025 (33 predictoras para Ridge).
  4. Construcción de la tabla mensual Abr-Oct 2025 (213 predictoras) y ampliada Ene-Oct 2025 (303 predictoras para XGBoost).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from exactextract import exact_extract

from src.shared.utils import localizar_raiz_proyecto, buscar_archivo


def limpiar_bases_satelitales(df_basico_orig: pd.DataFrame, df_pro_orig: pd.DataFrame):
    """Reproduce los Bloques 10 al 18: limpia BÁSICO y PRO y formatea fechas."""
    cols_indices = [
        c for c in df_basico_orig.columns
        if c.endswith(("_promedio", "_std", "_max", "_min"))
    ]
    # 1. Excluir las 3,056 filas sin dato de nubosidad ni índices en Landsat (Bloque 10)
    filas_excluir = (
        df_basico_orig["porcentaje_nubosidad"].isna()
        & df_basico_orig[cols_indices].isna().all(axis=1)
    )
    df_basico = df_basico_orig.loc[~filas_excluir].copy()

    # 2. Imputación experimental del caso aislado de VI6T en AGC_013 (19/05/2024, Bloque 12)
    cols_vi6t = ["vi6t_promedio", "vi6t_std", "vi6t_max", "vi6t_min"]
    fechas_b = pd.to_datetime(df_basico["fecha_captura"], format="%d/%m/%Y", errors="raise")
    es_agc013 = df_basico["ID_POLIGONO"].eq("AGC_013") & df_basico["sensor"].eq("Landsat")
    es_obj_vi6t = (
        es_agc013
        & fechas_b.eq(pd.Timestamp("2024-05-19"))
        & df_basico["porcentaje_nubosidad"].eq(97.3)
    )
    es_donante_vi6t = (
        es_agc013
        & df_basico["porcentaje_nubosidad"].ge(95)
        & df_basico["porcentaje_nubosidad"].lt(100)
        & ~es_obj_vi6t
        & df_basico[cols_vi6t].notna().all(axis=1)
    )
    medias_vi6t = df_basico.loc[es_donante_vi6t, cols_vi6t].mean()
    for col in cols_vi6t:
        df_basico.loc[es_obj_vi6t, col] = medias_vi6t[col]

    # 3. Imputación por interpolación lineal temporal de los 4 casos de FAPAR (07/12/2022, Bloques 14-15)
    cols_fapar = ["fapar_promedio", "fapar_std", "fapar_max", "fapar_min"]
    rev_fapar = df_basico.loc[df_basico["sensor"].eq("Sentinel-2")].copy()
    rev_fapar["fecha_rev"] = pd.to_datetime(rev_fapar["fecha_captura"], format="%d/%m/%Y", errors="raise")
    casos_fapar = rev_fapar.loc[
        rev_fapar[cols_fapar].isna().any(axis=1)
        & rev_fapar["porcentaje_nubosidad"].notna()
        & rev_fapar["porcentaje_nubosidad"].ne(100)
    ]

    for idx, caso in casos_fapar.iterrows():
        dias = (rev_fapar["fecha_rev"] - caso["fecha_rev"]).dt.days
        ref_validas = rev_fapar.loc[
            rev_fapar["ID_POLIGONO"].eq(caso["ID_POLIGONO"])
            & rev_fapar[cols_fapar].notna().all(axis=1)
            & rev_fapar["porcentaje_nubosidad"].eq(0)
            & dias.abs().le(30)
            & dias.ne(0)
        ]
        ant = ref_validas.loc[ref_validas["fecha_rev"].lt(caso["fecha_rev"])].sort_values("fecha_rev").iloc[-1]
        pos = ref_validas.loc[ref_validas["fecha_rev"].gt(caso["fecha_rev"])].sort_values("fecha_rev").iloc[0]
        peso = (caso["fecha_rev"] - ant["fecha_rev"]).days / (pos["fecha_rev"] - ant["fecha_rev"]).days
        for col in cols_fapar:
            if pd.isna(caso[col]):
                df_basico.at[idx, col] = ant[col] + peso * (pos[col] - ant[col])

    # 4. Formatear fechas y ordenar (Bloque 18)
    df_pro = df_pro_orig.copy()
    for tabla in (df_basico, df_pro):
        tabla["ID_POLIGONO"] = tabla["ID_POLIGONO"].astype(str).str.strip()
        tabla["fecha_captura"] = pd.to_datetime(tabla["fecha_captura"], format="%d/%m/%Y", errors="raise")
        tabla.sort_values(["ID_POLIGONO", "fecha_captura", "sensor"], inplace=True)

    return df_basico, df_pro


def extraer_ambiente_mensual(
    gdf_parcelas: gpd.GeoDataFrame,
    rutas_inv: dict[str, Path],
    carpeta_proc: Path,
    recalcular_rasters: bool = False,
) -> pd.DataFrame:
    """Reproduce los Bloques 26 al 29: extrae clima mensual y topografía con exactextract."""
    ruta_ambiente_pq = carpeta_proc / "df_ambiente_mensual.parquet"
    if ruta_ambiente_pq.is_file() and not recalcular_rasters:
        print(f"  [OK] Reutilizando clima y topografía pre-extraídos: {ruta_ambiente_pq.name}")
        return pd.read_parquet(ruta_ambiente_pq)

    inventarios = {k: pd.read_csv(v) for k, v in rutas_inv.items()}
    parcelas_por_crs = {}

    # 1. Extraer Topografía (Elevacion y Pendiente)
    filas_topo = []
    for _, fila in inventarios["topografia"].iterrows():
        var_topo = fila["variable"]
        ruta_tif = rutas_inv["topografia"].parent / Path(str(fila["archivo"]).replace("\\", "/"))
        with rasterio.open(ruta_tif) as src:
            wkt = src.crs.to_wkt()
            if wkt not in parcelas_por_crs:
                parcelas_por_crs[wkt] = gdf_parcelas[["ID_POLIGONO", "geometry"]].to_crs(src.crs)
            res = exact_extract(
                src, parcelas_por_crs[wkt], ["mean"], include_cols=["ID_POLIGONO"], output="pandas"
            )
        res["variable"] = var_topo
        filas_topo.append(res.rename(columns={"mean": "media_ponderada"}))

    df_topografia = (
        pd.concat(filas_topo, ignore_index=True)
        .pivot(index="ID_POLIGONO", columns="variable", values="media_ponderada")
        .rename(columns={"Elevacion": "elevacion_m", "Pendiente": "pendiente_grados"})
        .reset_index()
    )
    df_topografia.columns.name = None

    # 2. Extraer Clima Mensual (144 archivos .tif: Precipitacion, Tmax, Tmin)
    filas_clima = []
    total_archivos = len(inventarios["precipitacion"]) + len(inventarios["temperatura"])
    procesados = 0

    for nombre_inv in ("precipitacion", "temperatura"):
        inv = inventarios[nombre_inv]
        for _, fila in inv.iterrows():
            var_clima = "Precipitacion" if nombre_inv == "precipitacion" else fila["variable"]
            ruta_tif = rutas_inv[nombre_inv].parent / Path(str(fila["archivo"]).replace("\\", "/"))
            with rasterio.open(ruta_tif) as src:
                wkt = src.crs.to_wkt()
                if wkt not in parcelas_por_crs:
                    parcelas_por_crs[wkt] = gdf_parcelas[["ID_POLIGONO", "geometry"]].to_crs(src.crs)
                res = exact_extract(
                    src, parcelas_por_crs[wkt], ["mean"], include_cols=["ID_POLIGONO"], output="pandas"
                )
            res = res.rename(columns={"mean": "media_ponderada"})
            res["variable"] = var_clima
            res["anio"] = int(fila["anio"])
            res["mes"] = int(fila["mes"])
            filas_clima.append(res)
            procesados += 1
            if procesados % 24 == 0 or procesados == total_archivos:
                print(f"  -> Rasters climáticos procesados: {procesados}/{total_archivos}")

    df_clima_largo = pd.concat(filas_clima, ignore_index=True)
    df_clima_mensual = (
        df_clima_largo.pivot(index=["ID_POLIGONO", "anio", "mes"], columns="variable", values="media_ponderada")
        .rename(columns={"Precipitacion": "precipitacion_mm", "Tmax": "tmax_c", "Tmin": "tmin_c"})
        .reset_index()
    )
    df_clima_mensual.columns.name = None
    df_clima_mensual["fecha_mes"] = pd.to_datetime(
        {"year": df_clima_mensual["anio"], "month": df_clima_mensual["mes"], "day": 1}
    )
    df_clima_mensual = df_clima_mensual[
        ["ID_POLIGONO", "anio", "mes", "fecha_mes", "precipitacion_mm", "tmax_c", "tmin_c"]
    ].sort_values(["ID_POLIGONO", "fecha_mes"]).reset_index(drop=True)

    df_ambiente = df_clima_mensual.merge(df_topografia, on="ID_POLIGONO", how="left", validate="many_to_one")
    carpeta_proc.mkdir(parents=True, exist_ok=True)
    df_clima_mensual.to_parquet(carpeta_proc / "df_clima_mensual.parquet", index=False)
    df_ambiente.to_parquet(ruta_ambiente_pq, index=False)
    return df_ambiente


def construir_base_mensual(
    df_ambiente_mensual: pd.DataFrame,
    df_basico_prep: pd.DataFrame,
    df_pro_prep: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce los Bloques 30 y 31: resume sensores por mes y los une con clima y topografía."""
    clave_mensual = ["ID_POLIGONO", "fecha_mes"]
    df_base_mensual = df_ambiente_mensual.copy()

    for fuente, tabla in [("basico", df_basico_prep), ("pro", df_pro_prep)]:
        for sensor, grupo in tabla.groupby("sensor"):
            grupo = grupo.copy()
            grupo["fecha_mes"] = grupo["fecha_captura"].dt.to_period("M").dt.to_timestamp()
            cols_prom = [c for c in grupo.columns if c.endswith("_promedio")]
            if sensor == "Landsat":
                cols_prom = [c for c in cols_prom if c.startswith("vi6t_")]
            elif sensor == "Sentinel-2":
                cols_prom = [c for c in cols_prom if not c.startswith("vi6t_")]

            agrupado = grupo.groupby(clave_mensual)
            mensual = agrupado[cols_prom].agg(["mean", "count"])
            mensual.columns = [
                f"{col}_{'media_mensual' if est == 'mean' else 'n_obs'}"
                for col, est in mensual.columns
            ]
            calidad = agrupado.agg(
                capturas_totales=("fecha_captura", "size"),
                capturas_nubosidad_100=("porcentaje_nubosidad", lambda s: int(s.eq(100).sum())),
                nubosidad_media=("porcentaje_nubosidad", "mean"),
            )
            mensual = mensual.join(calidad, validate="one_to_one").reset_index()

            prefijo = f"{fuente}_{sensor}".lower().replace("-", "_")
            nombres_nuevos = {
                col: f"{prefijo}__{col}"
                for col in mensual.columns if col not in clave_mensual
            }
            mensual_pref = mensual.rename(columns=nombres_nuevos)

            df_base_mensual = df_base_mensual.merge(
                mensual_pref, on=clave_mensual, how="left", validate="one_to_one", indicator="_estado"
            )
            df_base_mensual[f"{prefijo}__registro_mensual_presente"] = df_base_mensual["_estado"].eq("both")
            df_base_mensual.drop(columns="_estado", inplace=True)

    return df_base_mensual


def construir_tablas_modelado(df_base_mensual: pd.DataFrame, df_rendimiento: pd.DataFrame, carpeta_proc: Path):
    """
    Reproduce los Bloques 32, 33, 33.2, 33.3, 33.4 y 57:
      1. Tabla resumen del ciclo Abr-Oct 2025 (33 predictoras -> Ridge).
      2. Tabla mensual Abr-Oct 2025 (213 predictoras).
      3. Tabla mensual ampliada Ene-Oct 2025 (303 predictoras -> XGBoost).
    """
    inicio_ciclo = pd.Timestamp("2025-04-01")
    fin_ciclo_exc = pd.Timestamp("2025-11-01")

    df_ciclo_mensual = df_base_mensual.loc[
        df_base_mensual["fecha_mes"].ge(inicio_ciclo)
        & df_base_mensual["fecha_mes"].lt(fin_ciclo_exc)
    ].copy()

    # --- A. TABLA RESUMEN DEL CICLO (33 PREDICTORAS PARA RIDGE — Bloque 33) ---
    agrupado_ciclo = df_ciclo_mensual.groupby("ID_POLIGONO")
    resumen_amb = agrupado_ciclo.agg(
        precipitacion_acumulada_mm=("precipitacion_mm", lambda s: s.sum(min_count=7)),
        tmax_media_ciclo_c=("tmax_c", "mean"),
        tmin_media_ciclo_c=("tmin_c", "mean"),
        elevacion_m=("elevacion_m", "first"),
        pendiente_grados=("pendiente_grados", "first"),
    )
    cols_ind_mensuales = [c for c in df_ciclo_mensual.columns if c.endswith("_media_mensual")]
    medias_ind = agrupado_ciclo[cols_ind_mensuales].mean().rename(
        columns={c: c.replace("_promedio_media_mensual", "__media_ciclo") for c in cols_ind_mensuales}
    )
    meses_ind = agrupado_ciclo[cols_ind_mensuales].count().rename(
        columns={c: c.replace("_promedio_media_mensual", "__meses_disponibles") for c in cols_ind_mensuales}
    )

    resumen_parcela = resumen_amb.join(medias_ind).join(meses_ind).reset_index()
    df_eda_parcela = df_rendimiento.merge(resumen_parcela, on="ID_POLIGONO", how="left", validate="one_to_one")

    # --- B. TABLA DE MESES SEPARADOS ABR-OCT 2025 (213 PREDICTORAS — Bloques 33.2 a 33.4) ---
    fechas_ciclo = pd.date_range("2025-04-01", "2025-10-01", freq="MS")
    cols_clima_m = ["precipitacion_mm", "tmax_c", "tmin_c"]
    cols_conteos_m = [c for c in df_ciclo_mensual.columns if c.endswith("_n_obs")]
    cols_presencia_m = [c for c in df_ciclo_mensual.columns if c.endswith("__registro_mensual_presente")]
    cols_expandir = cols_clima_m + cols_ind_mensuales + cols_conteos_m + cols_presencia_m

    ancha_m = df_ciclo_mensual[["ID_POLIGONO", "fecha_mes"] + cols_expandir].pivot(
        index="ID_POLIGONO", columns="fecha_mes", values=cols_expandir
    )
    orden_cols = pd.MultiIndex.from_product([cols_expandir, fechas_ciclo], names=ancha_m.columns.names)
    ancha_m = ancha_m.reindex(columns=orden_cols)
    ancha_m.columns = [
        f"{var.replace('_promedio_media_mensual', '__media_mensual').replace('_promedio_n_obs', '__n_obs')}__{fec.strftime('%Y_%m')}"
        for var, fec in ancha_m.columns
    ]
    df_parcela_meses_base = ancha_m.reset_index()

    topo_unica = df_ciclo_mensual[["ID_POLIGONO", "elevacion_m", "pendiente_grados"]].drop_duplicates()
    df_parcela_meses = (
        df_rendimiento[["ID_POLIGONO", "AREA_HA", "CONJUNTO", "RENDIMIENTO_T_HA"]]
        .merge(topo_unica, on="ID_POLIGONO", how="left", validate="one_to_one")
        .merge(df_parcela_meses_base, on="ID_POLIGONO", how="left", validate="one_to_one")
    )

    cols_estaticas = ["AREA_HA", "elevacion_m", "pendiente_grados"]
    cols_clima_ancho = [c for c in df_parcela_meses.columns if c.startswith(("precipitacion_mm__", "tmax_c__", "tmin_c__"))]
    cols_ind_ancho = [c for c in df_parcela_meses.columns if "__media_mensual__" in c]
    cols_213 = cols_estaticas + cols_clima_ancho + cols_ind_ancho

    # --- C. TABLA MENSUAL AMPLIADA ENE-OCT 2025 (303 PREDICTORAS PARA XGBOOST — Bloque 57) ---
    ids_ordenados = pd.Index(sorted(df_rendimiento["ID_POLIGONO"].astype(str).str.strip()), name="ID_POLIGONO")
    base_213 = (
        df_parcela_meses.set_index("ID_POLIGONO")
        .loc[ids_ordenados, cols_213]
        .apply(pd.to_numeric, errors="raise")
        .astype(float)
    )

    meses_3m = pd.date_range("2025-01-01", "2025-03-01", freq="MS")
    mediciones_30 = cols_clima_m + cols_ind_mensuales
    sub_3m = df_base_mensual.loc[
        df_base_mensual["fecha_mes"].isin(meses_3m),
        ["ID_POLIGONO", "fecha_mes"] + mediciones_30,
    ].copy()
    sub_3m[mediciones_30] = sub_3m[mediciones_30].apply(pd.to_numeric, errors="raise")

    ancho_3m = sub_3m.pivot(index="ID_POLIGONO", columns="fecha_mes", values=mediciones_30)
    orden_3m = pd.MultiIndex.from_product([mediciones_30, meses_3m], names=ancho_3m.columns.names)
    ancho_3m = ancho_3m.reindex(index=ids_ordenados, columns=orden_3m)
    ancho_3m.columns = [
        f"{var.replace('_promedio_media_mensual', '__media_mensual')}__{fec.strftime('%Y_%m')}"
        for var, fec in ancho_3m.columns
    ]
    ancho_3m = ancho_3m.astype(float)

    tabla_303_xgb = pd.concat([base_213, ancho_3m], axis=1)
    assert tabla_303_xgb.shape == (197, 303), f"Se esperaban (197, 303), se obtuvo {tabla_303_xgb.shape}"

    # Guardar artefactos en data/processed/
    carpeta_proc.mkdir(parents=True, exist_ok=True)
    df_base_mensual.to_parquet(carpeta_proc / "df_base_mensual.parquet", index=False)
    df_eda_parcela.to_csv(carpeta_proc / "tabla_integrada_parcela_ciclo_2025.csv", index=False, encoding="utf-8-sig")
    df_parcela_meses.to_csv(carpeta_proc / "tabla_parcela_meses_2025_sin_imputar.csv", index=False, encoding="utf-8-sig", na_rep="")
    tabla_303_xgb.reset_index().to_csv(carpeta_proc / "tabla_mensual_303_xgb_2025.csv", index=False, encoding="utf-8-sig")

    print(f"  [OK] tabla_integrada_parcela_ciclo_2025.csv guardada: {df_eda_parcela.shape}")
    print(f"  [OK] tabla_parcela_meses_2025_sin_imputar.csv guardada: {df_parcela_meses.shape}")
    print(f"  [OK] tabla_mensual_303_xgb_2025.csv guardada: {tabla_303_xgb.shape}")


def main():
    raiz = localizar_raiz_proyecto()
    carpeta_raw = raiz / "data" / "raw" / "DATASET_RETO_AGRO_2026"
    carpeta_proc = raiz / "data" / "processed"

    print(f"=== Ejecutando Pipeline de Datos (Ensamble Final) | Raíz: {raiz} ===")

    ruta_y = buscar_archivo("ID_area_rendimiento_70_30_Reto_AgroCebada.csv", [carpeta_raw])
    ruta_basico = buscar_archivo("Conjunto_datos_BASICO_AgroCebada2026.csv", [carpeta_raw])
    ruta_pro = buscar_archivo("Conjunto_datos_PRO_AgroCebada.csv", [carpeta_raw])
    ruta_shp = buscar_archivo("Parcelas_Reto_AGC_CONJUNTO.shp", [carpeta_raw])

    rutas_inv = {
        "precipitacion": buscar_archivo("inventario_CHIRPS_precipitacion_mensual_2022_2025.csv", [carpeta_raw]),
        "temperatura": buscar_archivo("inventario_temperatura_2022_2025.csv", [carpeta_raw]),
        "topografia": buscar_archivo("inventario_topografia.csv", [carpeta_raw]),
    }

    df_rendimiento = pd.read_csv(ruta_y).drop(columns=["Unnamed: 4"], errors="ignore")
    df_rendimiento["ID_POLIGONO"] = df_rendimiento["ID_POLIGONO"].astype(str).str.strip()

    gdf_parcelas = gpd.read_file(ruta_shp).rename(columns={"ID_POLIGON": "ID_POLIGONO"})
    gdf_parcelas["ID_POLIGONO"] = gdf_parcelas["ID_POLIGONO"].astype(str).str.strip()

    print("1. Limpiando conjuntos satelitales BÁSICO y PRO...")
    df_basico_orig = pd.read_csv(ruta_basico, low_memory=False)
    df_pro_orig = pd.read_csv(ruta_pro, low_memory=False)
    df_basico_prep, df_pro_prep = limpiar_bases_satelitales(df_basico_orig, df_pro_orig)

    print("2. Extrayendo clima mensual y topografía por parcela...")
    df_ambiente_mensual = extraer_ambiente_mensual(gdf_parcelas, rutas_inv, carpeta_proc)

    print("3. Integrando series mensuales satelitales y ambientales...")
    df_base_mensual = construir_base_mensual(df_ambiente_mensual, df_basico_prep, df_pro_prep)

    print("4. Generando tablas finales para Ridge (33 vars) y XGBoost (303 vars)...")
    construir_tablas_modelado(df_base_mensual, df_rendimiento, carpeta_proc)
    print("=== Pipeline finalizado con éxito ===")


if __name__ == "__main__":
    main()