"""
Experimento 2 (EXP-002): Fusión Multimodal Completa + Topografía (449 features).
Calcula métricas globales Out-Of-Fold (MAE_OOF, RMSE_OOF, R2_OOF) y promedios por fold.
"""
from pathlib import Path
import shutil
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


# ----------------------------------------------------------------------------
# LOCALIZACIÓN AUTOMÁTICA DE LA RAÍZ DEL PROYECTO Y RUTAS LOCALES
# ----------------------------------------------------------------------------
def localizar_raiz_proyecto() -> Path:
    orígenes = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for origen in orígenes:
        for carpeta in [origen, *origen.parents]:
            if (carpeta / "data").is_dir() and (carpeta / "src").is_dir():
                return carpeta
    return Path.cwd().resolve()


ROOT_DIR = localizar_raiz_proyecto()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def localizar_archivo(ruta_preferida: Path, nombre_archivo: str) -> Path:
    if ruta_preferida.is_file():
        return ruta_preferida
    coincidencias = sorted((ROOT_DIR / "data").rglob(nombre_archivo))
    if coincidencias:
        return coincidencias[0]
    return ruta_preferida


# ----------------------------------------------------------------------------
# CONFIGURACIÓN EXP-002
# ----------------------------------------------------------------------------
EXP_ID = "exp_002"
DIR_SALIDA = ROOT_DIR / "reports" / "resultados" / EXP_ID

RUTA_PANEL = ROOT_DIR / "data" / "processed" / "panel.parquet"
RUTA_ESTATICOS = ROOT_DIR / "data" / "processed" / "estaticos.parquet"

RUTA_SHP = localizar_archivo(
    ROOT_DIR / "data/raw/DATASET_RETO_AGRO_2026/Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO.shp",
    "Parcelas_Reto_AGC_CONJUNTO.shp",
)
RUTA_Y = localizar_archivo(
    ROOT_DIR / "data/raw/DATASET_RETO_AGRO_2026/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
    "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
)

COL_ID = "ID_POLIGON"
COL_Y = "RENDIMIENTO_T_HA"
INICIO = "2025-04-01"
FIN = "2025-10-01"

SEED = 0
N_SPLITS = 5
N_BLOQUES_MAX = 10
BUSCAR_HIPERPARAMETROS = False


# ----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ----------------------------------------------------------------------------
def cargar_panel():
    panel = pd.read_parquet(RUTA_PANEL)
    panel[COL_ID] = panel[COL_ID].astype(str).str.strip()
    panel["time"] = pd.to_datetime(panel["time"])
    return panel


def cargar_zonas():
    """Lee el shapefile, calcula coordenadas/área e integra topografía desde estaticos.parquet."""
    zonas = gpd.read_file(RUTA_SHP)
    zonas[COL_ID] = zonas[COL_ID].astype(str).str.strip()
    assert zonas[COL_ID].is_unique, "Hay IDs duplicados en el shapefile"

    zonas_utm = zonas.to_crs(zonas.estimate_utm_crs())
    centroides = zonas_utm.centroid

    coords_zonas = pd.DataFrame({
        COL_ID: zonas[COL_ID],
        "coord_x": centroides.x,
        "coord_y": centroides.y,
    })
    estaticas = pd.DataFrame({
        COL_ID: zonas[COL_ID],
        "area_km2": zonas_utm.area / 1e6,
    })

    if RUTA_ESTATICOS.exists():
        df_topo = pd.read_parquet(RUTA_ESTATICOS)
        df_topo[COL_ID] = df_topo[COL_ID].astype(str).str.strip()
        cols_topo = [c for c in df_topo.select_dtypes("number").columns if c not in estaticas.columns]
        if cols_topo:
            estaticas = estaticas.merge(df_topo[[COL_ID] + cols_topo], on=COL_ID, how="left")
            print(f"Variables topográficas integradas desde {RUTA_ESTATICOS.name}: {cols_topo}")

    return zonas, coords_zonas, estaticas


# ----------------------------------------------------------------------------
# 2. FEATURES: UNA FILA POR ZONA
# ----------------------------------------------------------------------------
def elegir_variables_dinamicas(panel):
    excluir = {"area_km2", "Elevacion", "Pendiente", "lon", "lat", "mes_sin", "mes_cos"}
    return [c for c in panel.select_dtypes("number").columns
            if c not in excluir
            and "_lag" not in c and "_roll" not in c
            and not c.endswith("_count")]


def construir_features(panel, inicio, fin, vars_dinamicas, estaticas):
    ventana = panel[(panel["time"] >= inicio) & (panel["time"] <= fin)]

    agg = ventana.groupby(COL_ID)[vars_dinamicas].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    cobertura = (ventana.groupby(COL_ID)[vars_dinamicas[0]].count()
                        .rename("meses_con_dato").reset_index())
    agg = agg.merge(cobertura, on=COL_ID, how="left")

    return agg.merge(estaticas, on=COL_ID, how="left")


def armar_dataset(X_all, coords_zonas):
    y_df = pd.read_csv(RUTA_Y)
    posibles_ids = [c for c in y_df.columns if "id" in c.lower() and "poligon" in c.lower()]
    if posibles_ids:
        y_df = y_df.rename(columns={posibles_ids[0]: COL_ID})
    elif "ID" in y_df.columns:
        y_df = y_df.rename(columns={"ID": COL_ID})

    y_df[COL_ID] = y_df[COL_ID].astype(str).str.strip()

    datos = (X_all
             .merge(y_df[[COL_ID, COL_Y]], on=COL_ID, how="inner")
             .merge(coords_zonas, on=COL_ID, how="left")
             .dropna(subset=[COL_Y, "coord_x", "coord_y"])
             .reset_index(drop=True))

    y = datos[COL_Y]
    coords = datos[["coord_x", "coord_y"]].to_numpy()
    X = datos.drop(columns=[COL_ID, COL_Y, "coord_x", "coord_y"])
    X = X.dropna(axis=1, how="all")
    return datos, X, y, coords


# ----------------------------------------------------------------------------
# 3. VALIDACIÓN POR BLOQUES ESPACIALES
# ----------------------------------------------------------------------------
def crear_bloques(coords):
    n_bloques = int(np.clip(len(coords) // 5, N_SPLITS, N_BLOQUES_MAX))
    return KMeans(n_clusters=n_bloques, n_init=10,
                  random_state=SEED).fit_predict(coords)


# ----------------------------------------------------------------------------
# 4. MODELOS
# ----------------------------------------------------------------------------
def definir_modelos():
    return {
        "baseline_media": DummyRegressor(strategy="mean"),

        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.5,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),

        "xgboost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


PARAMS_BUSQUEDA_XGB = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.01, 0.03, 0.1],
    "max_depth": [2, 3, 4],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda": [1, 5, 10],
}


# ----------------------------------------------------------------------------
# 5. EVALUACIÓN OUT-OF-FOLD (OOF) Y POR PARTICIÓN (ESTILO ERICK)
# ----------------------------------------------------------------------------
def evaluar(modelos, X, y, grupos, cv):
    """
    Calcula las predicciones Out-Of-Fold (OOF) en las 138 parcelas y devuelve:
      1. Tabla resumen con métricas globales OOF (MAE_OOF, RMSE_OOF, R2_OOF)
         y promedios por fold (MAE (val), RMSE (val), R2 (val), R2 (val) std, R2 (train)).
      2. Tabla detallada por fold (metricas_folds).
      3. Diccionario con las series de predicciones OOF por modelo.
    """
    particiones = list(cv.split(X, y, groups=grupos))
    oof_preds = {}
    filas_resumen = {}
    filas_folds = []

    for nombre, modelo in modelos.items():
        print(f"      ... Evaluando OOF y particiones para: {nombre}...")
        pred_oof = np.full(len(y), np.nan)

        for fold, (tr, va) in enumerate(particiones, start=1):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]

            m_fold = clone(modelo).fit(X_tr, y_tr)
            p_val = m_fold.predict(X_va)
            p_tr = m_fold.predict(X_tr)

            pred_oof[va] = p_val

            filas_folds.append({
                "modelo": nombre,
                "fold": fold,
                "n_train": len(tr),
                "n_validacion": len(va),
                "MAE_val": mean_absolute_error(y_va, p_val),
                "RMSE_val": float(np.sqrt(mean_squared_error(y_va, p_val))),
                "R2_val": r2_score(y_va, p_val),
                "R2_train": r2_score(y_tr, p_tr),
                "sesgo_pred_menos_real": float(np.mean(p_val - y_va.to_numpy())),
            })

        serie_oof = pd.Series(pred_oof, index=y.index, name=nombre)
        oof_preds[nombre] = serie_oof

        df_m_folds = pd.DataFrame([f for f in filas_folds if f["modelo"] == nombre])
        mae_oof = mean_absolute_error(y, serie_oof)
        rmse_oof = float(np.sqrt(mean_squared_error(y, serie_oof)))
        r2_oof = r2_score(y, serie_oof)

        filas_resumen[nombre] = {
            "MAE_OOF": mae_oof,
            "RMSE_OOF": rmse_oof,
            "R2_OOF": r2_oof,
            "MAE (val)": df_m_folds["MAE_val"].mean(),
            "RMSE (val)": df_m_folds["RMSE_val"].mean(),
            "R2 (val)": df_m_folds["R2_val"].mean(),
            "R2 (val) std": df_m_folds["R2_val"].std(ddof=0),
            "R2 (train)": df_m_folds["R2_train"].mean(),
        }
        print(
            f"          [OK] {nombre} -> "
            f"RMSE_OOF: {rmse_oof:.4f} | R2_OOF: {r2_oof:.4f} | "
            f"RMSE (val): {filas_resumen[nombre]['RMSE (val)']:.3f}"
        )

    df_resumen = pd.DataFrame(filas_resumen).T
    df_resumen.index.name = "modelo"

    if "baseline_media" in df_resumen.index:
        rmse_base = df_resumen.loc["baseline_media", "RMSE_OOF"]
        df_resumen["reduccion_RMSE_vs_baseline_pct"] = (
            100.0 * (1.0 - df_resumen["RMSE_OOF"] / rmse_base)
        )

    return df_resumen, pd.DataFrame(filas_folds), oof_preds


def evaluar_anidado_xgb(X, y, grupos, cv_externo):
    rmses, r2s = [], []
    pred_oof = np.full(len(y), np.nan)
    for tr, va in cv_externo.split(X, y, groups=grupos):
        busqueda = RandomizedSearchCV(
            XGBRegressor(random_state=SEED, n_jobs=-1),
            param_distributions=PARAMS_BUSQUEDA_XGB,
            n_iter=25,
            cv=GroupKFold(n_splits=3),
            scoring="neg_root_mean_squared_error",
            random_state=SEED,
        )
        busqueda.fit(X.iloc[tr], y.iloc[tr], groups=grupos[tr])
        pred = busqueda.predict(X.iloc[va])
        pred_oof[va] = pred
        rmses.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
        r2s.append(r2_score(y.iloc[va], pred))
    return {
        "MAE_OOF": mean_absolute_error(y, pred_oof),
        "RMSE_OOF": float(np.sqrt(mean_squared_error(y, pred_oof))),
        "R2_OOF": r2_score(y, pred_oof),
        "RMSE (val)": np.mean(rmses),
        "R2 (val)": np.mean(r2s),
        "R2 (val) std": np.std(r2s),
    }


def importancia_por_cv(modelo, X, y, grupos, cv):
    imps = []
    splits = list(cv.split(X, y, groups=grupos))
    for i, (tr, va) in enumerate(splits, start=1):
        print(f"         Fold {i}/{len(splits)} en proceso...", end="\r")
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(
            m, X.iloc[va], y.iloc[va], n_repeats=5, random_state=SEED,
            scoring="neg_root_mean_squared_error", n_jobs=-1)
        imps.append(r.importances_mean)
    print(f"         Folds completados ({len(splits)}/{len(splits)})          ")
    return (pd.Series(np.mean(imps, axis=0), index=X.columns)
              .sort_values(ascending=False))


# ----------------------------------------------------------------------------
# 6. GRÁFICAS
# ----------------------------------------------------------------------------
def graficar_predicciones(y, oof):
    fig, axes = plt.subplots(1, len(oof), figsize=(5 * len(oof), 4.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (nombre, pred) in zip(axes, oof.items()):
        ax.scatter(y, pred, alpha=0.7)
        lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(nombre)
        ax.set_xlabel("Observado")
    axes[0].set_ylabel("Predicho (fuera de muestra)")
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "observado_vs_predicho.png", dpi=150)
    plt.close(fig)


def mapa_bloques(zonas, datos, bloques):
    tmp = datos[[COL_ID]].assign(bloque=bloques)
    ax = zonas.merge(tmp, on=COL_ID).plot(
        column="bloque", cmap="tab10", categorical=True, figsize=(6, 6))
    ax.set_title("Bloques espaciales de validación")
    plt.savefig(DIR_SALIDA / "bloques.png", dpi=150)
    plt.close()


def mapa_residuos(zonas, datos, y, oof):
    residuos = {n: (y - p).to_numpy() for n, p in oof.items()}
    vmax = max(np.abs(r).max() for r in residuos.values())
    fig, axes = plt.subplots(1, len(oof), figsize=(6 * len(oof), 6))
    axes = np.atleast_1d(axes)
    for ax, (nombre, r) in zip(axes, residuos.items()):
        tmp = datos[[COL_ID]].assign(residuo=r)
        zonas.merge(tmp, on=COL_ID).plot(
            column="residuo", cmap="RdBu", vmin=-vmax, vmax=vmax,
            legend=True, ax=ax)
        ax.set_title(f"Residuos: {nombre}\n(observado - predicho)")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "residuos.png", dpi=150)
    plt.close(fig)


def graficar_importancias(importancias, top=15):
    fig, axes = plt.subplots(1, len(importancias),
                             figsize=(6 * len(importancias), 5))
    axes = np.atleast_1d(axes)
    for ax, (nombre, serie) in zip(axes, importancias.items()):
        serie.head(top)[::-1].plot.barh(ax=ax)
        ax.set_title(nombre)
        ax.set_xlabel("Aumento del RMSE al permutar la variable")
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "importancias.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. PREDICCIÓN EN OTRO PERIODO
# ----------------------------------------------------------------------------
def predecir_periodo(modelo, columnas, panel, inicio, fin, vars_dinamicas,
                     estaticas):
    f = construir_features(panel, inicio, fin, vars_dinamicas, estaticas)
    ids = f[COL_ID]
    f = f.reindex(columns=columnas)
    return pd.Series(modelo.predict(f), index=ids, name=f"{COL_Y}_pred")


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    destino_copia = DIR_SALIDA / f"train_{EXP_ID}.py"
    if Path(__file__).resolve() != destino_copia.resolve():
        shutil.copy(__file__, destino_copia)

    print(f"=== Ejecutando {EXP_ID} | Raíz: {ROOT_DIR} ===")

    panel = cargar_panel()
    zonas, coords_zonas, estaticas = cargar_zonas()
    vars_dinamicas = elegir_variables_dinamicas(panel)
    print(f"{len(vars_dinamicas)} variables dinámicas detectadas.")

    X_all = construir_features(panel, INICIO, FIN, vars_dinamicas, estaticas)
    datos, X, y, coords = armar_dataset(X_all, coords_zonas)
    print(f"Zonas con objetivo: {len(datos)} | features: {X.shape[1]}")

    bloques = crear_bloques(coords)
    mapa_bloques(zonas, datos, bloques)
    cv = GroupKFold(n_splits=N_SPLITS)

    # --- Evaluación OOF y por Folds ---
    modelos = definir_modelos()
    resultados, metricas_folds, oof_todos = evaluar(modelos, X, y, bloques, cv)

    print("\n--- Comparación Global Out-Of-Fold (Formato Erick) ---")
    cols_oof = ["MAE_OOF", "RMSE_OOF", "R2_OOF", "reduccion_RMSE_vs_baseline_pct"]
    print(resultados[cols_oof].sort_values("RMSE_OOF").round(6))

    print("\n--- Tabla Completa (OOF + Promedio por Bloques Espaciales) ---")
    print(resultados.round(4))

    resultados.to_csv(DIR_SALIDA / "metricas.csv")
    metricas_folds.to_csv(DIR_SALIDA / "metricas_por_fold.csv", index=False)

    # Guardar predicciones OOF por parcela
    df_oof = pd.DataFrame({
        COL_ID: datos[COL_ID],
        "bloque_espacial": bloques,
        "rendimiento_observado": y,
        **oof_todos,
    })
    df_oof.to_csv(DIR_SALIDA / "predicciones_oof.csv", index=False)

    if BUSCAR_HIPERPARAMETROS:
        anidado = evaluar_anidado_xgb(X, y, bloques, cv)
        print("\nXGBoost con búsqueda anidada:", {k: round(v, 4) for k, v in anidado.items()})

    # --- Diagnóstico gráfico (excluyendo baseline_media) ---
    oof = {n: s for n, s in oof_todos.items() if n != "baseline_media"}
    graficar_predicciones(y, oof)
    mapa_residuos(zonas, datos, y, oof)

    print("\nCalculando importancia por permutación...")
    importancias = {n: importancia_por_cv(modelos[n], X, y, bloques, cv) for n in oof}
    graficar_importancias(importancias)

    mejor = resultados.drop(index="baseline_media")["RMSE_OOF"].idxmin()
    print(f"\nModelo final ({EXP_ID}): {mejor}")
    final = clone(modelos[mejor]).fit(X, y)
    joblib.dump({"modelo": final, "columnas": list(X.columns),
                 "vars_dinamicas": vars_dinamicas},
                DIR_SALIDA / "modelo_final.joblib")
    print(f"Resultados y métricas OOF guardados en: {DIR_SALIDA}")


if __name__ == "__main__":
    main()