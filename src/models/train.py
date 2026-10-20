"""
Entrenamiento y evaluación espacial Out-Of-Fold del modelo final seleccionado:
Ensamble 50/50 = 0.5 * Ridge (Ciclo, 33 variables) + 0.5 * XGBoost (Mensual + Ene-Mar 2025, 303 variables).
Reproduce los Bloques 50, 58, 59, 69 y 73 FINAL.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.shared.utils import (
    localizar_raiz_proyecto,
    cargar_matrices_ensamble,
    crear_particiones_espaciales,
    calcular_metricas,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

REJILLA_ALPHA_RIDGE = [0.1, 1.0, 10.0, 100.0, 1000.0]

CONFIGURACION_XGB_FINAL = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "random_state": 0,
    "n_jobs": -1,
    "missing": np.nan,
}

PESO_RIDGE = 0.5
PESO_XGB = 0.5


def construir_pipeline_ridge() -> Pipeline:
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("ridge", Ridge(solver="svd")),
    ])


def evaluar_oof_ensamble(
    Xtr_ridge: pd.DataFrame,
    Xtr_xgb: pd.DataFrame,
    y_train: pd.Series,
    grupos_espaciales: pd.Series,
    particiones_espaciales: list,
):
    """
    Evalúa en los 5 folds espaciales externos (138 parcelas):
      1. ridge_ciclo_anidado (33 variables, selección interna de alpha en 3 folds)
      2. xgb_e4_mensual_3m (303 variables, configuración fija)
      3. ensamble_ridge_xgb3m_50_50 (promedio 50/50 sin optimización retrospectiva)
    """
    n = len(y_train)
    pred_oof_ridge = np.full(n, np.nan)
    pred_oof_xgb = np.full(n, np.nan)
    fold_validacion = np.full(n, -1, dtype=int)
    filas_folds = []

    for fold, (tr_idx, va_idx) in enumerate(particiones_espaciales, start=1):
        fold_validacion[va_idx] = fold
        grupos_tr = grupos_espaciales.iloc[tr_idx]
        particiones_internas = list(
            GroupKFold(n_splits=3).split(
                Xtr_ridge.iloc[tr_idx], y_train.iloc[tr_idx], groups=grupos_tr
            )
        )

        # 1. Ridge ciclo anidado (33 variables)
        busqueda_r = GridSearchCV(
            estimator=construir_pipeline_ridge(),
            param_grid={"ridge__alpha": REJILLA_ALPHA_RIDGE},
            scoring="neg_root_mean_squared_error",
            cv=particiones_internas,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        busqueda_r.fit(Xtr_ridge.iloc[tr_idx], y_train.iloc[tr_idx])
        p_val_r = busqueda_r.best_estimator_.predict(Xtr_ridge.iloc[va_idx])
        pred_oof_ridge[va_idx] = p_val_r

        # 2. XGBoost mensual + 3 meses previos (303 variables)
        modelo_x = XGBRegressor(**CONFIGURACION_XGB_FINAL)
        modelo_x.fit(Xtr_xgb.iloc[tr_idx], y_train.iloc[tr_idx])
        p_val_x = modelo_x.predict(Xtr_xgb.iloc[va_idx])
        pred_oof_xgb[va_idx] = p_val_x

        # 3. Ensamble 50/50 en el fold
        p_val_ens = PESO_RIDGE * p_val_r + PESO_XGB * p_val_x

        for nombre_mod, preds_f in [
            ("ridge_ciclo_anidado", p_val_r),
            ("xgb_e4_mensual_3m", p_val_x),
            ("ensamble_ridge_xgb3m_50_50", p_val_ens),
        ]:
            met = calcular_metricas(y_train.iloc[va_idx], preds_f)
            filas_folds.append({
                "modelo": nombre_mod,
                "fold": fold,
                "parcelas": len(va_idx),
                "MAE_val": met["MAE"],
                "RMSE_val": met["RMSE"],
                "R2_val": met["R2"],
                "sesgo_pred_menos_real": met["sesgo_pred_menos_real"],
            })

    pred_oof_ens = PESO_RIDGE * pred_oof_ridge + PESO_XGB * pred_oof_xgb

    df_oof = pd.DataFrame({
        "rendimiento_observado": y_train,
        "fold_validacion": fold_validacion,
        "ridge_ciclo_anidado": pred_oof_ridge,
        "xgb_e4_mensual_3m": pred_oof_xgb,
        "ensamble_ridge_xgb3m_50_50": pred_oof_ens,
    }, index=y_train.index)
    df_oof.index.name = "ID_POLIGONO"

    filas_globales = []
    for nombre_mod in ["ensamble_ridge_xgb3m_50_50", "xgb_e4_mensual_3m", "ridge_ciclo_anidado"]:
        met_g = calcular_metricas(y_train, df_oof[nombre_mod])
        filas_globales.append({
            "modelo": nombre_mod,
            "MAE_OOF": met_g["MAE"],
            "RMSE_OOF": met_g["RMSE"],
            "R2_OOF": met_g["R2"],
            "sesgo_pred_menos_real": met_g["sesgo_pred_menos_real"],
        })

    df_global = pd.DataFrame(filas_globales).sort_values("RMSE_OOF").reset_index(drop=True)
    df_folds = pd.DataFrame(filas_folds)
    return df_global, df_folds, df_oof


def ajustar_modelo_final(
    Xtr_ridge: pd.DataFrame,
    Xtr_xgb: pd.DataFrame,
    y_train: pd.Series,
    grupos_espaciales: pd.Series,
):
    """
    Reproduce el Bloque 73 FINAL:
      - Selecciona alpha de Ridge mediante GroupKFold(n_splits=3) sobre las 138 parcelas
        y reajusta con las 138 etiquetas (alpha = 1.0).
      - Ajusta XGBoost (303 variables) sobre las 138 etiquetas.
    """
    cv_final_3 = list(
        GroupKFold(n_splits=3).split(Xtr_ridge, y_train, groups=grupos_espaciales)
    )
    busqueda_r_final = GridSearchCV(
        estimator=construir_pipeline_ridge(),
        param_grid={"ridge__alpha": REJILLA_ALPHA_RIDGE},
        scoring="neg_root_mean_squared_error",
        cv=cv_final_3,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    busqueda_r_final.fit(Xtr_ridge, y_train)
    modelo_ridge_final = busqueda_r_final.best_estimator_
    alpha_final = float(busqueda_r_final.best_params_["ridge__alpha"])

    modelo_xgb_final = XGBRegressor(**CONFIGURACION_XGB_FINAL)
    modelo_xgb_final.fit(Xtr_xgb, y_train)

    paquete_modelos = {
        "ridge": modelo_ridge_final,
        "xgboost": modelo_xgb_final,
        "columnas_ridge": list(Xtr_ridge.columns),
        "columnas_xgboost": list(Xtr_xgb.columns),
        "peso_ridge": PESO_RIDGE,
        "peso_xgboost": PESO_XGB,
        "alpha_final": alpha_final,
        "cv_results_ridge": pd.DataFrame(busqueda_r_final.cv_results_),
        "nota": (
            "Requiere matrices agregadas con las columnas documentadas; "
            "no incluye extracción de datos brutos."
        ),
    }
    return paquete_modelos


def main():
    raiz = localizar_raiz_proyecto()
    dir_models = raiz / "src" / "results"
    dri_res = raiz / "src" / "results"
    dir_models.mkdir(parents=True, exist_ok=True)
    dri_res.mkdir(parents=True, exist_ok=True)

    print(f"=== Entrenamiento del Modelo Final (Ensamble 50/50) | Raíz: {raiz} ===")

    datos = cargar_matrices_ensamble(raiz)
    Xtr_r = datos["Xtr_ridge"]
    Xtr_x = datos["Xtr_xgb"]
    y_tr = datos["y_train"]

    print(f"Parcelas de entrenamiento: {len(y_tr)} | Vars Ridge: {Xtr_r.shape[1]} | Vars XGBoost: {Xtr_x.shape[1]}")

    grupos, particiones, _, crs_utm = crear_particiones_espaciales(
        datos["gdf_parcelas"], datos["ids_train"]
    )
    print(f"Particiones espaciales creadas ({crs_utm}): 10 bloques KMeans -> 5 folds GroupKFold")

    # 1. Evaluación espacial Out-Of-Fold (Bloques 50, 58 y 59)
    print("\n1. Ejecutando evaluación espacial Out-Of-Fold (5 folds)...")
    df_global, df_folds, df_oof = evaluar_oof_ensamble(Xtr_r, Xtr_x, y_tr, grupos, particiones)

    print("\n--- Resultados Globales OOF (138 Parcelas) ---")
    print(df_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    df_global.to_csv(dri_res / "metricas.csv", index=False, encoding="utf-8-sig")
    df_folds.to_csv(dri_res / "metricas_por_fold.csv", index=False, encoding="utf-8-sig")
    df_oof.to_csv(dri_res / "predicciones_oof.csv", index=True, encoding="utf-8-sig")

    # 2. Ajuste final con las 138 parcelas (Bloque 73 FINAL)
    print("\n2. Ajustando modelos finales con las 138 parcelas etiquetadas...")
    paquete = ajustar_modelo_final(Xtr_r, Xtr_x, y_tr, grupos)
    print(f"   -> Alpha final seleccionado para Ridge: {paquete['alpha_final']}")

    paquete_serializar = {k: v for k, v in paquete.items() if k != "cv_results_ridge"}
    ruta_joblib_models = dir_models / "modelos_ensamble.joblib"
    ruta_joblib_exp005 = dri_res / "modelo_final.joblib"

    joblib.dump(paquete_serializar, ruta_joblib_models)
    joblib.dump(paquete_serializar, ruta_joblib_exp005)
    paquete["cv_results_ridge"].to_csv(
        dri_res / "seleccion_alpha_final.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n[OK] Paquete de modelos guardado en: {ruta_joblib_models}")
    print(f"[OK] Métricas y predicciones OOF guardadas en: {dri_res}")


if __name__ == "__main__":
    main()