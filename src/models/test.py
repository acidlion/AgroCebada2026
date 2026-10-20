"""
Generación y verificación de las 59 predicciones finales del conjunto de prueba
utilizando el paquete entrenado 'modelos_ensamble.joblib' (Bloque 73 FINAL).
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.shared.utils import localizar_raiz_proyecto, cargar_matrices_ensamble


def generar_entrega_final():
    raiz = localizar_raiz_proyecto()
    dir_models = raiz / "src" / "results"
    dri_res = raiz / "src" / "results"
    dir_models.mkdir(parents=True, exist_ok=True)
    dri_res.mkdir(parents=True, exist_ok=True)

    ruta_joblib = dir_models / "modelos_ensamble.joblib"
    if not ruta_joblib.is_file():
        ruta_joblib = dri_res / "modelo_final.joblib"
    if not ruta_joblib.is_file():
        raise FileNotFoundError(
            "No se encontró modelos_ensamble.joblib. Ejecuta primero: python -m src.models.train"
        )

    print(f"=== Generando Predicciones Finales (59 Parcelas) | Modelo: {ruta_joblib.name} ===")

    paquete = joblib.load(ruta_joblib)
    datos = cargar_matrices_ensamble(raiz)

    ids_train = datos["ids_train"]
    ids_pred = datos["ids_pred"]
    Xpr_r = datos["Xpr_ridge"]
    Xpr_x = datos["Xpr_xgb"]

    # 1. Controles estrictos de integridad (Bloque 73 FINAL)
    assert len(ids_pred) == 59 and ids_pred.is_unique, "Se esperaban 59 IDs objetivo únicos."
    assert set(ids_train).isdisjoint(set(ids_pred)), "Fuga de IDs entre entrenamiento y predicción."
    assert list(Xpr_r.columns) == list(paquete["columnas_ridge"]), "Las columnas de Ridge no coinciden."
    assert list(Xpr_x.columns) == list(paquete["columnas_xgboost"]), "Las columnas de XGBoost no coinciden."
    assert not np.isinf(Xpr_r.to_numpy()).any() and not np.isinf(Xpr_x.to_numpy()).any()

    # 2. Inferencia por componente y combinación 50/50
    pred_r = np.asarray(paquete["ridge"].predict(Xpr_r), dtype=float).reshape(-1)
    pred_x = np.asarray(paquete["xgboost"].predict(Xpr_x), dtype=float).reshape(-1)

    peso_r = float(paquete.get("peso_ridge", 0.5))
    peso_x = float(paquete.get("peso_xgboost", 0.5))
    pred_ensamble = peso_r * pred_r + peso_x * pred_x

    assert np.isfinite(pred_ensamble).all(), "Hay predicciones no finitas."
    if (pred_ensamble < 0).any():
        raise RuntimeError("Se detectaron predicciones negativas; revisar entradas.")

    detalle_59 = pd.DataFrame({
        "pred_ridge": pred_r,
        "pred_xgboost": pred_x,
        "pred_ensamble": pred_ensamble,
        "desacuerdo_componentes": np.abs(pred_r - pred_x),
    }, index=ids_pred)

    entrega_59 = pd.DataFrame({
        "ID_PARCELA": ids_pred.to_numpy(),
        "RENDIMIENTO": pred_ensamble,
    })

    # 3. Exportar CSV oficial en models/ y en reports/resultados/exp_005/
    ruta_csv_models = dir_models / "ENTREGA_ENSAMBLE_59_PARCELAS.csv"
    ruta_csv_exp005 = dri_res / "ENTREGA_ENSAMBLE_59_PARCELAS.csv"

    for ruta_csv in (ruta_csv_models, ruta_csv_exp005):
        entrega_59.to_csv(ruta_csv, index=False, encoding="utf-8-sig", float_format="%.10f")

    detalle_59.to_csv(dri_res / "detalle_componentes_59.csv", index=True, index_label="ID_POLIGONO", encoding="utf-8-sig")

    # 4. Releer el CSV para verificar integridad de escritura
    verificacion = pd.read_csv(ruta_csv_models, dtype={"ID_PARCELA": str})
    assert verificacion.shape == (59, 2)
    assert list(verificacion.columns) == ["ID_PARCELA", "RENDIMIENTO"]
    assert np.allclose(verificacion["RENDIMIENTO"], entrega_59["RENDIMIENTO"], rtol=0, atol=1e-9)

    sha256_csv = hashlib.sha256(ruta_csv_models.read_bytes()).hexdigest()
    manifiesto = {
        "procedimiento": "ensamble_ridge_xgb3m_50_50",
        "n_train": len(ids_train),
        "n_prediccion": len(ids_pred),
        "variables_ridge": len(paquete["columnas_ridge"]),
        "variables_xgboost": len(paquete["columnas_xgboost"]),
        "alpha_final_ridge": float(paquete["alpha_final"]),
        "pesos": {"ridge": peso_r, "xgboost": peso_x},
        "columnas_entrega": ["ID_PARCELA", "RENDIMIENTO"],
        "unidad": "t/ha",
        "sha256_csv": sha256_csv,
    }
    (dri_res / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- Resumen Estadístico de las 59 Estimaciones (t/ha) ---")
    print(entrega_59["RENDIMIENTO"].describe().round(6).to_string())
    print(f"\n[OK] Entrega final verificada (59 filas x 2 columnas) | SHA-256: {sha256_csv[:16]}...")
    print(f"[OK] CSV guardado en: {ruta_csv_models}")
    print(f"[OK] Copia en resultados: {ruta_csv_exp005}")


if __name__ == "__main__":
    generar_entrega_final()