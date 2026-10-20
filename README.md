# AgroCebada 2026: Estimación de Rendimiento de Cebada mediante Percepción Remota y Aprendizaje Automático

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-blueviolet)](https://github.com/astral-sh/uv)
[![Framework: Scikit-Learn & XGBoost](https://img.shields.io/badge/framework-scikit--learn%20%7C%20xgboost-orange)](https://scikit-learn.org/)
[![Dashboard: Dash / Plotly](https://img.shields.io/badge/dashboard-Dash%20%7C%20FastAPI-00bcd4)](https://plotly.com/dash/)
[![Challenge: Reto AgroCebada FIRA 2026](https://img.shields.io/badge/challenge-FIRA%20AgroCebada%202026-green)](https://www.fira.gob.mx)

Repositorio oficial del equipo **Cebadatos** para el **Reto AgroCebada FIRA 2026**. El proyecto implementa un pipeline analítico y geoespacial de extremo a extremo que procesa series temporales de sensores remotos ópticos y térmicos (Sentinel-2, Landsat, Planet), variables climáticas (precipitación CHIRPS, Tmax y Tmin) y topografía (INEGI CEM 4.0) para predecir con alta precisión el rendimiento de grano de cebada ($t/ha$) en 197 parcelas agrícolas ubicadas en Hidalgo, Tlaxcala y Puebla durante el ciclo agrícola 2025.

El núcleo analítico está compuesto por un **Ensamble Híbrido 50/50** entre un modelo lineal regularizado (`Ridge`, 33 variables condensadas del ciclo) y un modelo de potenciación de gradiente (`XGBoost`, 303 variables mensuales desagregadas con ventana de pre-siembra). Los resultados se encuentran desplegados en una interfaz interactiva (*Dashboard*) orientada a mitigar la asimetría de información y facilitar la inclusión financiera y colocación de crédito rural formal respaldado por FIRA.

El despliegue de la aplicación se hizo a través de https://render.com/
# Al abrir la aplicación desde https://tlayolli-agro.onrender.com/ después de un periodo de inactividad, puede tomar un tiempo de varios segundos en cargar.
---

## Métricas Clave y Modelo Ganador

El modelo fue seleccionado mediante validación cruzada espacial estricta fuera de muestra (*Out-Of-Fold*, OOF) sobre las **138 parcelas etiquetadas**, agrupadas en **10 clústeres geográficos KMeans** y evaluadas en **5 particiones GroupKFold** para garantizar la capacidad de generalización regional:

| Procedimiento | Representación de Datos | MAE OOF ($t/ha$) | RMSE OOF ($t/ha$) | $R^2$ OOF Espacial |
| :--- | :--- | :---: | :---: | :---: |
| **Ensamble 50/50 (Ganador)** | **Híbrida: 33 vars (ciclo) + 303 vars (mensual)** | **0.4489** | **0.6114** | **0.4803** |
| XGBoost Mensual Ampliado | Mensual Ene–Oct 2025 (303 vars) | 0.4591 | 0.6267 | 0.4538 |
| Ridge Ciclo Anidado | Resumen Ciclo Abr–Oct 2025 (33 vars) | 0.4701 | 0.6287 | 0.4504 |
| XGBoost Mensual Fijo | Mensual Abr–Oct 2025 (213 vars) | 0.4731 | 0.6370 | 0.4359 |
| Random Forest Mensual | Mensual Abr–Oct 2025 (213 vars) | 0.5048 | 0.6781 | 0.3607 |
| Baseline Ingenuo (Media) | Predicción de la media por fold | 0.8579 | 0.9589 | -0.2785 |

* **Evaluación retrospectiva complementaria (Reserva 28 parcelas por ID [EXP-006]):** $\text{RMSE} = 0.3671\text{ t/ha}$, $\text{MAE} = 0.2888\text{ t/ha}$, $R^2 = 0.7187$.
* **Resumen de entrega en las 59 parcelas objetivo:** Media de $3.9236\text{ t/ha}$, Mediana de $4.2164\text{ t/ha}$, Desv. Estándar de $0.6759\text{ t/ha}$ (Rango: $2.8344\text{ a }4.8200\text{ t/ha}$).

---

## Arquitectura del Repositorio

El proyecto mantiene una estructura modular, desacoplando los datos crudos, las matrices limpias, los pipelines de entrenamiento/inferencia, el dashboard y la bitácora reproducible de experimentos:

```text
AgroCebada2026/
│
├── data/                                  # Gestión de datos del proyecto
│   ├── raw/                               # Insumos originales (SHP, CSV satelitales, TIFs) [ignorado en .gitignore]
│   └── processed/                         # Matrices estructuradas listas para modelar (rastreado en git)
│       ├── tabla_integrada_parcela_ciclo_2025.csv    # 33 predictoras del ciclo (para Ridge)
│       ├── tabla_parcela_meses_2025_sin_imputar.csv  # 213 predictoras mensuales Abr–Oct 2025
│       ├── tabla_mensual_303_xgb_2025.csv            # 303 predictoras mensuales Ene–Oct 2025 (para XGBoost)
│       ├── diccionario_tabla_parcela_meses_2025.csv  # Metadatos de variables mensuales
│       ├── panel.parquet                             # Cubo panel espacio-temporal
│       └── estaticos.parquet                         # Variables topográficas y polígonos
│
├── notebooks/                             # Flujo experimental interactivo (Jupyter)
│   ├── 01_eda_erick.ipynb                 # EDA, calidad de datos y extracción raster (Bloques 1 al 36)
│   └── 02_model_experiments.ipynb         # Modelado, validación espacial y ensamble (Bloques 37 al 73)
│
├── src/                                   # Código fuente modular en producción (Módulos de Python)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── pipeline.py                    # Extracción geoespacial con exactextract y ensamblado de matrices
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                       # Validación espacial anidada y ajuste final del Ensamble 50/50
│   │   └── test.py                        # Inferencia sobre las 59 parcelas objetivo y exportación verificada
│   ├── shared/
│   │   ├── __init__.py
│   │   └── utils.py                       # Particiones espaciales (KMeans + GroupKFold), rutas y métricas OOF
│   └── results/                           # Carpeta de resultados obtenidos en la ejecución desde src
│
├── Dashboard/                             # Plataforma Web Interactiva (Dash / FastAPI)
│   ├── assets/                            # Hojas de estilo CSS, geometrías GeoJSON y atribuciones
│   ├── parcelas/                          # Capa vectorial Shapefile de las 197 parcelas (SHP, DBF, PRJ, SHX)
│   ├── Parquet/                           # Datos preprocesados en formato columnar (panel.parquet, estaticos.parquet)
│   ├── Precipitacion/                     # Insumos climáticos CHIRPS (Precipitacion_mensual/ 2022-2025 en .tif)
│   ├── Temperatura/                       # Insumos térmicos (Reto_AgroCebada_Temperatura_2022_2025/ Tmax y Tmin)
│   ├── Topografia/                        # Rasters de terreno (Reto_AgroCebada_Topografia_INEGI_CEM4/ 120m)
│   ├── Topologia/                         # Capas de relieve y curvas de nivel auxiliares
│   ├── app.py                             # Aplicación principal y servidor del Dashboard interactivo
│   ├── parcelas.py                        # Procesamiento cartográfico y renderizado espacial de parcelas
│   ├── preprocesado.py                    # Formateo, caché y filtros de datos en tiempo de ejecución
│   └── datos_prueba_jurado.csv            # Dataset de prueba y evaluación para la demostración ante el jurado
│
├── reports/                               # Documentación formal de la investigación
│   └── resultados/                        # Jerarquía auditable por experimento
│       ├── eda_bloques_01_36/             # Auditoría de nubosidad, calidad y extracción
│       │   └── revisiones_bloques/        # Reportes detallados de verificación de los Bloques 4 al 36
│       ├── exp_001/                       # EXP-001: Baseline inicial con 74 variables
│       │   ├── metricas.csv
│       │   ├── metricas_por_fold.csv
│       │   ├── predicciones_oof.csv
│       │   ├── modelo_final.joblib
│       │   ├── bloques.png
│       │   ├── observado_vs_predicho.png
│       │   ├── residuos.png
│       │   └── importancias.png
│       ├── exp_002/                       # EXP-002: Fusión multimodal bruta (449 variables)
│       │   ├── bloque_43_referencia_modelos/
│       │   ├── bloque_44_validacion_anidada/
│       │   ├── bloque_45_diagnostico_espacial/
│       │   ├── bloque_46_diagnostico_predictores/
│       │   ├── bloque_47_sensibilidad_predicciones/
│       │   ├── bloque_48_entrega_predicciones_xgb/
│       │   └── metricas.csv
│       ├── exp_003/                       # EXP-003: Comparativa multialgorítmica (Ridge, Lasso, CatBoost, MLP)
│       │   ├── bloque_49_red_neuronal/
│       │   ├── bloque_50_ridge_rendimiento/
│       │   ├── bloque_51_catboost_rendimiento/
│       │   ├── bloque_52_comparacion_consolidada/
│       │   ├── bloque_53_entrega_ridge_ciclo/
│       │   ├── bloque_54_revision_ridge_final/
│       │   ├── bloque_55_lasso_rendimiento/
│       │   └── metricas.csv
│       ├── exp_004/                       # EXP-004: Antecedentes temporales de pre-siembra (303 variables)
│       │   ├── bloques_56_57_58_temporal/
│       │   ├── bloque_60A_antecedentes_ridge/
│       │   └── metricas.csv
│       ├── exp_005/                       # EXP-005: Ensamble 50/50 definitivo y exportación final
│       │   ├── bloque_59_ensamble_ridge_xgb/
│       │   ├── bloque_60B_auditoria_errores/
│       │   ├── bloque_61_cierre_primera_etapa/
│       │   ├── bloque_62A_simplificacion_ridge/
│       │   ├── bloque_62B_kernel_ridge/
│       │   ├── bloque_62C_ventanas_temporales/
│       │   ├── bloque_63_consolidacion_segunda_etapa/
│       │   ├── bloque_63B_trazabilidad_vi6t/
│       │   ├── bloque_64_ridge_sin_vi6t/
│       │   ├── bloque_65_adaptacion_agroclimatica/
│       │   ├── bloque_66_pls_ciclo_mensual/
│       │   ├── bloque_67_svr_ciclo_mensual/
│       │   ├── bloques_68A_68B_combinacion_pls/
│       │   ├── bloque_69_consolidacion_final/
│       │   ├── bloque_73_entrega_final_ensamble/
│       │   ├── metricas.csv
│       │   ├── metricas_por_fold.csv
│       │   ├── predicciones_oof.csv
│       │   └── ENTREGA_ENSAMBLE_59_PARCELAS.csv
│       └── exp_006/                       # EXP-006: Benchmarking con ExtraTrees y reserva de 28 parcelas
│           ├── bloque_71_ensamble_reserva_28/
│           ├── bloque_72_extratrees_bases_propias/
│           ├── metricas.csv
│           ├── metricas_reserva_28.csv
│           └── predicciones_oof.csv
│
├── models/                                # Artefactos serializados y entregables oficiales
│   ├── modelos_ensamble.joblib            # Pipeline final serializado (Ridge + XGBoost 50/50, alpha = 1.0)
│   └── ENTREGA_ENSAMBLE_59_PARCELAS.csv   # Predicciones oficiales para FIRA (ID_PARCELA, RENDIMIENTO)
│
├── .env.example                           # Plantilla de configuración de variables de entorno
├── .gitignore                             # Reglas de exclusión de Git
├── .python-version                        # Versión estándar de Python del entorno virtual
├── pyproject.toml                         # Especificación declarativa de dependencias con uv
├── uv.lock                                # Bloqueo determinista de dependencias para reproducibilidad estricta
└── README.md                              # Documentación del proyecto
