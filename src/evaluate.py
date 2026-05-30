#INICIO DEL PROGRAMA
#CONTADOR DE ERRORES: 0
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

#IMPORTAMOS LOS DOS ENFOQUES DEL PROYECTO
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#MANDAMOS LLAMAR A LA FUNCION PRINCIPAL PARA PROCESAR LOS DATOS
from preprocessing import preparar_datos, obtener_ruta_csv

#DEFINIMOS LA RUTA DE SALIDA PARA LAS GRAFICAS
RUTA_SALIDA = Path(__file__).resolve().parent.parent / "results"
RUTA_SALIDA.mkdir(exist_ok=True)

#CARGAMOS LOS DATOS DE ENTRENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test, scaler = preparar_datos(
    obtener_ruta_csv()
)

# =========================================================
# ENTRENAMIENTO DE MODELOS
# =========================================================

#CREAMOS Y ENTRENAMOS EL MODELO DE REGRESION LOGISTICA
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

#PREDICCIONES Y PROBABILIDADES DE REGRESION LOGISTICA
y_pred_log  = log_model.predict(X_test)
y_prob_log  = log_model.predict_proba(X_test)[:, 1]

#CREAMOS Y ENTRENAMOS EL MODELO DE RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

#PREDICCIONES Y PROBABILIDADES DE RANDOM FOREST
y_pred_rf = rf_model.predict(X_test)
y_prob_rf  = rf_model.predict_proba(X_test)[:, 1]

# =========================================================
# METRICAS DE EVALUACION
# =========================================================

print("\n === REGRESION LOGISTICA ===")
print("Accuracy  : ", accuracy_score(y_test, y_pred_log))
print("Precision : ", precision_score(y_test, y_pred_log))
print("Recall    : ", recall_score(y_test, y_pred_log))
print("F1 Score  : ", f1_score(y_test, y_pred_log))
print("AUC-ROC   : ", roc_auc_score(y_test, y_prob_log))
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_log))

print("\n===== RANDOM FOREST =====")
print("Accuracy  : ", accuracy_score(y_test, y_pred_rf))
print("Precision : ", precision_score(y_test, y_pred_rf))
print("Recall    : ", recall_score(y_test, y_pred_rf))
print("F1 Score  : ", f1_score(y_test, y_pred_rf))
print("AUC-ROC   : ", roc_auc_score(y_test, y_prob_rf))
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_rf))

# =========================================================
# GRAFICA 1: MATRICES DE CONFUSION
# =========================================================

#MATRIZ DE CONFUSION — REGRESION LOGISTICA
cm_log   = confusion_matrix(y_test, y_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot()
plt.title("Matriz de Confusion - Regresion Logistica")
plt.savefig(RUTA_SALIDA / "confusion_matrix_logistica.png", bbox_inches="tight")
plt.close()

#MATRIZ DE CONFUSION — RANDOM FOREST
cm_rf   = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot()
plt.title("Matriz de Confusion - Random Forest")
plt.savefig(RUTA_SALIDA / "confusion_matrix_rf.png", bbox_inches="tight")
plt.close()

# =========================================================
# GRAFICA 2: CURVAS ROC
# =========================================================

#CALCULAMOS LOS PUNTOS DE LA CURVA ROC PARA CADA MODELO
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
auc_log = roc_auc_score(y_test, y_prob_log)
auc_rf  = roc_auc_score(y_test, y_prob_rf)

fig, ax = plt.subplots(figsize=(7, 6))

#TRAZAMOS LAS CURVAS ROC DE AMBOS MODELOS
ax.plot(fpr_log, tpr_log, color="#378ADD", linewidth=2,
        label=f"Regresión Logística (AUC = {auc_log:.3f})")
ax.plot(fpr_rf,  tpr_rf,  color="#1D9E75", linewidth=2,
        label=f"Random Forest       (AUC = {auc_rf:.3f})")

#LINEA DE REFERENCIA (CLASIFICADOR ALEATORIO)
ax.plot([0, 1], [0, 1], color="#888780", linewidth=1,
        linestyle="--", label="Clasificador aleatorio")

ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
ax.set_title("Curvas ROC — Comparación de Modelos", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(RUTA_SALIDA / "curvas_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Gráfica guardada: results/curvas_roc.png")

# =========================================================
# GRAFICA 3: IMPORTANCIA DE VARIABLES (RANDOM FOREST)
# =========================================================

import pandas as pd

#OBTENEMOS LOS NOMBRES DE LAS COLUMNAS DEL DATASET ORIGINAL
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import cargar_Datos, obtener_ruta_csv
df_raw = cargar_Datos(obtener_ruta_csv())
nombres_features = df_raw.drop("target", axis=1).columns.tolist()

#EXTRAEMOS Y ORDENAMOS LA IMPORTANCIA DE VARIABLES DEL RANDOM FOREST
importancias = pd.Series(
    rf_model.feature_importances_,
    index=nombres_features
).sort_values(ascending=True)

#MAPEAMOS LOS NOMBRES TECNICOS A DESCRIPCIONES LEGIBLES
etiquetas = {
    "age"     : "Edad",
    "sex"     : "Sexo",
    "cp"      : "Tipo de dolor (cp)",
    "trestbps": "Presión arterial",
    "chol"    : "Colesterol",
    "fbs"     : "Azúcar en ayunas",
    "restecg" : "ECG en reposo",
    "thalach" : "Frec. cardíaca máx.",
    "exang"   : "Angina por ejercicio",
    "oldpeak" : "Depresión ST",
    "slope"   : "Pendiente ST",
    "ca"      : "Vasos principales",
    "thal"    : "Talasemia"
}
importancias.index = [etiquetas.get(i, i) for i in importancias.index]

fig, ax = plt.subplots(figsize=(8, 6))
colores = ["#E24B4A" if v >= importancias.quantile(0.75) else "#378ADD"
           for v in importancias.values]
barras = ax.barh(importancias.index, importancias.values, color=colores, height=0.6)

#AGREGAMOS LOS VALORES AL FINAL DE CADA BARRA
for barra, valor in zip(barras, importancias.values):
    ax.text(valor + 0.002, barra.get_y() + barra.get_height() / 2,
            f"{valor:.3f}", va="center", fontsize=9)

ax.set_xlabel("Importancia relativa", fontsize=11)
ax.set_title("Importancia de Variables — Random Forest", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, importancias.max() * 1.18)

from matplotlib.patches import Patch
leyenda = [
    Patch(color="#E24B4A", label="Alta importancia (top 25%)"),
    Patch(color="#378ADD", label="Importancia estándar")
]
ax.legend(handles=leyenda, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(RUTA_SALIDA / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Gráfica guardada: results/feature_importance.png")

print("\n[EVALUACION COMPLETADA]")
print(f"  Todas las gráficas se encuentran en: results/")

#DEVELOPER AARON YAEL VAZQUEZ RUIZ | 22151154