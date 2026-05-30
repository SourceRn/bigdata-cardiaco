#ANALISIS EXPLORATORIO DE DATOS (EDA)
#CONTADOR DE ERRORES: 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from preprocessing import obtener_ruta_csv

#DEFINIMOS LAS RUTAS DE SALIDA
RUTA_CSV    = obtener_ruta_csv()
RUTA_SALIDA = Path(__file__).resolve().parent.parent / "results"
RUTA_SALIDA.mkdir(exist_ok=True)

#CARGAMOS EL DATASET SIN PREPROCESAR PARA ANALIZAR LOS DATOS EN BRUTO
df = pd.read_csv(RUTA_CSV)

print("=" * 55)
print("  ANALISIS EXPLORATORIO DE DATOS — HEART DISEASE")
print("=" * 55)

# ---------------------------------------------------------
# 1. INFORMACION GENERAL
# ---------------------------------------------------------
print("\n[1] INFORMACION GENERAL DEL DATASET")
print(f"  Filas totales      : {len(df)}")
print(f"  Columnas           : {df.shape[1]}")
print(f"  Filas duplicadas   : {df.duplicated().sum()}")
print(f"  Valores nulos      : {df.isnull().sum().sum()}")

# ---------------------------------------------------------
# 2. ESTADISTICAS DESCRIPTIVAS
# ---------------------------------------------------------
print("\n[2] ESTADISTICAS DESCRIPTIVAS")
print(df.describe().round(2).to_string())

# ---------------------------------------------------------
# 3. TIPOS DE VARIABLES
# ---------------------------------------------------------
print("\n[3] TIPOS DE VARIABLES")
print(df.dtypes.to_string())

# ---------------------------------------------------------
# 4. DISTRIBUCION DE LA VARIABLE OBJETIVO
# ---------------------------------------------------------
print("\n[4] DISTRIBUCION DE LA VARIABLE OBJETIVO (target)")
conteo = df["target"].value_counts()
pct    = df["target"].value_counts(normalize=True) * 100
for val, cnt in conteo.items():
    etiqueta = "Enfermo (1)" if val == 1 else "Sano    (0)"
    print(f"  {etiqueta} : {cnt:>4} registros  ({pct[val]:.1f}%)")

# ---------------------------------------------------------
# 5. GRAFICAS
# ---------------------------------------------------------

#PALETA DE COLORES CONSISTENTE
COLOR_SANO    = "#1D9E75"
COLOR_ENFERMO = "#E24B4A"
COLOR_NEUTRO  = "#378ADD"

#--- 5.1 DISTRIBUCION DEL TARGET ---
fig, ax = plt.subplots(figsize=(5, 4))
etiquetas = ["Sano (0)", "Enfermo (1)"]
colores   = [COLOR_SANO, COLOR_ENFERMO]
barras    = ax.bar(etiquetas, conteo.values, color=colores, width=0.5)
for barra, valor in zip(barras, conteo.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 2,
        str(valor),
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )
ax.set_title("Distribución de la variable objetivo", fontsize=13)
ax.set_ylabel("Número de pacientes")
ax.set_ylim(0, conteo.max() * 1.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RUTA_SALIDA / "eda_distribucion_target.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Gráfica guardada: results/eda_distribucion_target.png")

#--- 5.2 HISTOGRAMAS DE VARIABLES CONTINUAS ---
vars_continuas = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
nombres = {
    "age"     : "Edad",
    "trestbps": "Presión arterial en reposo",
    "chol"    : "Colesterol",
    "thalach" : "Frecuencia cardíaca máxima",
    "oldpeak" : "Depresión ST (oldpeak)"
}
for i, var in enumerate(vars_continuas):
    ax = axes[i]
    sanos    = df[df["target"] == 0][var]
    enfermos = df[df["target"] == 1][var]
    ax.hist(sanos,    bins=20, alpha=0.65, color=COLOR_SANO,    label="Sano")
    ax.hist(enfermos, bins=20, alpha=0.65, color=COLOR_ENFERMO, label="Enfermo")
    ax.set_title(nombres[var], fontsize=11)
    ax.set_xlabel(var)
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[-1].set_visible(False)
fig.suptitle("Distribución de variables continuas por clase", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(RUTA_SALIDA / "eda_histogramas.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Gráfica guardada: results/eda_histogramas.png")

#--- 5.3 MAPA DE CORRELACION ---
fig, ax = plt.subplots(figsize=(10, 8))
corr   = df.corr(numeric_only=True)
im     = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
ticks  = list(corr.columns)
ax.set_xticks(range(len(ticks)))
ax.set_yticks(range(len(ticks)))
ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(ticks, fontsize=10)
for i in range(len(ticks)):
    for j in range(len(ticks)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(val) < 0.7 else "white")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Mapa de correlación entre variables", fontsize=13)
plt.tight_layout()
plt.savefig(RUTA_SALIDA / "eda_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Gráfica guardada: results/eda_correlacion.png")

#--- 5.4 BOXPLOTS: VARIABLES CLAVE POR CLASE ---
vars_box = ["age", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for ax, var in zip(axes, vars_box):
    datos = [df[df["target"] == 0][var], df[df["target"] == 1][var]]
    bp    = ax.boxplot(datos, patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(COLOR_SANO)
    bp["boxes"][1].set_facecolor(COLOR_ENFERMO)
    ax.set_xticklabels(["Sano", "Enfermo"])
    ax.set_title(nombres.get(var, var), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Distribución de variables clave por diagnóstico", fontsize=13)
plt.tight_layout()
plt.savefig(RUTA_SALIDA / "eda_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Gráfica guardada: results/eda_boxplots.png")

print("\n[EDA COMPLETADO]")
print(f"  Todas las gráficas se encuentran en: results/")

#DEVELOPER AARON YAEL VAZQUEZ RUIZ | 22151154