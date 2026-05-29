#INICIO DEL PROGRAMA
#CONTADOR DE ERRORES: 0
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#MANDAMOS LLAMAR A LA FUNCION PRINCIPAL PARA PROCESAR LOS DATOS
from preprocessing import preparar_datos, obtener_ruta_csv

#CARGAMOS LOS DATOS DE ENTRENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test, scaler = preparar_datos(obtener_ruta_csv())


# ===========================================================
# MODELO 1: REGRESION LOGISTICA IMPLEMENTADA DESDE CERO
# ===========================================================

#FUNCION SIGMOIDE: CONVIERTE CUALQUIER VALOR A UN RANGO ENTRE 0 Y 1
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

#FUNCION DE ENTRENAMIENTO: GRADIENTE DESCENDENTE
def entrenar_regresion_logistica(X, y, tasa_aprendizaje=0.1, iteraciones=1000):
    #AGREGAMOS EL TERMINO DE SESGO (BIAS) A LOS DATOS DE ENTRENAMIENTO
    X_b = np.c_[np.ones(X.shape[0]), X]
    #INICIALIZAMOS LOS PESOS EN CERO
    theta = np.zeros(X_b.shape[1])
    m = len(y)

    for _ in range(iteraciones):
        #CALCULAMOS LA PREDICCION CON LA FUNCION SIGMOIDE
        z = X_b.dot(theta)
        y_pred = sigmoide(z)

        #CALCULAMOS EL GRADIENTE Y ACTUALIZAMOS LOS PESOS
        gradiente = X_b.T.dot(y_pred - y) / m
        theta -= tasa_aprendizaje * gradiente

    return theta

#FUNCION DE PREDICCION
def predecir(X, theta, umbral=0.5):
    #AGREGAMOS EL TERMINO DE SESGO (BIAS) A LOS DATOS DE PRUEBA
    X_b = np.c_[np.ones(X.shape[0]), X]
    probabilidades = sigmoide(X_b.dot(theta))
    #SI LA PROBABILIDAD ES MAYOR AL UMBRAL, LA CLASE ES 1, DE LO CONTRARIO ES 0
    return (probabilidades >= umbral).astype(int)

#ENTRENAMOS EL MODELO DE REGRESION LOGISTICA
print("\n=== ENTRENANDO REGRESION LOGISTICA ===")
theta = entrenar_regresion_logistica(
    X_train,
    y_train,
    tasa_aprendizaje=0.1,
    iteraciones=1000
)

#REALIZAMOS PREDICCIONES CON EL MODELO DE REGRESION LOGISTICA
y_pred_log = predecir(X_test, theta)

#EVALUAMOS LA PRECISION DEL MODELO DE REGRESION LOGISTICA
log_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Precisión del Modelo de Regresión Logística: {log_accuracy:.4f}")


# ===========================================================
# MODELO 2: RANDOM FOREST
# ===========================================================

#CREAMOS EL MODELO DE RANDOM FOREST CON 100 ÁRBOLES Y UNA SEMILLA ALEATORIA DE 42
print("\n=== ENTRENANDO RANDOM FOREST ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

#ENTRENAMOS EL MODELO DE RANDOM FOREST CON LOS DATOS DE ENTRENAMIENTO
rf_model.fit(X_train, y_train)

#REALIZAMOS PREDICCIONES CON EL MODELO DE RANDOM FOREST
y_pred_rf = rf_model.predict(X_test)

#CALCULAMOS LA PRECISION DEL MODELO DE RANDOM FOREST
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Precisión del Modelo de Random Forest: {rf_accuracy:.4f}")


# ===========================================================
# COMPARACION FINAL
# ===========================================================

print("\n========== COMPARACION DE MODELOS ==========")
print(f"  Regresión Logística (manual): {log_accuracy:.4f}")
print(f"  Random Forest (sklearn):      {rf_accuracy:.4f}")

mejor = "Regresión Logística" if log_accuracy > rf_accuracy else "Random Forest"
print(f"\n  Mejor modelo: {mejor}")
print("=============================================")

#DEVELOPER AARON YAEL VAZQUEZ RUIZ | 22151154