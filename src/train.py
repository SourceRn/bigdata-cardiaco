#INICIO DEL PROGRAMA
#CONTADOR DE ERRORES: 1
import numpy as np

#IMPORTAMOS LOS DOS ENFOQUES DEL PROYECTO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#MANDAMOS LLAMAR A LA FUNCION PRINCIPAL PARA PROCESAR LOS DATOS
from preprocessing import preparar_datos, obtener_ruta_csv

X_train, X_test, y_train, y_test, scaler = preparar_datos(obtener_ruta_csv())

#CALCULAMOS VARIABLES PARA EL MODELO DE REGRESION LOGISTICA
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

#ENTRENAMOS EL MODELO DE REGRESION LOGISTICA
theta = np.linalg.inv(
    X_train_bias.T.dot(X_train_bias)
).dot(X_train_bias.T).dot(y_train)

#REALIZAMOS PREDICCIONES CON EL MODELO DE REGRESION LOGISTICA
y_pred_linear = X_test_bias.dot(theta)

#CONVERTIMOS LAS PREDICCIONES A CLASES BINARIAS
y_pred_linear_class = (
    y_pred_linear >= 0.5
).astype(int)

#EVALUAMOS LA PRECISION DEL MODELO DE REGRESION LOGISTICA
linear_accuracy = accuracy_score(
    y_test,
    y_pred_linear_class
)

print(f"Precisión del Modelo de Regresión Logística: {linear_accuracy:.4f}")

#ENTRENAMOS EL SEGUNDO MODELO | RANDOM FOREST
#CREAMOS EL MODELO DE RANDOM FOREST CON 100 ÁRBOLES Y UNA SEMILLA ALEATORIA DE 42
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

#ENTRENAMOS EL MODELO DE RANDOM FOREST CON LOS DATOS DE ENTRENAMIENTO
rf_model.fit(X_train, y_train)

#EVALUAMOS LA PRECISION DEL MODELO DE RANDOM FOREST
y_pred_rf = rf_model.predict(X_test)

#CALCULAMOS LA PRECISION DEL MODELO DE RANDOM FOREST
rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print(f"Precisión del Modelo de Random Forest: {rf_accuracy:.4f}")

#DEVELOPER AARON YAEL VAZQUEZ RUIZ | 22151154