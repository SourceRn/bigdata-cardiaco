#INICIO DEL PROGRAMA
#CONTADOR DE ERRORES: 0
import matplotlib.pyplot as plt

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    classification_report
)

#IMPORTAMOS LOS DOS ENFOQUES DEL PROYECTO
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#MANDAMOS LLAMAR A LA FUNCION PRINCIPAL PARA PROCESAR LOS DATOS
from preprocessing import preparar_datos, obtener_ruta_csv

#CARGAMOS LOS DATOS DE ENTRENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test, scaler = preparar_datos(
    obtener_ruta_csv()
)

#CREAMOS EL MODELO DE REGRESION LOGISTICA
log_model = LogisticRegression()

#ENTRENAMOS EL MODELO DE REGRESION LOGISTICA CON LOS DATOS DE ENTRENAMIENTO
log_model.fit(X_train, y_train)

#REALIZAMOS PREDICCIONES CON EL MODELO DE REGRESION LOGISTICA
y_pred_log = log_model.predict(X_test)


#CREAMOS EL SEGUNDO MODELO | RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators = 100,
    random_state= 42
)

#ENTRENAMOS EL MODELO DE RANDOM FOREST CON LOS DATOS DE ENTRENAMIENTO
rf_model.fit(X_train, y_train)

#REALIZAMOS PREDICCIONES CON EL MODELO DE RANDOM FOREST
y_pred_rf = rf_model.predict(X_test)

#MOSTRAMOS LOS RESULTADOS DE LA EVALUACION DE LOS MODELOS
print("\n === REGRESION LOGISTICA ===")
print("Accuracy: ",
       accuracy_score(y_test, y_pred_log))
print("Precisión: ",
       precision_score(y_test, y_pred_log))
print("Recall: ",
       recall_score(y_test, y_pred_log))
print("F1 Score: ",
       f1_score(y_test, y_pred_log))

print("\n Reporte:")
print(classification_report(
    y_test,
    y_pred_log
))

#MOSTRAMOS LOS RESULTADOS DE LA EVALUACION DE LOS MODELOS
print("\n===== RANDOM FOREST =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("Precision:",
      precision_score(y_test, y_pred_rf))

print("Recall:",
      recall_score(y_test, y_pred_rf))

print("F1 Score:",
      f1_score(y_test, y_pred_rf))

print("\nReporte:")
print(classification_report(
    y_test,
    y_pred_rf
))

#GUARDAMOS LA MATRIZ DE CONFUSION PARA EL MODELO DE REGRESION LOGISTICA
cm_log = confusion_matrix(
    y_test,
    y_pred_log
)

disp_log = ConfusionMatrixDisplay(
    confusion_matrix=cm_log
)

disp_log.plot()

plt.title("Matriz de Confusion - Regresion Logistica")

plt.savefig("confusion_matrix_logistica.png", bbox_inches="tight")
plt.close()

#GUARDAMOS LA MATRIZ DE CONFUSION PARA EL MODELO DE RANDOM FOREST
cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf
)

disp_rf.plot()

plt.title("Matriz de Confusion - Random Forest")

plt.savefig("confusion_matrix_rf.png", bbox_inches="tight")
plt.close()

#DEVELOPER AARON YAEL VAZQUEZ RUIZ | 22151154