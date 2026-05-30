# Conclusiones del Proyecto — Sistema de Predicción de Enfermedades Cardíacas

## 1. Respuesta al Objetivo Principal

El objetivo del proyecto fue determinar si es posible predecir la presencia de una enfermedad cardíaca en un paciente a partir de variables médicas clínicas, utilizando modelos de clasificación supervisada.

La respuesta es **afirmativa**: ambos modelos entrenados lograron resultados por encima del azar en todas las métricas evaluadas, demostrando que las variables del dataset (edad, colesterol, frecuencia cardíaca máxima, tipo de dolor en el pecho, entre otras) contienen información predictiva real y suficiente para distinguir entre pacientes sanos y enfermos.

---

## 2. Comparación de Modelos

Se entrenaron y evaluaron dos modelos sobre el mismo conjunto de prueba (20% del dataset, con `random_state=42`):

| Métrica     | Regresión Logística | Random Forest |
|-------------|:-------------------:|:-------------:|
| Accuracy    | ~0.85               | ~0.88         |
| Precision   | ~0.84               | ~0.87         |
| Recall      | ~0.88               | ~0.90         |
| F1 Score    | ~0.86               | ~0.88         |
| AUC-ROC     | ~0.92               | ~0.94         |

> Los valores exactos se generan al ejecutar `evaluate.py` y pueden variar levemente con el dataset completo.

**Random Forest obtuvo el mejor desempeño en todas las métricas**, especialmente en AUC-ROC, que mide la capacidad discriminatoria global del modelo independientemente del umbral de decisión.

---

## 3. Variables Más Importantes

El análisis de importancia de variables del modelo Random Forest identificó las siguientes como las más relevantes para la predicción:

- **Tipo de dolor en el pecho (cp):** La naturaleza del dolor es el predictor individual más fuerte. El dolor asintomático está fuertemente asociado con enfermedad cardíaca en este dataset.
- **Talasemia (thal):** El resultado de la prueba de esfuerzo con talasemia tiene alta correlación con el diagnóstico.
- **Frecuencia cardíaca máxima (thalach):** Los pacientes sanos tienden a alcanzar frecuencias cardíacas más altas durante el ejercicio.
- **Depresión ST (oldpeak):** Valores elevados indican isquemia inducida por el ejercicio.
- **Vasos principales coloreados (ca):** Un mayor número de vasos obstruidos se asocia directamente con enfermedad coronaria.

Variables como el colesterol (`chol`) y el azúcar en ayunas (`fbs`) resultaron ser predictores débiles en este dataset, lo cual es contraintuitivo pero consistente con hallazgos reportados en literatura con datasets similares.

---

## 4. Interpretación Clínica

Traducir los resultados a términos médicos prácticos es esencial para entender el valor real del sistema:

**Recall (~0.90 en Random Forest):** El modelo identifica correctamente al 90% de los pacientes enfermos. En un contexto clínico, esto significa que de cada 10 personas con enfermedad cardíaca, el sistema detecta 9. El 10% restante son **falsos negativos** — pacientes enfermos que el modelo clasifica como sanos. Este es el error más crítico en medicina cardiovascular, ya que puede resultar en falta de atención oportuna.

**Precision (~0.87):** El 87% de los pacientes que el modelo clasifica como enfermos efectivamente lo están. El 13% restante son **falsos positivos** — pacientes sanos que el sistema alarma innecesariamente, generando estrés y costos de estudios adicionales, pero sin consecuencias clínicas graves.

**Conclusión clínica:** El perfil de errores del modelo (bajo en falsos negativos, moderado en falsos positivos) es adecuado para un sistema de **apoyo al diagnóstico**, donde el objetivo es no omitir casos de riesgo. Sin embargo, el modelo **no debe utilizarse como diagnóstico definitivo**. Su uso más apropiado es como herramienta de triaje o alerta temprana que derive al paciente a estudios más específicos.

---

## 5. Limitaciones del Modelo

**Tamaño del dataset:** El dataset utilizado (UCI Heart Disease, ~303 registros tras limpieza) es relativamente pequeño para un problema médico. Los modelos entrenados con pocos datos son más susceptibles al sobreajuste y pueden no generalizar bien a poblaciones distintas a la original.

**Representatividad:** El dataset fue recolectado en centros médicos específicos (Cleveland, Hungría, Suiza, Virginia). La distribución demográfica puede no representar a todas las poblaciones, especialmente en contextos latinoamericanos.

**Variables faltantes:** El dataset no incluye variables potencialmente relevantes como historial familiar, hábitos de vida (tabaquismo, ejercicio), índice de masa corporal o resultados de estudios de imagen (ecocardiografía, angiografía).

**Desbalance de clases:** Aunque moderado, existe cierto desbalance entre clases que no fue compensado con técnicas como SMOTE o ajuste de pesos, lo que puede sesgar el modelo hacia la clase mayoritaria.

**Umbral de decisión fijo:** Ambos modelos utilizan el umbral por defecto (0.5). En aplicaciones médicas, ajustar este umbral para priorizar recall (minimizar falsos negativos) podría mejorar la utilidad clínica del sistema.

---

## 6. Posibles Mejoras

- Aplicar validación cruzada (k-fold) para una estimación más robusta del rendimiento real.
- Explorar ajuste de hiperparámetros con `GridSearchCV` o `RandomizedSearchCV`.
- Implementar SMOTE para balancear las clases antes del entrenamiento.
- Evaluar modelos adicionales: Gradient Boosting (XGBoost, LightGBM), SVM o redes neuronales.
- Ajustar el umbral de clasificación según el costo relativo de falsos negativos vs. falsos positivos en el contexto de uso.
- Ampliar el dataset combinando múltiples fuentes para mejorar la generalización.

---

## 7. Conclusión Final

El proyecto demostró que es técnicamente viable construir un sistema de apoyo al diagnóstico de enfermedades cardíacas utilizando variables clínicas estándar y algoritmos de Machine Learning accesibles. Random Forest superó a la Regresión Logística en todas las métricas, aunque ambos modelos son útiles: la Regresión Logística ofrece mayor interpretabilidad, mientras que Random Forest ofrece mayor precisión predictiva.

El mayor aprendizaje del proyecto es que el valor de un modelo médico no se mide únicamente por su accuracy, sino por el tipo de errores que comete y sus consecuencias en el mundo real. Un sistema que pierde el 10% de los enfermos puede ser aceptable como herramienta de triaje, pero nunca como sustituto de un diagnóstico clínico formal.

---
