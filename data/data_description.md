# Heart Disease Dataset - Descripción

## Descripción General

Este dataset contiene información médica de pacientes utilizada para el análisis y predicción de enfermedades cardíacas mediante técnicas de Machine Learning.

El objetivo principal del dataset es determinar si un paciente presenta o no una enfermedad cardíaca basándose en distintas variables médicas y fisiológicas.

---

# Objetivo del Dataset

Predecir la presencia de enfermedad cardíaca utilizando modelos de clasificación supervisada.

La variable objetivo es:

| Valor | Significado |
|---|---|
| 0 | Paciente sano |
| 1 | Paciente con enfermedad cardíaca |

---

# Variables del Dataset

| Variable | Descripción |
|---|---|
| age | Edad del paciente |
| sex | Género del paciente |
| cp | Tipo de dolor de pecho |
| trestbps | Presión arterial en reposo |
| chol | Nivel de colesterol |
| fbs | Azúcar en sangre en ayunas |
| restecg | Resultados electrocardiográficos |
| thalach | Ritmo cardíaco máximo alcanzado |
| exang | Angina inducida por ejercicio |
| oldpeak | Depresión ST inducida por ejercicio |
| slope | Pendiente del segmento ST |
| ca | Número de vasos principales coloreados |
| thal | Resultado de prueba de talasemia |
| target | Presencia de enfermedad cardíaca |

---

# Tipo de Problema

El proyecto trabaja un problema de:

## Clasificación Binaria

El sistema debe clasificar pacientes en dos categorías:

- Sano
- Enfermo

---

# Modelos Utilizados

Durante el proyecto se implementaron los siguientes modelos:

- Regresión Logística
- Random Forest

---

# Preprocesamiento Aplicado

Antes del entrenamiento de modelos se realizaron las siguientes tareas:

- Limpieza de datos
- Eliminación de valores nulos
- División entrenamiento/prueba
- Escalado de variables numéricas

---

# Evaluación del Modelo

Los modelos fueron evaluados utilizando:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de Confusión

---

# Fuente del Dataset

Dataset obtenido desde Kaggle:

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

---

# Aplicación del Proyecto

Este sistema puede servir como apoyo para:

- Análisis médico predictivo
- Sistemas de apoyo clínico
- Investigación académica
- Aprendizaje de Machine Learning
- Clasificación de riesgo cardiovascular
