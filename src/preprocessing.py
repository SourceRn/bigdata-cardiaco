#INICIO DEL PROYECTO
#CONTADOR DE ERRORES: 1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#LIBRERIA PARA ELIMINAR EL USO DE RUTAS ABSOLUTAS
from pathlib import Path

#DEFINIMOS LA RUTA RELATIVA PARA CARGAR EL ARCHIVO CSV
def obtener_ruta_csv():
    ruta_actual = Path(__file__).resolve()
    carpeta_src = ruta_actual.parent
    carpeta_proyecto = carpeta_src.parent
    DATA_PATH = carpeta_proyecto / "data" / "heart.csv"

    return DATA_PATH

#DEFINIMOS LA FUNCION DE CARGA DE DATOS
def cargar_Datos(ruta_cvs):
    #Df CARGARA EL ARCHIVO CSV
    df = pd.read_csv(ruta_cvs)
    return df

#DEFINIMOS LA FUNCIÓN PARA LIMPIAR LOS DATOS
def limpiar_datos(df):
    #LO QUE ELIMINAMOS SERAN LAS FILAS DUPLICADAS Y LOS VALORES NULOS
    df = df.drop_duplicates()
    df = df.dropna()
    return df

#DEFINIMOS LA FUNCION PARA SEPARAR LOS DATOS DEL CSV [df]
def separar_datos(df):
    #SEPARAMOS LAS VARIABLES EN DOS TIPOS: [X] PISTAS Y [Y] RESPUESTAS
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y

#DEFINIMOS METODO PARA DIVIDIR LOS DATOS EN DOS CATEGORIAS: ENTRENAMIENTO Y PRUEBA
def dividir_datos(X, y):
    #%80 DE LOS DATOS SERAN ENTRENAMIENTO Y EL %20 SE UTILIZARA PARA PRUEBAS
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2, 
        random_state=42
    )

    return X_train, X_test, y_train, y_test
    
#DEFINIMOS LA FUNCION PARA NORMALIZAR LOS DATOS
def escalar_datos(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

#DEFINIMOS LA FUNCION PRINCIPAL PARA PROCESAR LOS DATOS
def preparar_datos(ruta_cvs):
    #ESTA FUNCION INCLUYE TODOS LOS PASOS ANTERIORES PARA PROCESAR LOS DATOS DE MANERA COMPLETA
    df = cargar_Datos(ruta_cvs)

    print("Primeras Filas Del Data Set: ")
    print(df.head())

    print("\nInformación General Del Data Set: ")
    print(df.info())

    df = limpiar_datos(df)
    X, y = separar_datos(df)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

#DEVELOPER AARON YAEL VAZQUEZ RUIZ | 22151154