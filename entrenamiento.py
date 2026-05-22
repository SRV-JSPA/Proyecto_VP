import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


GESTOS = {
    0: "palma_abierta",
    1: "puno_cerrado",
    2: "indice_arriba",
    3: "pulgar_arriba",
    4: "pulgar_abajo",
    5: "pinch",
    6: "paz",
    7: "anime",
    8: "ok_sign",
}


DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_BASE, "data")
DIRECTORIO_FIGURAS = os.path.join(DIRECTORIO_BASE, "figuras")
DIRECTORIO_MODELOS = os.path.join(DIRECTORIO_BASE, "modelos")

ARCHIVO_LIMPIO = os.path.join(DIRECTORIO_DATOS, "gestos_limpio.csv")


def cargar_datos():

    if not os.path.exists(ARCHIVO_LIMPIO):
        return None, None, None

    dataframe = pd.read_csv(ARCHIVO_LIMPIO)

    columnas_landmarks = []
    columnas_dedos = []

    for columna in dataframe.columns:
        if columna.startswith("punto"):
            columnas_landmarks.append(columna)

    for columna in dataframe.columns:
        if columna.startswith("dedo_"):
            columnas_dedos.append(columna)

    columnas_caracteristicas = columnas_landmarks + columnas_dedos

    if "etiqueta" not in dataframe.columns:
        return None, None, None

    if len(columnas_caracteristicas) == 0:
        return None, None, None

    datos_x = dataframe[columnas_caracteristicas].values
    datos_y = dataframe["etiqueta"].values

    return datos_x, datos_y, columnas_caracteristicas


def dividir_datos(datos_x, datos_y):

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        datos_x,
        datos_y,
        test_size=0.2,
        random_state=42,
        stratify=datos_y,
    )

    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba


def escalar_datos(x_entrenamiento, x_prueba):

    escalador = StandardScaler()

    x_entrenamiento_escalado = escalador.fit_transform(x_entrenamiento)
    x_prueba_escalado = escalador.transform(x_prueba)

    return x_entrenamiento_escalado, x_prueba_escalado, escalador


def crear_modelos():

    modelo_random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    modelo_svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42,
    )

    modelo_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    modelos = {
        "Random Forest": modelo_random_forest,
        "SVM": modelo_svm,
        "MLP": modelo_mlp,
    }

    return modelos


def entrenar_modelo(nombre_modelo, modelo, x_entrenamiento, y_entrenamiento):

    tiempo_inicio = time.time()

    modelo.fit(x_entrenamiento, y_entrenamiento)

    tiempo_fin = time.time()
    tiempo_entrenamiento = tiempo_fin - tiempo_inicio

    return modelo, tiempo_entrenamiento


def evaluar_modelo(nombre_modelo, modelo, x_prueba, y_prueba):

    tiempo_inicio = time.time()

    predicciones = modelo.predict(x_prueba)

    tiempo_fin = time.time()
    tiempo_inferencia = tiempo_fin - tiempo_inicio

    exactitud = accuracy_score(y_prueba, predicciones)

    precision = precision_score(
        y_prueba,
        predicciones,
        average="weighted",
        zero_division=0,
    )

    sensibilidad = recall_score(
        y_prueba,
        predicciones,
        average="weighted",
        zero_division=0,
    )

    puntaje_f1 = f1_score(
        y_prueba,
        predicciones,
        average="weighted",
        zero_division=0,
    )

    etiquetas_ordenadas = sorted(GESTOS.keys())
    nombres_gestos = []

    for etiqueta in etiquetas_ordenadas:
        nombres_gestos.append(GESTOS[etiqueta])

    matriz = confusion_matrix(
        y_prueba,
        predicciones,
        labels=etiquetas_ordenadas,
    )

    reporte = classification_report(
        y_prueba,
        predicciones,
        labels=etiquetas_ordenadas,
        target_names=nombres_gestos,
        zero_division=0,
    )

    resultados = {
        "nombre": nombre_modelo,
        "modelo": modelo,
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "puntaje_f1": puntaje_f1,
        "tiempo_inferencia": tiempo_inferencia,
        "matriz_confusion": matriz,
        "reporte": reporte,
        "predicciones": predicciones,
    }

    return resultados


def crear_pipeline_para_validacion(modelo):

    pipeline = Pipeline(
        [
            ("escalador", StandardScaler()),
            ("modelo", modelo),
        ]
    )

    return pipeline


def hacer_validacion_cruzada(nombre_modelo, modelo, datos_x, datos_y):

    pipeline = crear_pipeline_para_validacion(modelo)

    validador = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    puntajes = cross_val_score(
        pipeline,
        datos_x,
        datos_y,
        cv=validador,
        scoring="accuracy",
        n_jobs=-1,
    )

    media = np.mean(puntajes)
    desviacion = np.std(puntajes)

    return media, desviacion


def entrenar_y_evaluar_modelos(modelos, x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, datos_x, datos_y):

    todos_los_resultados = []

    nombres_modelos = list(modelos.keys())

    indice_modelo = 0

    while indice_modelo < len(nombres_modelos):
        nombre_modelo = nombres_modelos[indice_modelo]
        modelo = modelos[nombre_modelo]

        modelo_entrenado, tiempo_entrenamiento = entrenar_modelo(
            nombre_modelo,
            modelo,
            x_entrenamiento,
            y_entrenamiento,
        )

        resultados = evaluar_modelo(
            nombre_modelo,
            modelo_entrenado,
            x_prueba,
            y_prueba,
        )

        resultados["tiempo_entrenamiento"] = tiempo_entrenamiento

        media_cv, desviacion_cv = hacer_validacion_cruzada(
            nombre_modelo,
            modelo,
            datos_x,
            datos_y,
        )

        resultados["media_cv"] = media_cv
        resultados["desviacion_cv"] = desviacion_cv

        todos_los_resultados.append(resultados)

        indice_modelo = indice_modelo + 1

    return todos_los_resultados


def graficar_matrices_confusion(todos_los_resultados):

    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)

    cantidad_modelos = len(todos_los_resultados)

    figura, ejes = plt.subplots(
        1,
        cantidad_modelos,
        figsize=(7 * cantidad_modelos, 6),
    )

    if cantidad_modelos == 1:
        ejes = [ejes]

    nombres_gestos = []

    for etiqueta in sorted(GESTOS.keys()):
        nombres_gestos.append(GESTOS[etiqueta])

    indice_modelo = 0

    while indice_modelo < cantidad_modelos:
        resultado = todos_los_resultados[indice_modelo]
        eje_actual = ejes[indice_modelo]

        sns.heatmap(
            resultado["matriz_confusion"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=nombres_gestos,
            yticklabels=nombres_gestos,
            ax=eje_actual,
        )

        titulo = resultado["nombre"] + "\nAccuracy: " + "{:.2%}".format(resultado["exactitud"])

        eje_actual.set_title(titulo)
        eje_actual.set_xlabel("Predicción")
        eje_actual.set_ylabel("Real")
        eje_actual.tick_params(axis="x", rotation=45)
        eje_actual.tick_params(axis="y", rotation=0)

        indice_modelo = indice_modelo + 1

    plt.tight_layout()

    ruta_figura = os.path.join(DIRECTORIO_FIGURAS, "matrices_confusion.png")
    plt.savefig(ruta_figura, dpi=150, bbox_inches="tight")
    plt.close()


def graficar_comparacion_modelos(todos_los_resultados):

    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)

    figura, ejes = plt.subplots(1, 2, figsize=(16, 6))

    nombres = []
    exactitudes = []
    precisiones = []
    sensibilidades = []
    puntajes_f1 = []
    tiempos_entrenamiento = []
    tiempos_inferencia = []

    indice = 0

    while indice < len(todos_los_resultados):
        resultado = todos_los_resultados[indice]

        nombres.append(resultado["nombre"])
        exactitudes.append(resultado["exactitud"])
        precisiones.append(resultado["precision"])
        sensibilidades.append(resultado["sensibilidad"])
        puntajes_f1.append(resultado["puntaje_f1"])
        tiempos_entrenamiento.append(resultado["tiempo_entrenamiento"])
        tiempos_inferencia.append(resultado["tiempo_inferencia"])

        indice = indice + 1

    posiciones = np.arange(len(nombres))
    ancho_barra = 0.2

    ejes[0].bar(posiciones - 1.5 * ancho_barra, exactitudes, ancho_barra, label="Accuracy")
    ejes[0].bar(posiciones - 0.5 * ancho_barra, precisiones, ancho_barra, label="Precision")
    ejes[0].bar(posiciones + 0.5 * ancho_barra, sensibilidades, ancho_barra, label="Recall")
    ejes[0].bar(posiciones + 1.5 * ancho_barra, puntajes_f1, ancho_barra, label="F1")

    ejes[0].set_title("Comparación de métricas")
    ejes[0].set_ylabel("Puntaje")
    ejes[0].set_xticks(posiciones)
    ejes[0].set_xticklabels(nombres)
    ejes[0].set_ylim(0, 1.1)
    ejes[0].legend()

    ejes[1].bar(posiciones - 0.2, tiempos_entrenamiento, 0.4, label="Entrenamiento")
    ejes[1].bar(posiciones + 0.2, tiempos_inferencia, 0.4, label="Inferencia")

    ejes[1].set_title("Comparación de tiempos")
    ejes[1].set_ylabel("Segundos")
    ejes[1].set_xticks(posiciones)
    ejes[1].set_xticklabels(nombres)
    ejes[1].legend()

    plt.tight_layout()

    ruta_figura = os.path.join(DIRECTORIO_FIGURAS, "comparacion_modelos.png")
    plt.savefig(ruta_figura, dpi=150)
    plt.close()


def graficar_validacion_cruzada(todos_los_resultados):

    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)

    figura, eje = plt.subplots(figsize=(10, 6))

    nombres = []
    medias = []
    desviaciones = []

    indice = 0

    while indice < len(todos_los_resultados):
        resultado = todos_los_resultados[indice]

        nombres.append(resultado["nombre"])
        medias.append(resultado["media_cv"])
        desviaciones.append(resultado["desviacion_cv"])

        indice = indice + 1

    barras = eje.bar(
        nombres,
        medias,
        yerr=desviaciones,
        capsize=10,
        edgecolor="black",
    )

    indice_barra = 0

    while indice_barra < len(barras):
        barra_actual = barras[indice_barra]

        texto = "{:.2%}".format(medias[indice_barra]) + " ± " + "{:.2%}".format(desviaciones[indice_barra])

        eje.text(
            barra_actual.get_x() + barra_actual.get_width() / 2.0,
            barra_actual.get_height() + desviaciones[indice_barra] + 0.01,
            texto,
            ha="center",
            va="bottom",
            fontsize=10,
        )

        indice_barra = indice_barra + 1

    eje.set_title("Validación cruzada")
    eje.set_ylabel("Accuracy promedio")
    eje.set_ylim(0, 1.15)

    plt.tight_layout()

    ruta_figura = os.path.join(DIRECTORIO_FIGURAS, "validacion_cruzada.png")
    plt.savefig(ruta_figura, dpi=150)
    plt.close()


def guardar_tabla_comparativa(todos_los_resultados):

    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)

    filas = []

    indice = 0

    while indice < len(todos_los_resultados):
        resultado = todos_los_resultados[indice]

        fila = {
            "Modelo": resultado["nombre"],
            "Accuracy": round(resultado["exactitud"], 4),
            "Precision": round(resultado["precision"], 4),
            "Recall": round(resultado["sensibilidad"], 4),
            "F1": round(resultado["puntaje_f1"], 4),
            "CV_Media": round(resultado["media_cv"], 4),
            "CV_Desviacion": round(resultado["desviacion_cv"], 4),
            "Tiempo_Entrenamiento": round(resultado["tiempo_entrenamiento"], 4),
            "Tiempo_Inferencia": round(resultado["tiempo_inferencia"], 4),
        }

        filas.append(fila)

        indice = indice + 1

    tabla = pd.DataFrame(filas)

    ruta_tabla = os.path.join(DIRECTORIO_FIGURAS, "comparacion_modelos.csv")
    tabla.to_csv(ruta_tabla, index=False)


def seleccionar_mejor_modelo(todos_los_resultados):

    mejor_resultado = None
    mejor_f1 = -1

    indice = 0

    while indice < len(todos_los_resultados):
        resultado = todos_los_resultados[indice]

        if resultado["puntaje_f1"] > mejor_f1:
            mejor_f1 = resultado["puntaje_f1"]
            mejor_resultado = resultado

        indice = indice + 1

    print("")
    print("=" * 70)
    print("MEJOR MODELO")
    print("=" * 70)
    print("Modelo: " + str(mejor_resultado["nombre"]))
    print("Accuracy:  " + str(round(mejor_resultado["exactitud"], 4)))
    print("Precision: " + str(round(mejor_resultado["precision"], 4)))
    print("Recall:    " + str(round(mejor_resultado["sensibilidad"], 4)))
    print("F1 Score:  " + str(round(mejor_resultado["puntaje_f1"], 4)))
    print("CV Media:  " + str(round(mejor_resultado["media_cv"], 4)))

    return mejor_resultado


def limpiar_txt_modelos():

    os.makedirs(DIRECTORIO_MODELOS, exist_ok=True)

    archivos_modelos = os.listdir(DIRECTORIO_MODELOS)

    indice = 0

    while indice < len(archivos_modelos):
        archivo_actual = archivos_modelos[indice]

        if archivo_actual.endswith(".txt"):
            ruta_archivo = os.path.join(DIRECTORIO_MODELOS, archivo_actual)

            if os.path.exists(ruta_archivo):
                os.remove(ruta_archivo)

        indice = indice + 1


def exportar_modelo(mejor_resultado, escalador, columnas_caracteristicas):

    os.makedirs(DIRECTORIO_MODELOS, exist_ok=True)

    limpiar_txt_modelos()

    ruta_modelo = os.path.join(DIRECTORIO_MODELOS, "mejor_modelo.pkl")
    ruta_escalador = os.path.join(DIRECTORIO_MODELOS, "escalador.pkl")
    ruta_columnas = os.path.join(DIRECTORIO_MODELOS, "columnas_caracteristicas.pkl")

    joblib.dump(mejor_resultado["modelo"], ruta_modelo)
    joblib.dump(escalador, ruta_escalador)
    joblib.dump(columnas_caracteristicas, ruta_columnas)

    return ruta_modelo, ruta_escalador


def main():

    datos_x, datos_y, columnas_caracteristicas = cargar_datos()

    if datos_x is None:
        return

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_datos(
        datos_x,
        datos_y,
    )

    x_entrenamiento_escalado, x_prueba_escalado, escalador = escalar_datos(
        x_entrenamiento,
        x_prueba,
    )

    modelos = crear_modelos()

    todos_los_resultados = entrenar_y_evaluar_modelos(
        modelos,
        x_entrenamiento_escalado,
        x_prueba_escalado,
        y_entrenamiento,
        y_prueba,
        datos_x,
        datos_y,
    )

    graficar_matrices_confusion(todos_los_resultados)
    graficar_comparacion_modelos(todos_los_resultados)
    graficar_validacion_cruzada(todos_los_resultados)
    guardar_tabla_comparativa(todos_los_resultados)

    mejor_resultado = seleccionar_mejor_modelo(todos_los_resultados)

    exportar_modelo(
        mejor_resultado,
        escalador,
        columnas_caracteristicas,
    )


main()