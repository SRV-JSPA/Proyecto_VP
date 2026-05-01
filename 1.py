"""
===========================================================================
GestureOS - Script de Recolección de Datos
===========================================================================
Este programa captura landmarks de una mano utilizando MediaPipe y OpenCV.
Los datos capturados se guardan en un archivo CSV para entrenar un modelo de
clasificación de gestos.

Controles:
0-8  Seleccionar gesto
r    Iniciar o detener grabación
q    Salir y guardar datos
===========================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
import csv


GESTOS = {
    0: "palma_abierta",
    1: "puno_cerrado",
    2: "indice_arriba",
    3: "pulgar_arriba",
    4: "pulgar_abajo",
    5: "pinch",
    6: "paz",
    7: "tres_dedos",
    8: "ok_sign",
}


mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilos_dibujo = mp.solutions.drawing_styles


def normalizar_puntos_mano(puntos_mano):
    lista_puntos = []

    indice_punto = 0

    while indice_punto < len(puntos_mano.landmark):
        punto_actual_mediapipe = puntos_mano.landmark[indice_punto]

        coordenada_x = punto_actual_mediapipe.x
        coordenada_y = punto_actual_mediapipe.y
        coordenada_z = punto_actual_mediapipe.z

        punto_actual = []
        punto_actual.append(coordenada_x)
        punto_actual.append(coordenada_y)
        punto_actual.append(coordenada_z)

        lista_puntos.append(punto_actual)

        indice_punto = indice_punto + 1

    arreglo_puntos = np.array(lista_puntos)

    punto_muneca = arreglo_puntos[0]

    puntos_centrados = []

    indice_fila = 0

    while indice_fila < len(arreglo_puntos):
        punto_actual = arreglo_puntos[indice_fila]

        punto_centrado = punto_actual - punto_muneca

        puntos_centrados.append(punto_centrado)

        indice_fila = indice_fila + 1

    puntos_centrados = np.array(puntos_centrados)

    distancias = []

    indice_distancia = 0

    while indice_distancia < len(puntos_centrados):
        punto_actual = puntos_centrados[indice_distancia]

        valor_x = punto_actual[0]
        valor_y = punto_actual[1]
        valor_z = punto_actual[2]

        distancia = np.sqrt((valor_x * valor_x) + (valor_y * valor_y) + (valor_z * valor_z))

        distancias.append(distancia)

        indice_distancia = indice_distancia + 1

    distancia_maxima = np.max(distancias)

    if distancia_maxima == 0:
        return None

    puntos_normalizados = []

    indice_normalizacion = 0

    while indice_normalizacion < len(puntos_centrados):
        punto_actual = puntos_centrados[indice_normalizacion]

        punto_normalizado = punto_actual / distancia_maxima

        puntos_normalizados.append(punto_normalizado)

        indice_normalizacion = indice_normalizacion + 1

    puntos_normalizados = np.array(puntos_normalizados)

    vector_final = []

    indice_punto_final = 0

    while indice_punto_final < len(puntos_normalizados):
        punto_actual = puntos_normalizados[indice_punto_final]

        vector_final.append(punto_actual[0])
        vector_final.append(punto_actual[1])
        vector_final.append(punto_actual[2])

        indice_punto_final = indice_punto_final + 1

    vector_final = np.array(vector_final)

    return vector_final


def calcular_estado_dedos(puntos_mano):
    puntos = puntos_mano.landmark

    puntas_dedos = [4, 8, 12, 16, 20]
    articulaciones_dedos = [3, 6, 10, 14, 18]

    estados_dedos = []

    indice_punta_pulgar = puntas_dedos[0]
    indice_articulacion_pulgar = articulaciones_dedos[0]

    punto_punta_pulgar = puntos[indice_punta_pulgar]
    punto_articulacion_pulgar = puntos[indice_articulacion_pulgar]

    coordenada_x_punta_pulgar = punto_punta_pulgar.x
    coordenada_x_articulacion_pulgar = punto_articulacion_pulgar.x

    if coordenada_x_punta_pulgar < coordenada_x_articulacion_pulgar:
        estado_pulgar = 1
    else:
        estado_pulgar = 0

    estados_dedos.append(estado_pulgar)

    indice_dedo = 1

    while indice_dedo < 5:
        indice_punta = puntas_dedos[indice_dedo]
        indice_articulacion = articulaciones_dedos[indice_dedo]

        punto_punta = puntos[indice_punta]
        punto_articulacion = puntos[indice_articulacion]

        coordenada_y_punta = punto_punta.y
        coordenada_y_articulacion = punto_articulacion.y

        if coordenada_y_punta < coordenada_y_articulacion:
            estado_dedo = 1
        else:
            estado_dedo = 0

        estados_dedos.append(estado_dedo)

        indice_dedo = indice_dedo + 1

    return estados_dedos


def crear_nombres_columnas():
    columnas = []

    indice_punto = 0

    while indice_punto < 21:
        nombre_columna_x = "punto" + str(indice_punto) + "_x"
        nombre_columna_y = "punto" + str(indice_punto) + "_y"
        nombre_columna_z = "punto" + str(indice_punto) + "_z"

        columnas.append(nombre_columna_x)
        columnas.append(nombre_columna_y)
        columnas.append(nombre_columna_z)

        indice_punto = indice_punto + 1

    columnas.append("dedo_pulgar")
    columnas.append("dedo_indice")
    columnas.append("dedo_medio")
    columnas.append("dedo_anular")
    columnas.append("dedo_menique")

    columnas.append("etiqueta")

    return columnas


def dibujar_texto(cuadro, texto, posicion_x, posicion_y, tamano, color, grosor):
    cv2.putText(
        cuadro,
        texto,
        (posicion_x, posicion_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        tamano,
        color,
        grosor,
    )

    return cuadro


def dibujar_interfaz(cuadro, gesto_actual, esta_grabando, conteo_muestras, cuadros_por_segundo):
    dimensiones = cuadro.shape

    alto_cuadro = dimensiones[0]
    ancho_cuadro = dimensiones[1]

    capa_superpuesta = cuadro.copy()

    punto_inicio_panel = (0, 0)
    punto_fin_panel = (320, alto_cuadro)
    color_panel = (0, 0, 0)
    grosor_relleno = -1

    cv2.rectangle(
        capa_superpuesta,
        punto_inicio_panel,
        punto_fin_panel,
        color_panel,
        grosor_relleno,
    )

    transparencia_capa = 0.6
    transparencia_cuadro = 0.4
    valor_gamma = 0

    cv2.addWeighted(
        capa_superpuesta,
        transparencia_capa,
        cuadro,
        transparencia_cuadro,
        valor_gamma,
        cuadro,
    )

    cuadro = dibujar_texto(
        cuadro,
        "GestureOS - Data Collector :)",
        10,
        30,
        0.7,
        (0, 255, 200),
        2,
    )

    if gesto_actual in GESTOS:
        nombre_gesto = GESTOS[gesto_actual]
    else:
        nombre_gesto = "Ninguno"

    texto_gesto = "Gesto: [" + str(gesto_actual) + "] " + nombre_gesto

    cuadro = dibujar_texto(
        cuadro,
        texto_gesto,
        10,
        65,
        0.5,
        (255, 255, 255),
        1,
    )

    if esta_grabando:
        tiempo_actual = time.time()
        tiempo_multiplicado = tiempo_actual * 2
        tiempo_entero = int(tiempo_multiplicado)
        residuo_tiempo = tiempo_entero % 2

        if residuo_tiempo == 0:
            color_grabacion = (0, 0, 255)
        else:
            color_grabacion = (0, 0, 180)

        cv2.circle(
            cuadro,
            (20, 90),
            8,
            color_grabacion,
            -1,
        )

        cuadro = dibujar_texto(
            cuadro,
            "GRABANDO",
            35,
            95,
            0.5,
            (0, 0, 255),
            2,
        )
    else:
        cuadro = dibujar_texto(
            cuadro,
            "EN ESPERA",
            10,
            95,
            0.5,
            (100, 255, 100),
            1,
        )

    cuadro = dibujar_texto(
        cuadro,
        "--- Muestras ---",
        10,
        130,
        0.45,
        (200, 200, 200),
        1,
    )

    desplazamiento_y = 155

    for indice_gesto, nombre_gesto in GESTOS.items():
        if indice_gesto in conteo_muestras:
            cantidad_muestras = conteo_muestras[indice_gesto]
        else:
            cantidad_muestras = 0

        if cantidad_muestras >= 500:
            color_texto = (0, 255, 0)
        else:
            color_texto = (0, 255, 255)

        texto_muestra = "[" + str(indice_gesto) + "] " + nombre_gesto + ": " + str(cantidad_muestras)

        cuadro = dibujar_texto(
            cuadro,
            texto_muestra,
            10,
            desplazamiento_y,
            0.4,
            color_texto,
            1,
        )

        desplazamiento_y = desplazamiento_y + 22

    desplazamiento_y = desplazamiento_y + 15

    cuadro = dibujar_texto(
        cuadro,
        "--- Controles ---",
        10,
        desplazamiento_y,
        0.45,
        (200, 200, 200),
        1,
    )

    desplazamiento_y = desplazamiento_y + 25

    texto_control_1 = "0-8: Seleccionar gesto"
    texto_control_2 = "r: Iniciar/parar grabacion"
    texto_control_3 = "q: Salir y guardar"

    cuadro = dibujar_texto(
        cuadro,
        texto_control_1,
        10,
        desplazamiento_y,
        0.4,
        (200, 200, 200),
        1,
    )

    desplazamiento_y = desplazamiento_y + 20

    cuadro = dibujar_texto(
        cuadro,
        texto_control_2,
        10,
        desplazamiento_y,
        0.4,
        (200, 200, 200),
        1,
    )

    desplazamiento_y = desplazamiento_y + 20

    cuadro = dibujar_texto(
        cuadro,
        texto_control_3,
        10,
        desplazamiento_y,
        0.4,
        (200, 200, 200),
        1,
    )

    cuadros_redondeados = round(cuadros_por_segundo)
    texto_cuadros = "FPS: " + str(cuadros_redondeados)

    posicion_x_cuadros = ancho_cuadro - 100
    posicion_y_cuadros = 30

    cuadro = dibujar_texto(
        cuadro,
        texto_cuadros,
        posicion_x_cuadros,
        posicion_y_cuadros,
        0.5,
        (0, 255, 200),
        1,
    )

    return cuadro


def crear_conteo_muestras():
    conteo_muestras = {}

    for indice_gesto in GESTOS:
        conteo_muestras[indice_gesto] = 0

    return conteo_muestras


def crear_ruta_csv():
    carpeta_datos = "data"

    if os.path.exists(carpeta_datos):
        carpeta_existe = True
    else:
        carpeta_existe = False

    if not carpeta_existe:
        os.makedirs(carpeta_datos)

    fecha_actual = datetime.now()
    marca_tiempo = fecha_actual.strftime("%Y%m%d_%H%M%S")

    nombre_archivo = "datos_gestos_" + marca_tiempo + ".csv"
    ruta_csv = os.path.join(carpeta_datos, nombre_archivo)

    return ruta_csv


def abrir_camara():
    indice_camara = 0
    captura_video = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)

    if captura_video.isOpened():
        camara_abierta = True
    else:
        camara_abierta = False

    if not camara_abierta:
        print("ERROR: No se pudo abrir la camara.")
        print("Verifica que tu webcam este conectada y no este en uso.")
        return None

    ancho_deseado = 640
    alto_deseado = 480

    captura_video.set(cv2.CAP_PROP_FRAME_WIDTH, ancho_deseado)
    captura_video.set(cv2.CAP_PROP_FRAME_HEIGHT, alto_deseado)

    return captura_video


def imprimir_informacion_inicial(ruta_csv):
    print("=" * 60)
    print("GestureOS - Recolector de Datos")
    print("=" * 60)
    print("Archivo de salida: " + ruta_csv)
    print("Gestos configurados: " + str(len(GESTOS)))
    print("Presiona 'q' para salir y guardar.")
    print("=" * 60)


def procesar_puntos_mano(cuadro, resultados, esta_grabando, gesto_actual, datos_recolectados, conteo_muestras):
    mano_detectada = False

    if resultados.multi_hand_landmarks:
        for puntos_mano in resultados.multi_hand_landmarks:
            mano_detectada = True

            estilo_puntos = mp_estilos_dibujo.get_default_hand_landmarks_style()
            estilo_conexiones = mp_estilos_dibujo.get_default_hand_connections_style()

            mp_dibujo.draw_landmarks(
                cuadro,
                puntos_mano,
                mp_manos.HAND_CONNECTIONS,
                estilo_puntos,
                estilo_conexiones,
            )

            if esta_grabando:
                puntos_normalizados = normalizar_puntos_mano(puntos_mano)

                if puntos_normalizados is not None:
                    estados_dedos = calcular_estado_dedos(puntos_mano)

                    fila_datos = []

                    indice_valor = 0

                    while indice_valor < len(puntos_normalizados):
                        valor_actual = puntos_normalizados[indice_valor]
                        fila_datos.append(valor_actual)
                        indice_valor = indice_valor + 1

                    indice_dedo = 0

                    while indice_dedo < len(estados_dedos):
                        estado_actual = estados_dedos[indice_dedo]
                        fila_datos.append(estado_actual)
                        indice_dedo = indice_dedo + 1

                    fila_datos.append(gesto_actual)

                    datos_recolectados.append(fila_datos)

                    cantidad_actual = conteo_muestras[gesto_actual]
                    nueva_cantidad = cantidad_actual + 1
                    conteo_muestras[gesto_actual] = nueva_cantidad

    return mano_detectada


def mostrar_mensaje_sin_mano(cuadro):
    cv2.putText(
        cuadro,
        "NO SE DETECTA MANO",
        (200, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return cuadro


def leer_accion_teclado(tecla, gesto_actual, esta_grabando):
    salir_programa = False

    if ord("0") <= tecla <= ord("8"):
        nuevo_gesto = tecla - ord("0")

        if nuevo_gesto in GESTOS:
            gesto_actual = nuevo_gesto

            print(
                "Gesto seleccionado: ["
                + str(gesto_actual)
                + "] "
                + GESTOS[gesto_actual]
            )

    elif tecla == ord("r"):
        if esta_grabando:
            esta_grabando = False
        else:
            esta_grabando = True

        if esta_grabando:
            estado = "GRABANDO"
        else:
            estado = "DETENIDO"

        print(
            estado
            + " - Gesto: ["
            + str(gesto_actual)
            + "] "
            + GESTOS[gesto_actual]
        )

    elif tecla == ord("q"):
        salir_programa = True

    resultado = {
        "gesto_actual": gesto_actual,
        "esta_grabando": esta_grabando,
        "salir_programa": salir_programa,
    }

    return resultado


def guardar_datos_csv(ruta_csv, datos_recolectados):
    columnas = crear_nombres_columnas()

    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        escritor_csv.writerow(columnas)

        indice_fila = 0

        while indice_fila < len(datos_recolectados):
            fila_actual = datos_recolectados[indice_fila]
            escritor_csv.writerow(fila_actual)
            indice_fila = indice_fila + 1


def imprimir_resumen_final(ruta_csv, datos_recolectados, conteo_muestras):
    print("")
    print("=" * 60)
    print("DATOS GUARDADOS EXITOSAMENTE")
    print("=" * 60)
    print("Archivo: " + ruta_csv)
    print("Total de muestras: " + str(len(datos_recolectados)))
    print("")
    print("Muestras por gesto:")

    for indice_gesto, nombre_gesto in GESTOS.items():
        cantidad = conteo_muestras[indice_gesto]

        if cantidad >= 500:
            estado = "OK"
        else:
            estado = "NECESITA MAS"

        print(
            "  ["
            + str(indice_gesto)
            + "] "
            + nombre_gesto
            + ": "
            + str(cantidad)
            + " ("
            + estado
            + ")"
        )

    print("=" * 60)


def main():
    ruta_csv = crear_ruta_csv()

    gesto_actual = 0
    esta_grabando = False
    datos_recolectados = []
    conteo_muestras = crear_conteo_muestras()

    tiempo_anterior = time.time()
    cuadros_por_segundo = 0

    captura_video = abrir_camara()

    if captura_video is None:
        return

    imprimir_informacion_inicial(ruta_csv)

    with mp_manos.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as detector_manos:

        programa_activo = True

        while programa_activo:
            if captura_video.isOpened():
                camara_sigue_abierta = True
            else:
                camara_sigue_abierta = False

            if not camara_sigue_abierta:
                programa_activo = False
            else:
                lectura_correcta, cuadro = captura_video.read()

                if not lectura_correcta:
                    print("ERROR: No se pudo leer frame de la camara.")
                    programa_activo = False
                else:
                    cuadro = cv2.flip(cuadro, 1)

                    tiempo_actual = time.time()
                    diferencia_tiempo = tiempo_actual - tiempo_anterior
                    diferencia_tiempo = diferencia_tiempo + 0.000001
                    cuadros_por_segundo = 1 / diferencia_tiempo
                    tiempo_anterior = tiempo_actual

                    cuadro_rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)

                    cuadro_rgb.flags.writeable = False
                    resultados = detector_manos.process(cuadro_rgb)
                    cuadro_rgb.flags.writeable = True

                    mano_detectada = procesar_puntos_mano(
                        cuadro,
                        resultados,
                        esta_grabando,
                        gesto_actual,
                        datos_recolectados,
                        conteo_muestras,
                    )

                    if not mano_detectada:
                        if esta_grabando:
                            cuadro = mostrar_mensaje_sin_mano(cuadro)

                    cuadro = dibujar_interfaz(
                        cuadro,
                        gesto_actual,
                        esta_grabando,
                        conteo_muestras,
                        cuadros_por_segundo,
                    )

                    cv2.imshow("GestureOS - Data Collector :)", cuadro)

                    tecla = cv2.waitKey(1) & 0xFF

                    accion = leer_accion_teclado(
                        tecla,
                        gesto_actual,
                        esta_grabando,
                    )

                    gesto_actual = accion["gesto_actual"]
                    esta_grabando = accion["esta_grabando"]

                    if accion["salir_programa"]:
                        programa_activo = False

    captura_video.release()
    cv2.destroyAllWindows()

    if len(datos_recolectados) > 0:
        guardar_datos_csv(ruta_csv, datos_recolectados)
        imprimir_resumen_final(ruta_csv, datos_recolectados, conteo_muestras)
    else:
        print("")
        print("No se recolectaron datos.")


main()