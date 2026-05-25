import cv2
import mediapipe as mp
import numpy as np
import os
import time
import joblib
import pyautogui
import subprocess
import ctypes


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

ACCIONES = {
    0: "Screenshot",
    1: "Bloquear PC",
    2: "Desbloquear PC",
    3: "Subir Volumen",
    4: "Bajar Volumen",
    5: "Iniciar Grabacion",
    6: "Cambiar Ventana",
    7: "Apagar PC",
    8: "Detener Grabacion",
}

DIRECTORIO_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DIRECTORIO_MODELOS = os.path.join(DIRECTORIO_BASE, 'modelos')
RUTA_MODELO = os.path.join(DIRECTORIO_MODELOS, 'mejor_modelo.pkl')
RUTA_ESCALADOR = os.path.join(DIRECTORIO_MODELOS, 'escalador.pkl')

UMBRAL_CONFIANZA = 0.75
TIEMPO_ESPERA_ACCION = 1.5          
CUADROS_CONFIRMACION = 5          
CUADROS_CONFIRMACION_APAGAR = 15  

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

def ejecutar_screenshot():
    pyautogui.hotkey('win', 'shift', 's')


def ejecutar_bloquear_pc():
    ctypes.windll.user32.LockWorkStation()


def ejecutar_desbloquear_pc():
    pyautogui.moveRel(1, 0)
    pyautogui.moveRel(-1, 0)
    pyautogui.press('enter')


def ejecutar_subir_volumen():
    pyautogui.press('volumeup')
    pyautogui.press('volumeup')


def ejecutar_bajar_volumen():
    pyautogui.press('volumedown')
    pyautogui.press('volumedown')


def ejecutar_iniciar_grabacion():
    pyautogui.hotkey('win', 'alt', 'r')


def ejecutar_cambiar_ventana():
    pyautogui.hotkey('alt', 'tab')


def ejecutar_apagar_pc():
    subprocess.run(['shutdown', '/s', '/t', '10', '/c', 'GestureOS: Apagando en 10 segundos. Usa shutdown /a para cancelar.'])


def ejecutar_detener_grabacion():
    pyautogui.hotkey('win', 'alt', 'r')


FUNCIONES_ACCIONES = {
    0: ejecutar_screenshot,
    1: ejecutar_bloquear_pc,
    2: ejecutar_desbloquear_pc,
    3: ejecutar_subir_volumen,
    4: ejecutar_bajar_volumen,
    5: ejecutar_iniciar_grabacion,
    6: ejecutar_cambiar_ventana,
    7: ejecutar_apagar_pc,
    8: ejecutar_detener_grabacion,
}

def crear_estado_gestos():
    estado = {
        "gesto_actual": -1,
        "cuadros_seguidos": 0,
        "ultimo_gesto_ejecutado": -1,
        "tiempo_ultima_accion": 0,
        "accion_ejecutada": False,
        "nombre_ultima_accion": "",
        "tiempo_mensaje": 0,
    }

    return estado


def obtener_cuadros_necesarios(gesto):
    if gesto == 7:
        return CUADROS_CONFIRMACION_APAGAR
    else:
        return CUADROS_CONFIRMACION


def actualizar_estado_gesto(estado, gesto_predicho, confianza, umbral_confianza):

    ejecutar = False

    if confianza < umbral_confianza:
        estado["gesto_actual"] = -1
        estado["cuadros_seguidos"] = 0
        return ejecutar
    
    if gesto_predicho == estado["gesto_actual"]:
        estado["cuadros_seguidos"] = estado["cuadros_seguidos"] + 1
    else:
        estado["gesto_actual"] = gesto_predicho
        estado["cuadros_seguidos"] = 1
        estado["accion_ejecutada"] = False

    cuadros_necesarios = obtener_cuadros_necesarios(gesto_predicho)

    if estado["cuadros_seguidos"] >= cuadros_necesarios:
        if not estado["accion_ejecutada"]:
            tiempo_actual = time.time()
            tiempo_desde_ultima = tiempo_actual - estado["tiempo_ultima_accion"]
            if tiempo_desde_ultima >= TIEMPO_ESPERA_ACCION:
                ejecutar = True
                estado["ultimo_gesto_ejecutado"] = gesto_predicho
                estado["tiempo_ultima_accion"] = tiempo_actual
                estado["accion_ejecutada"] = True

    return ejecutar

def predecir_gesto(modelo, escalador, puntos_mano):
    puntos_normalizados = normalizar_puntos_mano(puntos_mano)

    if puntos_normalizados is None:
        return -1, 0.0

    estados_dedos = calcular_estado_dedos(puntos_mano)
    vector_caracteristicas = []
    indice_valor = 0

    while indice_valor < len(puntos_normalizados):
        vector_caracteristicas.append(puntos_normalizados[indice_valor])
        indice_valor = indice_valor + 1

    indice_dedo = 0

    while indice_dedo < len(estados_dedos):
        vector_caracteristicas.append(estados_dedos[indice_dedo])
        indice_dedo = indice_dedo + 1

    vector_caracteristicas = np.array(vector_caracteristicas).reshape(1, -1)
    vector_escalado = escalador.transform(vector_caracteristicas)
    gesto_predicho = modelo.predict(vector_escalado)[0]

    if hasattr(modelo, 'predict_proba'):
        probabilidades = modelo.predict_proba(vector_escalado)[0]
        confianza = np.max(probabilidades)
    else:
        confianza = 1.0

    return int(gesto_predicho), float(confianza)


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


def dibujar_interfaz(cuadro, gesto_predicho, confianza, estado, umbral_confianza, cuadros_por_segundo):
    dimensiones = cuadro.shape
    alto_cuadro = dimensiones[0]
    ancho_cuadro = dimensiones[1]
    capa_superpuesta = cuadro.copy()

    cv2.rectangle(
        capa_superpuesta,
        (0, 0),
        (360, alto_cuadro),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(capa_superpuesta, 0.6, cuadro, 0.4, 0, cuadro)
    cuadro = dibujar_texto(cuadro, "GestureOS", 10, 30, 0.7, (0, 255, 200), 2)
    cuadro = dibujar_texto(cuadro, "MODO INFERENCIA", 10, 55, 0.45, (100, 255, 100), 1)
    desplazamiento_y = 200

    desplazamiento_y = desplazamiento_y + 30

    tiempo_actual = time.time()
    tiempo_desde_mensaje = tiempo_actual - estado["tiempo_mensaje"]

    if tiempo_desde_mensaje < 2.0 and estado["nombre_ultima_accion"] != "":
        cuadro = dibujar_texto(
            cuadro,
            "EJECUTADO:",
            10,
            desplazamiento_y,
            0.5,
            (0, 255, 0),
            2,
        )

        desplazamiento_y = desplazamiento_y + 25

        cuadro = dibujar_texto(
            cuadro,
            estado["nombre_ultima_accion"],
            10,
            desplazamiento_y,
            0.45,
            (0, 255, 0),
            1,
        )

        desplazamiento_y = desplazamiento_y + 30

    desplazamiento_y = desplazamiento_y + 10

    cuadro = dibujar_texto(cuadro, "Gestos disponibles:", 10, desplazamiento_y, 0.4, (200, 200, 200), 1)

    desplazamiento_y = desplazamiento_y + 22

    for indice_gesto in sorted(GESTOS.keys()):
        nombre = GESTOS[indice_gesto]
        accion = ACCIONES[indice_gesto]

        if gesto_predicho == indice_gesto and confianza >= umbral_confianza:
            color_texto = (0, 255, 200)
        else:
            color_texto = (150, 150, 150)

        texto_gesto = "[" + str(indice_gesto) + "] " + nombre + " -> " + accion

        cuadro = dibujar_texto(cuadro, texto_gesto, 10, desplazamiento_y, 0.32, color_texto, 1)

        desplazamiento_y = desplazamiento_y + 18

    desplazamiento_y = desplazamiento_y + 15

    cuadro = dibujar_texto(
        cuadro,
        "Umbral: " + "{:.0%}".format(umbral_confianza) + " (+/- para ajustar)",
        10,
        desplazamiento_y,
        0.35,
        (200, 200, 200),
        1,
    )

    desplazamiento_y = desplazamiento_y + 18

    cuadro = dibujar_texto(cuadro, "q: Salir", 10, desplazamiento_y, 0.35, (200, 200, 200), 1)

    cuadros_redondeados = round(cuadros_por_segundo)
    texto_cuadros = "FPS: " + str(cuadros_redondeados)

    cuadro = dibujar_texto(cuadro, texto_cuadros, ancho_cuadro - 100, 30, 0.5, (0, 255, 200), 1)

    return cuadro


def cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        print("No se encontró el modelo: " + RUTA_MODELO)
        print("Ejecuta primero el script de entrenamiento.")
        return None, None

    if not os.path.exists(RUTA_ESCALADOR):
        print("No se encontró el escalador: " + RUTA_ESCALADOR)
        print("Ejecuta primero el script de entrenamiento.")
        return None, None

    modelo = joblib.load(RUTA_MODELO)
    escalador = joblib.load(RUTA_ESCALADOR)

    print("Modelo cargado: " + RUTA_MODELO)
    print("Escalador cargado: " + RUTA_ESCALADOR)

    return modelo, escalador


def abrir_camara():
    indice_camara = 0
    captura_video = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)

    if not captura_video.isOpened():
        print("Error al abrir la cámara.")
        return None

    captura_video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return captura_video


def main():

    print("GestureOS - Inferencia en Tiempo Real")

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    modelo, escalador = cargar_modelo()

    if modelo is None:
        return

    captura_video = abrir_camara()

    if captura_video is None:
        return

    estado = crear_estado_gestos()
    umbral_confianza = UMBRAL_CONFIANZA
    tiempo_anterior = time.time()
    cuadros_por_segundo = 0

    print("Controles: +/- ajustar umbral | q salir")
    print("")
    print("ACCIONES CONFIGURADAS:")

    for indice_gesto in sorted(GESTOS.keys()):
        nombre = GESTOS[indice_gesto]
        accion = ACCIONES[indice_gesto]
        print("  [" + str(indice_gesto) + "] " + nombre + " -> " + accion)

    print("")
    print("ADVERTENCIA :/ 'Apagar PC' requiere mantener el gesto 15 cuadros.")
    print("      Si se activa, tienes 10 seg para cancelar con: shutdown /a")

    with mp_manos.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as detector_manos:

        programa_activo = True

        while programa_activo:
            if not captura_video.isOpened():
                programa_activo = False
                continue

            lectura_correcta, cuadro = captura_video.read()

            if not lectura_correcta:
                print("Error leyendo la cámara.")
                programa_activo = False
                continue

            cuadro = cv2.flip(cuadro, 1)
            tiempo_actual = time.time()
            diferencia_tiempo = tiempo_actual - tiempo_anterior + 0.000001
            cuadros_por_segundo = 1 / diferencia_tiempo
            tiempo_anterior = tiempo_actual
            cuadro_rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
            cuadro_rgb.flags.writeable = False
            resultados = detector_manos.process(cuadro_rgb)
            cuadro_rgb.flags.writeable = True

            gesto_predicho = -1
            confianza = 0.0

            if resultados.multi_hand_landmarks:
                for puntos_mano in resultados.multi_hand_landmarks:
                    mp_dibujo.draw_landmarks(
                        cuadro,
                        puntos_mano,
                        mp_manos.HAND_CONNECTIONS,
                        mp_estilos_dibujo.get_default_hand_landmarks_style(),
                        mp_estilos_dibujo.get_default_hand_connections_style(),
                    )

                    gesto_predicho, confianza = predecir_gesto(
                        modelo,
                        escalador,
                        puntos_mano,
                    )

                    debe_ejecutar = actualizar_estado_gesto(
                        estado,
                        gesto_predicho,
                        confianza,
                        umbral_confianza,
                    )

                    if debe_ejecutar:
                        if gesto_predicho in FUNCIONES_ACCIONES:
                            nombre_accion = ACCIONES.get(gesto_predicho, "")

                            print(
                                ">> Ejecutando: ["
                                + str(gesto_predicho)
                                + "] "
                                + GESTOS.get(gesto_predicho, "")
                                + " -> "
                                + nombre_accion
                            )

                            try:
                                funcion_accion = FUNCIONES_ACCIONES[gesto_predicho]
                                funcion_accion()

                                estado["nombre_ultima_accion"] = nombre_accion
                                estado["tiempo_mensaje"] = time.time()

                            except Exception as error:
                                print("Error ejecutando acción: " + str(error))

            else:
                estado["gesto_actual"] = -1
                estado["cuadros_seguidos"] = 0

            cuadro = dibujar_interfaz(
                cuadro,
                gesto_predicho,
                confianza,
                estado,
                umbral_confianza,
                cuadros_por_segundo,
            )

            cv2.imshow("GestureOS - Inferencia", cuadro)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord('q'):
                programa_activo = False

            elif tecla == ord('+') or tecla == ord('='):
                umbral_confianza = umbral_confianza + 0.05

                if umbral_confianza > 0.99:
                    umbral_confianza = 0.99

                print("Umbral de confianza: " + "{:.0%}".format(umbral_confianza))

            elif tecla == ord('-'):
                umbral_confianza = umbral_confianza - 0.05

                if umbral_confianza < 0.30:
                    umbral_confianza = 0.30

                print("Umbral de confianza: " + "{:.0%}".format(umbral_confianza))

    captura_video.release()
    cv2.destroyAllWindows()

    print("\nGestureOS finalizado.")


main()