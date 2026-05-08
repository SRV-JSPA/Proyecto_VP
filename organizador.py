import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

DIRECTORIO_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DIRECTORIO_DATOS = os.path.join(DIRECTORIO_BASE, 'data')
ARCHIVO_COMBINADO = os.path.join(DIRECTORIO_DATOS, 'gestos_total.csv')
ARCHIVO_LIMPIO = os.path.join(DIRECTORIO_DATOS, 'gestos_limpio.csv')
DIRECTORIO_FIGURAS = os.path.join(DIRECTORIO_BASE, 'figuras')


def combinar_archivos_csv():
    patron_busqueda = os.path.join(DIRECTORIO_DATOS, 'datos_gestos_*.csv')
    archivos_csv = sorted(glob.glob(patron_busqueda))

    archivos_csv = [archivo for archivo in archivos_csv if 'total' not in archivo and 'limpio' not in archivo]

    if not archivos_csv:
        return None


    lista_dataframes = []

    indice_archivo = 0

    while indice_archivo < len(archivos_csv):
        archivo_actual = archivos_csv[indice_archivo]
        dataframe_actual = pd.read_csv(archivo_actual)

        lista_dataframes.append(dataframe_actual)

        indice_archivo = indice_archivo + 1

    datos_combinados = pd.concat(lista_dataframes, ignore_index=True)

    cantidad_antes = len(datos_combinados)
    datos_combinados = datos_combinados.drop_duplicates()
    cantidad_despues = len(datos_combinados)

    if cantidad_antes != cantidad_despues:
        duplicados_eliminados = cantidad_antes - cantidad_despues
        print("\nSe eliminaron " + str(duplicados_eliminados) + " duplicados.")

    datos_combinados.to_csv(ARCHIVO_COMBINADO, index=False)
    print("\nArchivo combinado guardado: " + ARCHIVO_COMBINADO)
    print("Total de muestras: " + str(len(datos_combinados)))

    return datos_combinados


def eliminar_filas_nulas(dataframe):
    cantidad_antes = len(dataframe)
    dataframe_limpio = dataframe.dropna()
    cantidad_despues = len(dataframe_limpio)

    filas_eliminadas = cantidad_antes - cantidad_despues

    if filas_eliminadas > 0:
        print("  Filas con valores nulos eliminadas: " + str(filas_eliminadas))
    else:
        print("  No se encontraron valores nulos.")

    return dataframe_limpio


def eliminar_filas_infinitas(dataframe):
    columnas_numericas = dataframe.select_dtypes(include=[np.number]).columns
    cantidad_antes = len(dataframe)

    mascara_finitos = np.isfinite(dataframe[columnas_numericas]).all(axis=1)
    dataframe_limpio = dataframe[mascara_finitos].copy()

    cantidad_despues = len(dataframe_limpio)
    filas_eliminadas = cantidad_antes - cantidad_despues

    if filas_eliminadas > 0:
        print("  Filas con valores infinitos eliminadas: " + str(filas_eliminadas))
    else:
        print("  No se encontraron valores infinitos.")

    return dataframe_limpio


def eliminar_etiquetas_invalidas(dataframe):
    etiquetas_validas = list(GESTOS.keys())
    cantidad_antes = len(dataframe)

    mascara_validas = dataframe['etiqueta'].isin(etiquetas_validas)
    dataframe_limpio = dataframe[mascara_validas].copy()

    cantidad_despues = len(dataframe_limpio)
    filas_eliminadas = cantidad_antes - cantidad_despues

    if filas_eliminadas > 0:
        print("  Filas con etiquetas inválidas eliminadas: " + str(filas_eliminadas))
    else:
        print("  No se encontraron etiquetas inválidas.")

    return dataframe_limpio


def corregir_valores_dedos(dataframe):
    columnas_dedos = [columna for columna in dataframe.columns if columna.startswith('dedo_')]

    filas_corregidas = 0

    indice_columna = 0

    while indice_columna < len(columnas_dedos):
        columna_actual = columnas_dedos[indice_columna]

        valores_incorrectos = ~dataframe[columna_actual].isin([0, 1])
        cantidad_incorrectos = valores_incorrectos.sum()

        if cantidad_incorrectos > 0:
            filas_corregidas = filas_corregidas + cantidad_incorrectos

            dataframe.loc[valores_incorrectos, columna_actual] = (
                dataframe.loc[valores_incorrectos, columna_actual]
                .apply(lambda valor: 1 if valor >= 0.5 else 0)
            )

        indice_columna = indice_columna + 1

    if filas_corregidas > 0:
        print(" Valores de dedos corregidos: " + str(filas_corregidas))
    else:
        print(" Valores de dedos ya son binarios (0/1).")

    return dataframe


def eliminar_outliers_por_gesto(dataframe, umbral_z=3.5):
    columnas_landmarks = [columna for columna in dataframe.columns if columna.startswith('punto')]

    lista_limpios = []
    total_outliers = 0

    etiquetas_unicas = sorted(dataframe['etiqueta'].unique())

    indice_etiqueta = 0

    while indice_etiqueta < len(etiquetas_unicas):
        etiqueta_actual = etiquetas_unicas[indice_etiqueta]
        subconjunto = dataframe[dataframe['etiqueta'] == etiqueta_actual].copy()

        cantidad_antes = len(subconjunto)

        datos_landmarks = subconjunto[columnas_landmarks]
        medianas = datos_landmarks.median()
        desviaciones = (datos_landmarks - medianas).abs()
        mad = desviaciones.median()

        mad_ajustado = mad.replace(0, 1e-6)

        z_scores_modificados = 0.6745 * (datos_landmarks - medianas) / mad_ajustado

        mascara_outliers = (z_scores_modificados.abs() > umbral_z).any(axis=1)

        subconjunto_limpio = subconjunto[~mascara_outliers]
        cantidad_despues = len(subconjunto_limpio)

        outliers_gesto = cantidad_antes - cantidad_despues
        total_outliers = total_outliers + outliers_gesto

        nombre_gesto = GESTOS.get(etiqueta_actual, str(etiqueta_actual))

        if outliers_gesto > 0:
            print("    [" + str(etiqueta_actual) + "] " + nombre_gesto + ": " + str(outliers_gesto) + " outliers")

        lista_limpios.append(subconjunto_limpio)

        indice_etiqueta = indice_etiqueta + 1

    dataframe_limpio = pd.concat(lista_limpios, ignore_index=True)

    print(" Total de outliers eliminados: " + str(total_outliers))

    return dataframe_limpio


def balancear_clases(dataframe, metodo="submuestreo"):
    conteo_clases = dataframe['etiqueta'].value_counts()

    if metodo == "submuestreo":
        objetivo = conteo_clases.min()
        print("  Balanceo por submuestreo. Objetivo: " + str(objetivo) + " muestras por gesto.")
    else:
        objetivo = conteo_clases.max()
        print("  Balanceo por sobremuestreo. Objetivo: " + str(objetivo) + " muestras por gesto.")

    lista_balanceados = []

    etiquetas_unicas = sorted(dataframe['etiqueta'].unique())

    indice_etiqueta = 0

    while indice_etiqueta < len(etiquetas_unicas):
        etiqueta_actual = etiquetas_unicas[indice_etiqueta]
        subconjunto = dataframe[dataframe['etiqueta'] == etiqueta_actual]

        cantidad_actual = len(subconjunto)

        if cantidad_actual > objetivo:
            subconjunto_ajustado = subconjunto.sample(n=objetivo, random_state=42)
        elif cantidad_actual < objetivo:
            muestras_faltantes = objetivo - cantidad_actual
            muestras_extra = subconjunto.sample(n=muestras_faltantes, replace=True, random_state=42)
            subconjunto_ajustado = pd.concat([subconjunto, muestras_extra], ignore_index=True)
        else:
            subconjunto_ajustado = subconjunto

        lista_balanceados.append(subconjunto_ajustado)

        indice_etiqueta = indice_etiqueta + 1

    dataframe_balanceado = pd.concat(lista_balanceados, ignore_index=True)

    dataframe_balanceado = dataframe_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

    print("  Total de muestras después del balanceo: " + str(len(dataframe_balanceado)))

    return dataframe_balanceado


def limpiar_datos(dataframe):
    print("Limpieza de dataset")

    cantidad_original = len(dataframe)
    print("\nMuestras originales: " + str(cantidad_original))

    dataframe = eliminar_filas_nulas(dataframe)

    dataframe = eliminar_filas_infinitas(dataframe)
    
    dataframe = eliminar_etiquetas_invalidas(dataframe)

    dataframe = corregir_valores_dedos(dataframe)

    dataframe = eliminar_outliers_por_gesto(dataframe, umbral_z=3.5)

    conteo_clases = dataframe['etiqueta'].value_counts()
    ratio_desbalance = conteo_clases.max() / conteo_clases.min()

    if ratio_desbalance > 2.0:
        print("  Ratio de desbalance: " + "{:.2f}".format(ratio_desbalance))
        dataframe = balancear_clases(dataframe, metodo="submuestreo")
    else:
        print("  Ratio de desbalance: " + "{:.2f}".format(ratio_desbalance))

    # Resumen final
    cantidad_final = len(dataframe)
    muestras_removidas = cantidad_original - cantidad_final

    print("Resumen")
    print("  Muestras originales:  " + str(cantidad_original))
    print("  Muestras finales:     " + str(cantidad_final))
    print("  Muestras removidas:   " + str(muestras_removidas))

    if cantidad_original > 0:
        porcentaje_conservado = (cantidad_final / cantidad_original) * 100
        print("  Porcentaje conservado: " + "{:.1f}".format(porcentaje_conservado) + "%")

    dataframe.to_csv(ARCHIVO_LIMPIO, index=False)

    return dataframe


def visualizar_distribucion(dataframe):
    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)

    figura, ejes = plt.subplots(figsize=(12, 6))

    conteo_clases = dataframe['etiqueta'].value_counts().sort_index()

    nombres_gestos = []

    for indice in conteo_clases.index:
        nombre = GESTOS.get(indice, "Gesto " + str(indice))
        nombres_gestos.append(nombre)

    colores = plt.cm.Set3(np.linspace(0, 1, len(conteo_clases)))
    barras = ejes.bar(nombres_gestos, conteo_clases.values, color=colores, edgecolor='black')

    ejes.axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Mínimo')

    indice_barra = 0

    while indice_barra < len(barras):
        barra_actual = barras[indice_barra]
        valor_actual = conteo_clases.values[indice_barra]

        posicion_x = barra_actual.get_x() + barra_actual.get_width() / 2.0
        posicion_y = barra_actual.get_height() + 10

        ejes.text(
            posicion_x,
            posicion_y,
            str(valor_actual),
            ha='center',
            va='bottom',
            fontweight='bold',
        )

        indice_barra = indice_barra + 1

    ejes.set_title('Distribución de Muestras por Gesto', fontsize=14, fontweight='bold')
    ejes.set_xlabel('Gesto', fontsize=12)
    ejes.set_ylabel('Número de Muestras', fontsize=12)
    ejes.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    ruta_figura_1 = os.path.join(DIRECTORIO_FIGURAS, 'distribucion_gestos.png')
    plt.savefig(ruta_figura_1, dpi=150)
    plt.close()

    columnas_dedos = [columna for columna in dataframe.columns if columna.startswith('dedo_')]

    if columnas_dedos:
        figura, ejes = plt.subplots(figsize=(12, 6))

        promedios_dedos = dataframe.groupby('etiqueta')[columnas_dedos].mean()

        etiquetas_dedos = ['Pulgar', 'Índice', 'Medio', 'Anular', 'Meñique']
        promedios_dedos.columns = etiquetas_dedos

        etiquetas_gestos = []

        for indice in promedios_dedos.index:
            nombre = GESTOS.get(indice, "Gesto " + str(indice))
            etiquetas_gestos.append(nombre)

        promedios_dedos.index = etiquetas_gestos

        sns.heatmap(
            promedios_dedos,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ejes,
            linewidths=0.5,
        )

        ejes.set_title('Estado Promedio de Dedos por Gesto', fontsize=14, fontweight='bold')
        ejes.set_ylabel('Gesto')
        ejes.set_xlabel('Dedo')
        plt.tight_layout()

        ruta_figura_2 = os.path.join(DIRECTORIO_FIGURAS, 'dedos_por_gesto.png')
        plt.savefig(ruta_figura_2, dpi=150)
        plt.close()

    from sklearn.decomposition import PCA

    columnas_landmarks = [columna for columna in dataframe.columns if columna.startswith('punto')]
    datos_x = dataframe[columnas_landmarks].values
    datos_y = dataframe['etiqueta'].values

    analizador_pca = PCA(n_components=2)
    datos_2d = analizador_pca.fit_transform(datos_x)

    figura, ejes = plt.subplots(figsize=(12, 8))

    etiquetas_unicas = sorted(dataframe['etiqueta'].unique())

    indice_etiqueta = 0

    while indice_etiqueta < len(etiquetas_unicas):
        etiqueta_actual = etiquetas_unicas[indice_etiqueta]
        mascara = datos_y == etiqueta_actual

        nombre_gesto = GESTOS.get(etiqueta_actual, "Gesto " + str(etiqueta_actual))

        ejes.scatter(
            datos_2d[mascara, 0],
            datos_2d[mascara, 1],
            label=nombre_gesto,
            alpha=0.5,
            s=10,
        )

        indice_etiqueta = indice_etiqueta + 1

    varianza_pc1 = analizador_pca.explained_variance_ratio_[0]
    varianza_pc2 = analizador_pca.explained_variance_ratio_[1]

    ejes.set_title('Visualización PCA de los Gestos', fontsize=14, fontweight='bold')
    ejes.set_xlabel('PC1 (' + "{:.1%}".format(varianza_pc1) + ' varianza)')
    ejes.set_ylabel('PC2 (' + "{:.1%}".format(varianza_pc2) + ' varianza)')
    ejes.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    ruta_figura_3 = os.path.join(DIRECTORIO_FIGURAS, 'pca_gestos.png')
    plt.savefig(ruta_figura_3, dpi=150)
    plt.close()

def main():

    dataframe = combinar_archivos_csv()

    if dataframe is None or len(dataframe) == 0:
        return

    dataframe = limpiar_datos(dataframe)
    
    visualizar_distribucion(dataframe)
    
if __name__ == "__main__":
    main()