# El código siguiente, que crea un dataframe y quita las filas duplicadas, siempre se ejecuta y actúa como un preámbulo del script: 

# dataset = pandas.DataFrame(Orientacion, Luces_apagadas, AACC_enc, Calefacción_enc, Ventanas_abiertas, FH, Cortinas_abiertas, Ocupación_aula, TempEXTHIC, TempEXTMED, TempEXTHFC, HumEXTHIC, HumEXTMED, HumEXTHFC, TempINTMIN, TempINTMAX, TempINTMED, LumINTMIN, LumINTMAX, LumINTMED, HumINTMIN, HumINTMAX, HumINTMED)
# dataset = dataset.drop_duplicates()

# Pegue o escriba aquí el código de script:
import pandas as pd
import pickle
import urllib.request
import matplotlib.pyplot as plt

# URL de los archivos
ruta_descarga_ba = "https://raw.githubusercontent.com/vanessarodriguezhorcajo/Predictive-Model-for-Indoor-Environmental-Quality-in-Learning-Spaces/main/Predictive%20trained%20model%20of%20student%20well-being/knnc_ba.pkl"
ruta_descarga_scaler = "https://raw.githubusercontent.com/vanessarodriguezhorcajo/Predictive-Model-for-Indoor-Environmental-Quality-in-Learning-Spaces/main/Predictive%20trained%20model%20of%20student%20well-being/minmax_scaler.pkl"

# Función para descargar y cargar un archivo .pkl
def descargar_y_cargar_modelo(url):
    response = urllib.request.urlopen(url)
    if response.status == 200:
        data = response.read()
        return pickle.loads(data)
    else:
        raise Exception(f"Error al descargar el archivo: {response.status}")

# Cargar el modelo y el scaler
modelo_ba = descargar_y_cargar_modelo(ruta_descarga_ba)
scaler = descargar_y_cargar_modelo(ruta_descarga_scaler)

# Funciones de mapeo
def asignar_si_no_p(valor):
    if valor == 'Sí':
        return 1
    elif valor == 'No':
        return 0
    else:
        return 2

def asignar_fh(valor):
    mapeo_fh = {
        '09:00:00 - 11:00:00': 1,
        '11:00:00 - 13:00:00': 2,
        '13:00:00 - 15:00:00': 3,
        '15:00:00 - 17:00:00': 4,
        '17:00:00 - 19:00:00': 5,
        '19:00:00 - 21:00:00': 6
    }
    return mapeo_fh.get(valor, 0)

def asignar_punto_cardinal(valor):
    mapeo_grado_bienestar = {
        'N': 1,
        'S': 2,
        'E': 3,
        'O': 4,
        'NE': 5,
        'SE': 6,
        'NO': 7,
        'SO': 8,
    }
    return mapeo_grado_bienestar.get(valor, 0)

# Preprocesamiento de datos
def preprocesado(datos):
    datos['Luces_apagadas'] = datos['Luces_apagadas'].apply(asignar_si_no_p)
    datos['AACC_enc'] = datos['AACC_enc'].apply(asignar_si_no_p)
    datos['Calefacción_enc'] = datos['Calefacción_enc'].apply(asignar_si_no_p)
    datos['Ventanas_abiertas'] = datos['Ventanas_abiertas'].apply(asignar_si_no_p)
    datos['Orientacion'] = datos['Orientacion'].apply(asignar_punto_cardinal)
    datos['Cortinas_abiertas'] = datos['Cortinas_abiertas'].astype(int)
    datos['FH'] = datos['FH'].apply(asignar_fh)

    columnas_a_escalar = list(datos.columns)[7:-3]
    for columna in columnas_a_escalar:
        datos[columna] = scaler.fit_transform(datos[[columna]])

    return datos

# Función de predicción
def predecir_clase_ea(datos):
    prediccion = modelo_ba.predict(datos)
    return pd.DataFrame(prediccion, columns=['Estado_animico'])

# Reordenar columnas
nuevo_orden = ['Orientacion', 'Luces_apagadas', 'Cortinas_abiertas', 'AACC_enc',
               'Calefacción_enc', 'Ventanas_abiertas', 'Ocupación_aula', 'TempEXTHIC',
               'TempEXTMED', 'TempEXTHFC', 'HumEXTHIC', 'HumEXTMED', 'HumEXTHFC',
               'TempINTMIN', 'TempINTMAX', 'TempINTMED', 'LumINTMIN', 'LumINTMAX',
               'LumINTMED', 'HumINTMIN', 'HumINTMAX', 'HumINTMED', 'FH']

dataset = dataset[nuevo_orden]

# Preprocesar los datos
datos_procesados = preprocesado(dataset)

# Realizar predicciones
df_ea = predecir_clase_ea(datos_procesados)

# Mostrar bienestar anímico
fig, ax = plt.subplots()
ax.axis('off')
valor = df_ea.values[0][0]
ax.text(0.5, 0.6, 'Grado de bienestar anímico del alumno', ha='center', fontsize=18)
ax.text(0.5, 0.4, f'Predicción: {valor}', ha='center', fontsize=16)
plt.show()