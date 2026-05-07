import requests
import tensorflow as tf
import numpy as np
import sys
import os

# URL escucha del servidor FastAPI
URL_SERVIDOR = "http://127.0.0.1:8000"

# Crea un modelo MLP simple para clasificación de MNIST
def crear_modelo():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def ejecutar_cliente(id_cliente):
    print(f"\n{'='*10} INICIANDO CLIENTE {id_cliente} {'='*10}\n")
    
    # 1. Obtener configuración del servidor
    print ("1. Consultando configuración al servidor (GET/config)...\n")
    try:
        respuesta = requests.get(f"{URL_SERVIDOR}/config")
        config = respuesta.json()
        print(f"   -> El servidor ordena hacer {config['epochs_locales']} epochs para la ronda {config['ronda_actual']}.\n")
    except requests.exceptions.ConnectionError:
        print("   -> [ERROR] No se pudo conectar al servidor. Asegúrate de que el servidor esté encendido.\n")
        return
    
    # 2. Descargar modelo global
    print("2. Descargando modelo global actual del servidor (GET/model/download)...\n")
    respuesta_modelo = requests.get(f"{URL_SERVIDOR}/model/download")
    ruta_descarga = f"modelo_descargado_c{id_cliente}.npz"
    
    with open(ruta_descarga, "wb") as f:
        f.write(respuesta_modelo.content)
    print(f"   -> Archivo .npz descargado correctamente.\n")
    
    # 3. Cargar el modelo global descargado (simulación)
    print("3. Cargando fotos y preparando el cerebro...\n")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0  # Normalización
    
    # Dividimos los datos: Cliente 1 estudia la primera mitad, Cliente 2 la segunda mitad
    mitad = len(x_train) // 2
    if id_cliente == "1":
        x_local, y_local = x_train[:mitad], y_train[:mitad]
    else:
        x_local, y_local = x_train[mitad:], y_train[mitad:]
        
    modelo = crear_modelo()
    
    # Desempaquetar el archivo .npz y meter los pesos a la red neuronal
    datos_npz = np.load(ruta_descarga)
    # Comprobamos si es el modelo "dummy" inicial del servidor o uno real
    if "dummy" not in datos_npz.files:
        pesos_descargados = [datos_npz[f] for f in datos_npz.files]
        modelo.set_weights(pesos_descargados)
        print("   -> Pesos globales inyectados con éxito.\n")
    else:
        print("   -> Modelo global inicial (dummy) detectado. Uso uno nuevo.\n")
    
    # 4. Entrenamiento local
    print(f"4. Entrenando localmente de forma privada con los datos del Cliente {id_cliente}...\n")
    modelo.fit(x_local, y_local, epochs=config['epochs_locales'], verbose=1)
    
    # 5. Extraer pesos actualizados y guardarlos en un archivo .npz
    print("5. Extrayendo el conocimiento y creando paquete .npz...\n")
    pesos_nuevos = modelo.get_weights()
    ruta_subida = f"modelo_para_subir_c{id_cliente}.npz"
    np.savez(ruta_subida, *pesos_nuevos) # Guardamos lista de matrices en archivo comprimido binario
    
    # 6. Enviar pesos al servidor
    print("6. Subiendo pesos actualizados al servidor (POST/model/upload)...\n")
    with open(ruta_subida, "rb") as f:
        # Preparamos archivo como multipart/form-data para la petición POST
        archivos = {"file": (f"pesos_c{id_cliente}.npz", f, "application/octet-stream")}
        respuesta_upload = requests.post(f"{URL_SERVIDOR}/model/upload", files=archivos)
    
    print(f"   -> Respuesta del servidor: {respuesta_upload.json()}\n")
    print(f"{'='*30}\n")
    
    
if __name__ == "__main__":
    # Pedimos al usuario que identifique el cliente (1 o 2)
    mi_id = input("Introduce el ID del cliente (ej: 1 o 2) y pulsa ENTER: \n")
    ejecutar_cliente(mi_id)
    
