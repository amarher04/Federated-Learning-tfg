import requests
import tensorflow as tf
import numpy as np
import time
import sys
import os

# URL escucha del servidor FastAPI
URL_SERVIDOR = "http://127.0.0.1:8000"

# Crea un modelo MLP simple para clasificación de MNIST
def crear_modelo():
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def ejecutar_cliente_autonomo(id_cliente):
    print(f"\n{'='*10} INICIANDO CLIENTE {id_cliente} {'='*10}\n")
    
    # Preparar datos fijos para este cliente
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    #mitad = len(x_train) // 2
    #x_local, y_local = (x_train[:mitad], y_train[:mitad]) if id_cliente == "1" else (x_train[mitad:], y_train[mitad:])
    #num_muestras_locales = len(x_local)
    x_local, y_local = x_train[100:200], y_train[100:200]
    num_muestras_locales = len(x_local)
    modelo = crear_modelo()
    
    ronda_completada_por_mi = 0
    experimento_actual_por_mi = None # Para detectar cambios en la configuración del experimento (Si se inicia uno nuevo)
    
    # Bucle infinito de polling (Vigila al servidor)
    while True:
        print("Consultando configuración al servidor...\n")
        try:
            config = requests.get(f"{URL_SERVIDOR}/config").json()
            ronda_servidor = config["ronda_actual"]
            experimento_servidor = config["experimento_id"] # Leemos ID del experimento
            rondas_objetivo = config["rondas_objetivo"] # Leemos numero de rondas objetivo para este experimento
            
            # SINCRONIZAR SEMILLA
            if "seed_actual" in config:
                np.random.seed(config["seed_actual"])
                tf.random.set_seed(config["seed_actual"])
            
            # Si el servidor no ha iniciado un experimento (rondas_objetivo = 0) o ya hemos completado el experimento, esperamos
            if rondas_objetivo == 0 or ronda_servidor > rondas_objetivo:
                print(f"\r[Cliente {id_cliente}] Modo reposo. Esperando nuevo experimento... ", end="")
                time.sleep(5)
                continue  # Volvemos al inicio del bucle sin descargar ni entrenar
            
            # Si el servidor ha iniciado un nuevo experimento (ID diferente al que yo he completado), reseteamos
            if experimento_servidor != experimento_actual_por_mi:
                print(f"\n [Cliente {id_cliente}] Nuevo experimento detectado: {experimento_servidor}.\n")
                experimento_actual_por_mi = experimento_servidor
                ronda_completada_por_mi = 0 # Volvemos a empezar desde la ronda 0 para el nuevo experimento
            
            # Si el servidor ya ha avanzado a una ronda superior a la que yo he completado, me pongo a trabajar
            if ronda_servidor > ronda_completada_por_mi:
                print(f"\n [Cliente {id_cliente}] Nueva ronda detectada: {ronda_servidor} de {rondas_objetivo}. Empezando entrenamiento local...\n")
                
                # 1. Descargar modelo global
                resp_modelo = requests.get(f"{URL_SERVIDOR}/model/download")
                ruta_descarga = f"modelo_descargado_c{id_cliente}.npz"
                with open(ruta_descarga, "wb") as f:
                    f.write(resp_modelo.content)
                
                with np.load(ruta_descarga) as datos_npz:
                    modelo.set_weights([datos_npz[f] for f in datos_npz.files])
                    
                # 2. Entrenar localmente
                print(f"[Cliente {id_cliente}] Entrenando {config['epochs_locales']} epochs...\n")
                
                inicio_entrenamiento = time.time()
                modelo.fit(x_local, y_local, epochs=config["epochs_locales"], verbose=0)
                fin_entrenamiento = time.time()
                tiempo_ent_local = fin_entrenamiento - inicio_entrenamiento
                
                # 3. Subir con metadatos (Formulario)
                ruta_subida = f"modelo_subir_c{id_cliente}.npz"
                np.savez(ruta_subida, *modelo.get_weights())
                
                datos_formulario = {
                    "cliente_id": id_cliente,
                    "num_muestras": num_muestras_locales,
                    "ronda_cliente": ronda_servidor,
                    "tiempo_entrenamiento": tiempo_ent_local
                }
                archivos = {"file": (f"pesos_c{id_cliente}.npz", open(ruta_subida, "rb"), "application/octet-stream")}
                
                print(f"[Cliente {id_cliente}] Subiendo resultados al servidor...\n")
                requests.post(f"{URL_SERVIDOR}/model/upload", data=datos_formulario, files=archivos)
                
                ronda_completada_por_mi = ronda_servidor
                print(f"[Cliente {id_cliente}] Esperando a que el resto termine... \n")
                
            time.sleep(5)  # Espera un poco antes de volver a consultar al servidor
                
        except requests.exceptions.ConnectionError:
            print(f"[Cliente {id_cliente}] Buscando al servidor...")
            time.sleep(5)
            
        except KeyError:
            time.sleep(1)  # Por si el servidor esta reiniciandose y no ha enviado aun toda la configuración
    
if __name__ == "__main__":
    # Pedimos al usuario que identifique el cliente (1 o 2)
    mi_id = input("Introduce el ID del cliente (ej: 1 o 2) y pulsa ENTER: \n")
    ejecutar_cliente_autonomo(mi_id)
    
