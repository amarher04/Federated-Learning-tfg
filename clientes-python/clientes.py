import requests
import tensorflow as tf
import numpy as np
import time
import sys
import os

# URL escucha del servidor FastAPI
URL_SERVIDOR = "http://127.0.0.1:8000"

NUM_CLIENTES_FASE = 2  # Número de clientes que participan en la fase de entrenamiento

# ELIGE EL ESCENARIO DEL EXPERIMENTO 3.1:
# Opciones para Python: 100 (Escenario 1:1), 1000 (Escenario 1:10), 5000 (Escenario 1:50)
MUESTRAS_PYTHON_EXP_3_1 = 100 

# Crea un modelo MLP simple para clasificación de MNIST
def crear_modelo():
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def obtener_datos_cliente(id_cliente, modo="IID", num_clientes=NUM_CLIENTES_FASE):
    """
    Reparte 10.000 muestras entre el numero total de clientes indicado (num_clientes).
    id_cliente debe ser un número del 1 al num_clientes.
    """
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype(np.float32)
    
    id_num = int(id_cliente)
    if id_num >= 4:
        idx_cliente = id_num - 4  # ID 4 será el índice 0, ID 10 será el 6
    else:
        idx_cliente = id_num - 1  # Por compatibilidad si ejecutas ID 1, 2 o 3 en PC
    
    # Aqui indicamos el numero de muestras total que queremos repartir entre los clientes y lo dividimos entre el numero de clientes
    #tam_lote = 10000 // num_clientes
    
    tam_lote = MUESTRAS_PYTHON_EXP_3_1
    
    if modo == "IID":
        # Reparto aleatorio uniforme (1000 muestras variadas por cliente)
        # Usamos una semilla fija para el reparto inicial para que sea reproducible
        np.random.seed(42)
        indices = np.random.permutation(len(x_train))
        mis_indices = indices[idx_cliente * tam_lote : (idx_cliente + 1) * tam_lote]
        return x_train[mis_indices], y_train[mis_indices]
        
    elif modo == "NON_IID":
        # Reparto patológico: cada cliente solo recibe 2 dígitos consecutivos
        # Cliente 1 -> dígitos 0 y 1 | Cliente 2 -> dígitos 2 y 3 ...
        digito_1 = (idx_cliente * 2) % 10
        digito_2 = (idx_cliente * 2 + 1) % 10

        # Filtramos solo los índices que corresponden a esos dos dígitos
        indices_filtrados = np.where((y_train == digito_1) | (y_train == digito_2))[0]
        
        # Cogemos 1.000 muestras de ese subconjunto
        mis_indices = indices_filtrados[:tam_lote]
        return x_train[mis_indices], y_train[mis_indices]

def ejecutar_cliente_autonomo(id_cliente, modo_datos="IID"):
    print(f"\n{'='*10} INICIANDO CLIENTE {id_cliente} (Modo: {modo_datos}) {'='*10}\n")
    
    # Preparar datos fijos para este cliente
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_local, y_local = obtener_datos_cliente(id_cliente, modo_datos, num_clientes=NUM_CLIENTES_FASE)
    num_muestras_locales = len(x_local)
    print(f"\n[*] Datos asignados con éxito: {num_muestras_locales} muestras locales.")
    
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
                
                # ELIGE EL MODO PARA EL EXPERIMENTO 2.3:
                # Opciones: "F32_RAW", "F32_ZIP", "F16_RAW", "F16_ZIP"
                MODO_TELECOM = "F32_RAW"
                
                # 3. Subir con metadatos (Formulario)
                ruta_subida = f"modelo_subir_c{id_cliente}.npz"
                pesos_actuales = modelo.get_weights()
                
                if MODO_TELECOM == "F32_RAW":
                    # Float32 estándar sin comprimir
                    np.savez(ruta_subida, *pesos_actuales)
                elif MODO_TELECOM == "F32_ZIP":
                    # Float32 comprimido
                    np.savez_compressed(ruta_subida, *pesos_actuales)
                elif MODO_TELECOM == "F16_RAW":
                    # Convertimos cada capa a Float16 sin comprimir
                    pesos_f16 = [capa.astype(np.float16) for capa in pesos_actuales]
                    np.savez(ruta_subida, *pesos_f16)
                elif MODO_TELECOM == "F16_ZIP":
                    # Convertimos a Float16 y comprimimos
                    pesos_f16 = [capa.astype(np.float16) for capa in pesos_actuales]
                    np.savez_compressed(ruta_subida, *pesos_f16)
                
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
    mi_id = input("Introduce el ID del cliente y pulsa ENTER: \n")
    # CAMBIA AQUÍ A "IID" o "NON_IID" según la prueba que estés haciendo:
    modo = "IID"
    ejecutar_cliente_autonomo(mi_id, modo_datos=modo)
