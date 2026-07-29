from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import os
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime
from pydantic import BaseModel
import time

app = FastAPI(title="Servidor Aprendizaje Federado - V3")
os.makedirs("modelos_clientes", exist_ok=True)

# Configuración del log de datos en CSV (Se puede cambiar nombre para cada experimento)
ARCHIVO_CSV = "resultados_2_3_F16ZIP.csv"

# Si el archivo CSV no existe, lo creamos y escribimos la cabecera
if not os.path.exists(ARCHIVO_CSV):
    with open(ARCHIVO_CSV, mode='w', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["Fecha", "Experimento_ID", "Ronda", "Epochs_Locales", "Muestras_Totales", "Accuracy", "Loss", "Trafico_Ronda_KB", "Tiempo_Ronda_Seg", "Espera_Rezagados_Seg", "Tiempo_Medio_Entrenamiento_Seg", "Trafico_Acumulado_KB"])

# 1. PREPARAR EVALUACIÓN DE MODELO GLOBAL
print("Cargando datos de validación en el servidor...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalización

def crear_modelo():
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def guardar_modelo_binario(pesos, ruta="modelo_global.bin"):
    with open(ruta, "wb") as f:
        for capa in pesos:
            f.write(capa.astype(np.float32).tobytes())

modelo_global = crear_modelo()
np.savez("modelo_global.npz", *modelo_global.get_weights())
guardar_modelo_binario(modelo_global.get_weights())

# 2. ESTADO DEL SERVIDOR
estado_servidor = {
    "experimento_id": f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}", # ID único para cada ejecución del servidor basado en la fecha y hora de inicio.
    "ronda_actual": 1,
    "rondas_objetivo": 0, # Usamos como flag para saber si hay un experimento en curso o no (Si es 0, no hay experimento iniciado, servidor en reposo)
    "epochs_locales": 3,
    "clientes_esperados": 5,  # Número de clientes que esperamos recibir en cada ronda
    "metadatos_recibidos": [], # Guardara diccionarios con info de cada cliente (id, accuracy_local, etc)
    "tiempo_inicio_ronda": time.time(), # para calcular el tiempo total de la ronda
    "trafico_acumulado": 0.0,
    "seed_actual": 42
}

# 3. ENDPOINTS (GET /config, GET /model/download, POST /model/upload)
@app.get("/config")  # Endpoint para obtener configuración actual del servidor
def obtener_configuracion():
    return estado_servidor

@app.get("/model/download")  # Endpoint para descargar modelo .npz (Python)
def descargar_modelo():
    ruta_archivo = "modelo_global.npz"
    if os.path.exists(ruta_archivo):
        return FileResponse("modelo_global.npz", media_type="application/octet-stream", filename="modelo_global.npz")
    else:
        return {"error": "Modelo global no encontrado"}

@app.get("/model/download/bin")  # Endpoint para descargar modelo .bin (Android)
def descargar_modelo_bin():
    if os.path.exists("modelo_global.bin"):
        return FileResponse("modelo_global.bin", media_type="application/octet-stream", filename="modelo_global.bin")
    return {"error": "Modelo binario no encontrado"}

@app.post("/model/upload")  # Endpoint para subir modelo .npz (Python)
async def subir_pesos(
    file: UploadFile = File(...),           # METADATO 1
    cliente_id: str = Form(...),            # METADATO 2
    num_muestras: int = Form(...),          # METADATO 3
    ronda_cliente: int = Form(...),         # METADATO 4
    tiempo_entrenamiento: float = Form(...) # METADATO 5
):
    # Seguridad: Si un cliente va con retraso, ignoramos sus pesos
    if ronda_cliente != estado_servidor["ronda_actual"]:
        return {"error": "Ronda obsoleta"}
    
    ruta_guardado = f"modelos_clientes/{cliente_id}_{file.filename}"
    with open(ruta_guardado, "wb") as buffer:
        contenido = await file.read()
        buffer.write(contenido)
    
    # Guardamos los metadatos del cliente y la ruta de su archivo
    estado_servidor["metadatos_recibidos"].append({
        "id": cliente_id,
        "muestras": num_muestras,
        "ruta": ruta_guardado,
        "trafico_kb": len(contenido) / 1024.0,
        "tiempo_entrenamiento": tiempo_entrenamiento,
        "momento_llegada": time.time() # Registramos a qué hora exacta llego el cliente
    })
    
    print(f" Recibido Cliente {cliente_id} | Muestras: {num_muestras} | Ronda: {ronda_cliente} \n")
    
    # Lógica de agregación FedAvg ponderada
    if len(estado_servidor["metadatos_recibidos"]) >= estado_servidor["clientes_esperados"]:
        print(f"\n ¡Ejecutando FedAvg Ponderado! (Ronda {estado_servidor['ronda_actual']})... \n")
        
        # 1. Calcular N total (Suma de las muestras de todos)
        muestras_totales = sum(meta["muestras"] for meta in estado_servidor["metadatos_recibidos"])
        
        todos_los_pesos_ponderados = []
        for meta in estado_servidor["metadatos_recibidos"]:
            # Identificamos el formato del cliente (ANDROID .bin o PYTHON .npz)
            ruta_archivo = meta["ruta"]
            
            if ruta_archivo.endswith(".bin"):
                # Lógica para cliente ANDROID
                with open(ruta_archivo, "rb") as f:
                    datos = np.fromfile(f, dtype=np.float32)
                    
                # Reconstruimos la geometría de las 4 capas de la red
                w1 = datos[0 : 100352].reshape(784, 128)
                b1 = datos[100352 : 100480]
                w2 = datos[100480 : 101760].reshape(128, 10)
                b2 = datos[101760 : 101770]
                pesos_brutos = [w1, b1, w2, b2]
            
            else:
                # Lógica para cliente PYTHON (.npz)
                with np.load(ruta_archivo) as datos_npz:
                    pesos_brutos = [datos_npz[f].astype(np.float32) for f in datos_npz.files]
            
            # Ponderamos cada capa
            factor_importancia = meta["muestras"] / float(muestras_totales)
            pesos_ponderados = [capa * factor_importancia for capa in pesos_brutos]
            todos_los_pesos_ponderados.append(pesos_ponderados)
        
        # 2. Sumar los pesos ponderados de todos los clientes para cada capa
        nuevos_pesos_globales = []
        num_capas = len(todos_los_pesos_ponderados[0])
        for i in range(num_capas):
            capa_sumada = np.sum([cliente[i] for cliente in todos_los_pesos_ponderados], axis=0)
            nuevos_pesos_globales.append(capa_sumada)
        
        # 3. Calcular tiempo total de la ronda y tiempo medio de entrenamiento
        tiempo_fin_ronda = time.time()
        tiempo_total_ronda = tiempo_fin_ronda - estado_servidor["tiempo_inicio_ronda"]
        
        # Efecto Straggler (Espera por clientes lentos): Diferencia entre el primer cliente en llegar y el último
        tiempos_llegada = [m["momento_llegada"] for m in estado_servidor["metadatos_recibidos"]]
        espera_rezagados = max(tiempos_llegada) - min(tiempos_llegada) if len(tiempos_llegada) > 1 else 0
        
        # Tiempo medio de entrenamiento local
        tiempo_medio_ent = np.mean([m["tiempo_entrenamiento"] for m in estado_servidor["metadatos_recibidos"]])
            
        # 4. Evaluar y Loggear resultados
        modelo_global.set_weights(nuevos_pesos_globales)
        loss, acc = modelo_global.evaluate(x_test, y_test, verbose=0)
        trafico_total_ronda = sum(m["trafico_kb"] for m in estado_servidor["metadatos_recibidos"]) * 2 # Subida + Descarga
        estado_servidor["trafico_acumulado"] += trafico_total_ronda # Sumamos al histórico
        
        print(f" --- RESULTADOS RONDA {estado_servidor['ronda_actual']} --- \n")
        print(f"    Accuracy: {acc*100:.2f}% \n")
        print(f"    Loss: {loss:.4f} \n")
        print(f"    Tráfico: {trafico_total_ronda:.2f} KB \n")
        
        # --- Guardar resultados en CSV ---
        with open(ARCHIVO_CSV, mode='a', newline='') as archivo:
            writer = csv.writer(archivo)
            fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([
                fecha_actual,
                estado_servidor["experimento_id"],
                estado_servidor["ronda_actual"],
                estado_servidor["epochs_locales"],
                muestras_totales,
                round(acc, 4),
                round(loss, 4),
                round(trafico_total_ronda, 2),
                round(tiempo_total_ronda, 2),
                round(espera_rezagados, 2),
                round(tiempo_medio_ent, 2),
                round(estado_servidor["trafico_acumulado"], 2)
            ])
        print(f" Resultados de la ronda {estado_servidor['ronda_actual']} guardados en {ARCHIVO_CSV} \n")
        """
        # LOGICA DE PARADA TEMPRANA (90% ACCURACY)
        if acc >= 0.925 or estado_servidor["ronda_actual"] >= estado_servidor["rondas_objetivo"]:
            print("\n>>> OBJETIVO ALCANZADO (92.5% Acc) o RONDAS FINALIZADAS. Deteniendo experimento... <<<\n")
            estado_servidor["rondas_objetivo"] = 0 # Esto avisa al Director y a los clientes de que paren
        """    
        if estado_servidor["ronda_actual"] >= estado_servidor["rondas_objetivo"]:
            print("\n>>> RONDAS FINALIZADAS. Deteniendo experimento... <<<\n")
            estado_servidor["rondas_objetivo"] = 0 # Esto avisa al Director y a los clientes de que paren
        
        # 5. Guardar nuevo modelo global actualizado, Limpiar y Avanzar ronda
        np.savez("modelo_global.npz", *nuevos_pesos_globales)
        guardar_modelo_binario(nuevos_pesos_globales)
        for meta in estado_servidor["metadatos_recibidos"]:
            try:
                if os.path.exists(meta["ruta"]):
                    os.remove(meta["ruta"])
            except OSError:
                pass
        estado_servidor["metadatos_recibidos"].clear()
        estado_servidor["ronda_actual"] += 1
        estado_servidor["tiempo_inicio_ronda"] = time.time() # Reiniciamos el reloj para la siguiente ronda
        
    return {"mensaje": "Recibido y procesado correctamente"}

# Creamos un modelo para la petición de configuración del servidor
class NuevaConfiguracion(BaseModel):
    epochs_locales: int
    rondas_objetivo: int  # El director nos dira cuantas rondas quiere que dure el experimento
    seed: int


@app.post("/admin/configurar")  # Endpoint para configurar el servidor (modo admin)
def configurar_servidor(config: NuevaConfiguracion):
    # Fijar semillas para reproducibilidad
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    
    # Reseteamos el estado del servidor para un nuevo experimento de cero
    modelo_nuevo = crear_modelo()
    np.savez("modelo_global.npz", *modelo_nuevo.get_weights())
    guardar_modelo_binario(modelo_nuevo.get_weights())
    
    # Limpiar variables del servidor
    estado_servidor["epochs_locales"] = config.epochs_locales
    estado_servidor["ronda_actual"] = 1
    estado_servidor["rondas_objetivo"] = config.rondas_objetivo # Se activa flag de experimento en curso con el numero de rondas objetivo
    estado_servidor["metadatos_recibidos"].clear()
    estado_servidor["trafico_acumulado"] = 0.0
    estado_servidor["seed_actual"] = config.seed
    
    # Crear un nuevo ID de experimento para diferenciarlo en el CSV (ponemos epochs en el ID para identificarlo fácilmente)
    estado_servidor["experimento_id"] = f"EXP_EPOCHS_{config.epochs_locales}_{datetime.now().strftime('%H%M%S')}_SEED_{config.seed}"
    print(f"\n{'='*40}")
    print(f" MODO ADMIN: nuevo experimento iniciado con {config.epochs_locales} epochs locales")
    print(f"\n{'='*40}")
    
    return {"mensaje": f"Experimento reiniciado con {config.epochs_locales} epochs."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # Aceptamos conexiones de cualquier dispositivo en la red local
