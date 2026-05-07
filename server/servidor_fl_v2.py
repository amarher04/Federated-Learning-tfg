from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import os
import numpy as np
import tensorflow as tf

app = FastAPI(title="Servidor Aprendizaje Federado - V2")
os.makedirs("modelos_clientes", exist_ok=True)

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

modelo_global = crear_modelo()
np.savez("modelo_global.npz", *modelo_global.get_weights())

# 2. ESTADO DEL SERVIDOR
estado_servidor = {
    "ronda_actual": 1,
    "epochs_locales": 3,
    "clientes_esperados": 2,
    "metadatos_recibidos": [] # Guardara diccionarios con info de cada cliente (id, accuracy_local, etc)
}

# 3. ENDPOINTS (GET /config, GET /model/download, POST /model/upload)
@app.get("/config")
def obtener_configuracion():
    return estado_servidor

@app.get("/model/download")
def descargar_modelo():
    ruta_archivo = "modelo_global.npz"
    if os.path.exists(ruta_archivo):
        return FileResponse("modelo_global.npz", media_type="application/octet-stream", filename="modelo_global.npz")
    else:
        return {"error": "Modelo global no encontrado"}

@app.post("/model/upload")
async def subir_pesos(
    file: UploadFile = File(...),   # METADATO 1
    cliente_id: str = Form(...),    # METADATO 2
    num_muestras: int = Form(...),  # METADATO 3
    ronda_cliente: int = Form(...)  # METADATO 4
):
    # Seguridad: Si un cliente va con retraso, ignoramos sus pesos
    if ronda_cliente != estado_servidor["ronda_actual"]:
        return {"error": "Ronda obsoleta"}
    
    ruta_guardado = f"modelos_clientes/{file.filename}"
    with open(ruta_guardado, "wb") as buffer:
        contenido = await file.read()
        buffer.write(contenido)
    
    # Guardamos los metadatos del cliente y la ruta de su archivo
    estado_servidor["metadatos_recibidos"].append({
        "id": cliente_id,
        "muestras": num_muestras,
        "ruta": ruta_guardado,
        "trafico_kb": len(contenido) / 1024.0
    })
    
    print(f" Recibido Cliente {cliente_id} | Muestras: {num_muestras} | Ronda: {ronda_cliente} \n")
    
    # Lógica de agregación FedAvg ponderada
    if len(estado_servidor["metadatos_recibidos"]) >= estado_servidor["clientes_esperados"]:
        print(f"\n ¡Ejecutando FedAvg Ponderado! (Ronda {estado_servidor['ronda_actual']})... \n")
        
        # 1. Calcular N total (Suma de las muestras de todos)
        muestras_totales = sum(meta["muestras"] for meta in estado_servidor["metadatos_recibidos"])
        
        todos_los_pesos_ponderados = []
        for meta in estado_servidor["metadatos_recibidos"]:
            with np.load(meta["ruta"]) as datos_npz:
                pesos_brutos = [datos_npz[f] for f in datos_npz.files]
                # Ponderamos cada capa multiplicando los pesos de este cliente por (sus_muestras / muestras_totales)
                factor_importancia = meta["muestras"] / float(muestras_totales)
                pesos_ponderados = [capa * factor_importancia for capa in pesos_brutos]
                todos_los_pesos_ponderados.append(pesos_ponderados)
        
        # 2. Sumar los pesos ponderados de todos los clientes para cada capa
        nuevos_pesos_globales = []
        num_capas = len(todos_los_pesos_ponderados[0])
        for i in range(num_capas):
            capa_sumada = np.sum([cliente[i] for cliente in todos_los_pesos_ponderados], axis=0)
            nuevos_pesos_globales.append(capa_sumada)
            
        # 3. Evaluar y Loggear resultados
        modelo_global.set_weights(nuevos_pesos_globales)
        loss, acc = modelo_global.evaluate(x_test, y_test, verbose=0)
        trafico_total_ronda = sum(m["trafico_kb"] for m in estado_servidor["metadatos_recibidos"]) * 2 # Subida + Descarga
        
        print(f" --- RESULTADOS RONDA {estado_servidor['ronda_actual']} --- \n")
        print(f"    Accuracy: {acc*100:.2f}% \n")
        print(f"    Loss: {loss:.4f} \n")
        print(f"    Tráfico: {trafico_total_ronda:.2f} KB \n")
        
        # 4. Guardar nuevo modelo global actualizado, Limpiar y Avanzar ronda
        np.savez("modelo_global.npz", *nuevos_pesos_globales)
        for meta in estado_servidor["metadatos_recibidos"]:
            os.remove(meta["ruta"])
        estado_servidor["metadatos_recibidos"].clear()
        estado_servidor["ronda_actual"] += 1
        
    return {"mensaje": "Recibido y procesado correctamente"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



# Se podría hacer que el Servidor guarde el Accuracy, la Loss y el Tráfico de cada ronda en un archivo Excel (.csv) 
# de forma automática para tener los resultados guardados listos para la memoria del TFG