from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import os
import numpy as np

app = FastAPI(title="Servidor Aprendizaje Federado")

# VARIABLES DE ESTADO DEL SERVIDOR
estado_servidor = {
    "ronda_actual": 1,       # Ronda actual del entrenamiento federado
    "epochs_locales": 3,     # Numero de epochs para el entrenamiento local (moviles)
    "clientes_esperados": 2, # Clientes que se necesita para hacer algoritmo FedAvg
    "pesos_recibidos": 0     # Contador de cuantos han hecho sus deberes
}

# Carpeta para guardar los archivos temporales
os.makedirs("modelos_clientes", exist_ok=True)

# Ahora mismo simulamos que el servidor ya tiene un modelo global iniciado vacío guardado
# En el futuro, inicializaremos el MLP real y guardaremos sus pesos
np.savez("modelo_global.npz", dummy=np.array([0]))


# ENDPOINT 1: Configuración de la ronda (GET)
@app.get("/config")
def obtener_configuracion():
    return estado_servidor


# ENDPOINT 2: Descargar Modelo (GET)
@app.get("/model/download")
def descargar_modelo():
    ruta_archivo = "modelo_global.npz"
    if os.path.exists(ruta_archivo):
        return FileResponse(ruta_archivo, media_type='application/octet-stream', filename="modelo_global.npz")
    else:
        return {"error": "Modelo global no encontrado"}


# ENDPOINT 3: Subir Pesos actualizados (POST)
@app.post("/model/upload")
async def subir_pesos(file: UploadFile = File(...)):
    # Lee y y guarda el archivo que llega por la red
    ruta_guardado = f"modelos_clientes/{file.filename}"
    with open(ruta_guardado, "wb") as buffer:
        contenido = await file.read()
        buffer.write(contenido)
    
    # Incrementa el contador de pesos recibidos
    estado_servidor["pesos_recibidos"] += 1
    
    # Si ya se han recibido los pesos de todos los clientes esperados, se puede hacer FedAvg
    if estado_servidor["pesos_recibidos"] >= estado_servidor["clientes_esperados"]:
        print(f"\n Ronda {estado_servidor['ronda_actual']} completada. Realizando FedAvg... \n")
        
        # 1. Leer todos los archivos .npz de la carpeta
        rutas_archivos = [os.path.join("modelos_clientes", f) for f in os.listdir("modelos_clientes") if f.endswith(".npz")]
        
        todos_los_pesos = []
        for ruta in rutas_archivos:
            with np.load(ruta) as datos_npz:
                pesos_cliente = [datos_npz[f] for f in datos_npz.files]
                todos_los_pesos.append(pesos_cliente)
        
        # 2. Hacer FedAvg capa por capa
        nuevos_pesos_globales = []
        num_capas = len(todos_los_pesos[0])
        num_clientes = len(todos_los_pesos)
        
        for i in range(num_capas):
            # Sumamos los pesos de la capa i de todos los clientes y dividimos por el total
            suma_capas = sum(cliente[i] for cliente in todos_los_pesos)
            media_capa = suma_capas / float(num_clientes)
            nuevos_pesos_globales.append(media_capa)
        
        # 3. Guardar el nuevo modelo global actualizado sobreescribiendo el anterior
        np.savez("modelo_global.npz", *nuevos_pesos_globales)
        print("   -> Nuevo modelo_global.npz generado con éxito.\n")
        
        # 4. Limpiar carpeta de modelos de clientes para la siguiente ronda
        for ruta in rutas_archivos:
            os.remove(ruta)
        
        # 5. Reseteamos contadores para siguiente ronda
        estado_servidor["pesos_recibidos"] = 0
        estado_servidor["ronda_actual"] += 1
        
    return {"mensaje": "Pesos recibidos con éxito", "Ronda": estado_servidor["ronda_actual"]}
    

# Punto de entrada para arrancar el servidor
if __name__ == "__main__":
    print("Iniciando Servidor de Aprendizaje Federado...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
