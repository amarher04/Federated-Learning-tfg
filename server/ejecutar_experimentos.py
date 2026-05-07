import requests
import time

URL_SERVIDOR = "http://127.0.0.1:8000"
RONDAS_POR_EXPERIMENTO = 5

lista_epochs_a_probar = [1, 3, 5, 10]

print("Iniciando Bateria de Experimentos Automatizados...\n")

for epochs in lista_epochs_a_probar:
    print(f"\n >>> MANDANDO ORDEN: Empezar nuevo experimento con {epochs} Epochs locales <<< \n")
    
    # Dar la orden al servidor para configurar un nuevo experimento con X epochs locales
    try:
        respuesta = requests.post(
            f"{URL_SERVIDOR}/admin/configurar",
            json={"epochs_locales": epochs}
        )
        print(f"Respuesta del servidor: {respuesta.json()}")
    except Exception as e:
        print(f"Error al conectar con el servidor: {e}")
        break

    # Esperar hasta que los moviles terminen las rondas
    while True:
        # Preguntamos al servidor su ronda actual
        estado = requests.get(f"{URL_SERVIDOR}/config").json()
        ronda_actual = estado["ronda_actual"]
        
        if ronda_actual > RONDAS_POR_EXPERIMENTO:
            print(f"\n >>> Experimento con {epochs} epochs terminado con exito. Pasando al siguiente... <<< \n")
            break
        
        print(f"Monitor: Esperando... El servidor esta en la ronda {ronda_actual} de {RONDAS_POR_EXPERIMENTO}.")
        time.sleep(5)  # Esperar 5 segundos antes de preguntar de nuevo
        
    print("\n TODOS LOS EXPERIMENTOS HAN FINALIZADO. Revisa el archivo CSV con los resultados. \n")
        
        