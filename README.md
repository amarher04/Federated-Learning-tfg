# Federated Learning System with Android Clients 📱🤖

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()

## 📌 Descripción General
Este proyecto implementa un sistema de Aprendizaje Federado (Federated Learning) donde múltiples clientes Android colaboran para entrenar un modelo global de Machine Learning de forma descentralizada. 

El objetivo principal es permitir que los dispositivos móviles entrenen el modelo usando sus datos locales, enviando únicamente las actualizaciones de los pesos (y no los datos privados) al servidor central, preservando así la privacidad del usuario.

## 🏗 Arquitectura del Sistema
El proyecto se divide en dos componentes principales:
*   **Servidor Central (Python):** Se encarga de orquestar el entrenamiento, recibir los pesos de los clientes Android, agregar las actualizaciones (ej. usando algoritmos como FedAvg) y distribuir el nuevo modelo global actualizado.
*   **Clientes (Android):** Aplicaciones nativas que descargan el modelo global, lo entrenan con los datos locales del dispositivo y devuelven los gradientes actualizados al servidor a través de la red.

## 📂 Estructura del Repositorio
```text
Federated-Learning-System-with-Android-Clients/
│
├── android-client/          # Código fuente de la app nativa Android (Kotlin/Java)
│   ├── app/                 # Módulo principal de la aplicación
│   └── build.gradle         # Configuración de Gradle para Android
│
├── clientes-python/         # Código para ejecutar clientes desde python
│   ├── clientes.py
│
├── server/                  # Código del servidor central
│   ├── servidor_fl.py              # Script principal para levantar el servidor
│   └── requirements.txt     # Dependencias necesarias de Python
│
├── .gitignore               # Archivos excluidos del control de versiones (Git)
└── README.md                # Documentación principal del proyecto
```

## 🚀 Requisitos Previos

### Para el Servidor (Python)
*   Python 3.8 o superior.
*   Dependencias listadas en `server/requirements.txt`.

### Para el Cliente (Android)
*   **Android Studio** (Flamingo o superior recomendado).
*   SDK de Android actualizado (API 24+).
*   Dispositivo físico o emulador Android conectado a la misma red local que el servidor.

## ⚙️ Instalación y Configuración Paso a Paso

### 1. Iniciar el Servidor Central
Abre una terminal, clona el repositorio y navega a la carpeta del servidor:
```bash
git clone https://github.com/amarher04/Federated-Learning-System-with-Android-Clients.git
cd Federated-Learning-System-with-Android-Clients/server

# 1. Crear un entorno virtual (opcional pero muy recomendado)
python -m venv venv

# 2. Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Ejecutar el servidor de Federated Learning
python main.py
```
*El servidor se iniciará y quedará a la espera de que los clientes Android se conecten.*

### 2. Configurar y Ejecutar el Cliente Android
1. Abre **Android Studio**.
2. Selecciona `File > Open...` y abre la carpeta `android-client` de este repositorio.
3. Espera unos instantes a que Gradle sincronice y construya el proyecto inicial.
4. **⚠️ Configuración de Red Importante:** 
   Ve al código de red de la app (generalmente donde configuras tu cliente gRPC, Retrofit o Sockets) y cambia la dirección IP a la **IP local de la máquina donde se ejecuta el servidor Python** (ej. `192.168.1.50`). 
   *Nota: No uses `localhost` o `127.0.0.1` en el código de Android, ya que eso apuntaría al propio sistema interno del teléfono/emulador y no a tu ordenador.*
5. Conecta un dispositivo físico por USB/WiFi o inicia un Emulador de Android.
6. Haz clic en el botón verde de **Run (▶)** en la barra superior de Android Studio para instalar y abrir la app.

## 📊 Sobre los Datos de Entrenamiento
Para mantener la privacidad inherente del Federated Learning y evitar engordar el repositorio, **no se incluyen datasets reales ni modelos pre-entrenados pesados**. 
*   **Servidor:** Si tu código de servidor requiere algún dataset de validación global, descárgalo manualmente y colócalo en una carpeta ignorada por git (como `server/data/`).
*   **Cliente:** Los clientes Android generan datos sintéticos o leen datos de prueba directamente desde el almacenamiento local del dispositivo durante las pruebas.

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas para mejorar el sistema!
1. Haz un Fork del repositorio.
2. Crea una rama para tu nueva característica (`git checkout -b feature/MejoraRed`).
3. Haz commit de tus cambios (`git commit -m 'Añade soporte para TLS'`).
4. Haz push a tu rama (`git push origin feature/MejoraRed`).
5. Abre un Pull Request en este repositorio.

## 📄 Licencia
Este proyecto se distribuye bajo la Licencia MIT. Siéntete libre de utilizar, modificar y distribuir el código como necesites.
