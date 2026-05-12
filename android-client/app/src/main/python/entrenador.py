import os
import numpy as np
import tensorflow as tf

def entrenar_modelo_local(ruta_entrada, ruta_salida, epochs):
    try:
        # 1. Cargar datos limitados (1000 imágenes) para no saturar la RAM del móvil
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train[:1000] / 255.0
        y_train = y_train[:1000]

        # 2. Reconstruir la estructura del modelo
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 3. Cargar los pesos descargados del servidor
        with np.load(ruta_entrada) as data:
            pesos = [data[f] for f in data.files]
            model.set_weights(pesos)

        # 4. Entrenar
        model.fit(x_train, y_train, epochs=int(epochs), verbose=0)

        # 5. Sobreescribir el archivo con los nuevos pesos
        nuevos_pesos = model.get_weights()
        np.savez(ruta_salida, *nuevos_pesos)

        return "OK"
    except Exception as e:
        return f"ERROR: {str(e)}"