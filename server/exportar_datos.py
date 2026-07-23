import tensorflow as tf
import numpy as np

print("Descargando MNIST...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Cogemos 100 muestras (o las que quieras, pero deben coincidir con tu TFLite)
x_android = x_train[:100]
y_android = y_train[:100]

# Normalizamos y aplanamos las imágenes (de 28x28 a 784)
x_android = (x_android / 255.0).astype(np.float32).reshape(100, 784)
y_android = y_android.astype(np.int32)

# Guardamos en formato binario puro (Little Endian)
x_android.tofile("mnist_x.bin")
y_android.tofile("mnist_y.bin")

print("¡Archivos mnist_x.bin y mnist_y.bin generados!")
