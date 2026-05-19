import tensorflow as tf
import numpy as np
import os

class EntrenadorFederado(tf.Module):
    def __init__(self):
        super(EntrenadorFederado, self).__init__()
        # Definición del modelo (igual que el modelo global del servidor)
        # Aplanamos a 784 pixeles (28x28 de MNIST)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')  
        ])
        # Optimizador y función de pérdida
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        
    # --- FUNCIONES (SIGNATURES) PARA ANDROID ---
        
    # SIGNATURE 1: Entrenar
    @tf.function(input_signature=[
        tf.TensorSpec([None, 784], tf.float32),  # x (Imágenes aplanadas)
        tf.TensorSpec([None], tf.int32),          # y (Etiquetas reales)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            predicciones = self.model(x)
            loss = self.loss_fn(y, predicciones)
        # Calcula como modificar los pesos y los aplica
        gradientes = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradientes, self.model.trainable_variables))
        return {"loss": loss}
    
    # SIGNATURE 2: Predecir
    @tf.function(input_signature=[
        tf.TensorSpec([None, 784], tf.float32),
    ])
    def infer(self, x):
        return {"predicciones": self.model(x)}
    
    # SIGNATURE 3: Extraer pesos
    @tf.function(input_signature=[])
    def save(self):
        return {
            "pesos_0": self.model.trainable_variables[0],  # Pesos de la capa oculta
            "sesgos_0": self.model.trainable_variables[1],  # Sesgos de la capa oculta
            "pesos_1": self.model.trainable_variables[2],  # Pesos de la capa de salida
            "sesgos_1": self.model.trainable_variables[3],  # Sesgos de la capa de salida
        }
    
    # SIGNATURE 4: Inyectar pesos
    @tf.function(input_signature=[
        tf.TensorSpec([784, 128], tf.float32),
        tf.TensorSpec([128], tf.float32),
        tf.TensorSpec([128, 10], tf.float32),
        tf.TensorSpec([10], tf.float32),
    ])
    def restore(self, p0, s0, p1, s1):
        self.model.trainable_variables[0].assign(p0)
        self.model.trainable_variables[1].assign(s0)
        self.model.trainable_variables[2].assign(p1)
        self.model.trainable_variables[3].assign(s1)
        return {}
    

# --- CONVERSION A TENSORFLOW LITE ---
print("Construyendo el modelo...\n")
modulo = EntrenadorFederado()

# Guardamos el modelo temporalmente con sus 4 signatures
tf.saved_model.save(
    modulo,
    "modelo_temporal",
    signatures={
        'train': modulo.train.get_concrete_function(),
        'infer': modulo.infer.get_concrete_function(),
        'save': modulo.save.get_concrete_function(),
        'restore': modulo.restore.get_concrete_function(),
    }
)

# Convertimos el modelo a TFLite
print("Convirtiendo a formato movil (TFLite)...\n")
converter = tf.lite.TFLiteConverter.from_saved_model("modelo_temporal")
# Habilitamos operaciones de entrenamiento en moviles
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones estándar de TensorFlow Lite
    tf.lite.OpsSet.SELECT_TF_OPS     # Operaciones no compatibles con TFLite
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

# Guardamos el archivo TFLite final
with open("modelo_entrenable.tflite", "wb") as f:
    f.write(tflite_model)
    
print("¡EXITO! Archivo 'modelo_entrenable.tflite' generado correctamente.")
