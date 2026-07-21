import tensorflow as tf

class EntrenadorFederado(tf.Module):
    def __init__(self):
        super(EntrenadorFederado, self).__init__()
        # Definición del modelo (igual que el modelo global del servidor)
        # Aplanamos a 784 pixeles (28x28 de MNIST)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')  
        ])
        self.model(tf.zeros((1, 784))) # Forzamos creacion interna de variables con dato vacio
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() # Funcion de perdida
        
    # SIGNATURE 1: Entrenar
    @tf.function(input_signature=[
        tf.TensorSpec([100, 784], tf.float32, name="x"),  # x (Imágenes aplanadas, batch fijo de 100)
        tf.TensorSpec([100], tf.int32, name="y")          # y (Etiquetas reales, batch fijo de 100)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            predicciones = self.model(x)
            loss = self.loss_fn(y, predicciones)
        # Calcula como modificar los pesos y los aplica
        gradientes = tape.gradient(loss, self.model.trainable_variables)
        lr = tf.constant(0.01, dtype=tf.float32)
        for var, grad in zip(self.model.trainable_variables, gradientes):
            var.assign_sub(lr * grad)  # var = var - lr * grad
            
        return {"loss": tf.reshape(loss, [1])}  # Devolvemos el loss como un tensor de forma [1] para evitar problemas de mapeo en Kotlin
    
    # SIGNATURE 2: Inicializar memoria
    @tf.function(input_signature=[
        tf.TensorSpec([784, 128], tf.float32, name="p0"),
        tf.TensorSpec([128], tf.float32, name="s0"),
        tf.TensorSpec([128, 10], tf.float32, name="p1"),
        tf.TensorSpec([10], tf.float32, name="s1"),
    ])
    def restore(self, p0, s0, p1, s1):
        self.model.trainable_variables[0].assign(p0)
        self.model.trainable_variables[1].assign(s0)
        self.model.trainable_variables[2].assign(p1)
        self.model.trainable_variables[3].assign(s1)
        # Devolvemos un 1.0 para que el mapa de salida en Kotlin no esté vacío
        return {"status": tf.constant([1.0], dtype=tf.float32)}
    
    # SIGNATURE 3: Extraer pesos
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32, name="dummy")])
    def save(self, dummy): # antes de la modificacion era 'x' en vez de "dummy"
        return {
            "pesos_0": self.model.trainable_variables[0] + dummy,
            "sesgos_0": self.model.trainable_variables[1] + dummy,
            "pesos_1": self.model.trainable_variables[2] + dummy,
            "sesgos_1": self.model.trainable_variables[3] + dummy,
        }

print("Construyendo el modelo...\n")
modulo = EntrenadorFederado()
    
# --- GUARDADO Y CONVERSION A TENSORFLOW LITE ---
# Guardamos el modelo
print("Guardando y convirtiendo a TFLite...\n")
tf.saved_model.save(
    modulo,
    "modelo_temporal",
    signatures={
        'train': modulo.train.get_concrete_function(),
        'restore': modulo.restore.get_concrete_function(),
        'save': modulo.save.get_concrete_function(),
    }
)

# Convertimos el modelo a TFLite
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
    
print("¡EXITO! Archivo 'modelo_entrenable.tflite' generado correctamente.\n")
