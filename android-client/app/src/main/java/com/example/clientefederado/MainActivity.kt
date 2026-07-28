package com.example.clientefederado

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
// Añadimos libreria TensorFlow Lite
import org.tensorflow.lite.Interpreter

import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PantallaPrincipal(context = this)
        }
    }
}

// Funcion auxiliar para leer el archivo .tflite de la carpeta assets
fun cargarModeloDesdeAssets(context: Context, nombreArchivo: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(nombreArchivo)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
}

// FUNCIÓN PUENTE: Convierte la memoria RAM a Binario Puro
fun guardarPesosBinario(archivo: File, pesos0: Array<FloatArray>, sesgos0: FloatArray, pesos1: Array<FloatArray>, sesgos1: FloatArray) {
    // Calculamos el total de números flotantes (784x128 + 128 + 128x10 + 10 = 101770)
    val totalFloats = (784 * 128) + 128 + (128 * 10) + 10
    val buffer = ByteBuffer.allocate(totalFloats * 4) // 4 bytes por cada Float
    buffer.order(ByteOrder.LITTLE_ENDIAN) // Formato Little Endian para que Python lo lea correctamente

    for (fila in pesos0) for (v in fila) buffer.putFloat(v)
    for (v in sesgos0) buffer.putFloat(v)
    for (fila in pesos1) for (v in fila) buffer.putFloat(v)
    for (v in sesgos1) buffer.putFloat(v)

    val fos = FileOutputStream(archivo)
    fos.write(buffer.array())
    fos.close()
}

@Composable
fun PantallaPrincipal(context: Context) {
    // Variable guarda texto en pantalla. Si cambia, pantalla se actualiza
    var textoPantalla by remember { mutableStateOf("Estado: Esperando...") }
    // Lanzar tareas en segundo plano
    val corrutina = rememberCoroutineScope()
    // Variable para saber si el bucle autonomo ya esta corriendo
    var estaEjecutandose by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Aprendizaje Federado", modifier = Modifier.padding(bottom = 32.dp))

        Text(text = textoPantalla, modifier = Modifier.padding(16.dp))

        Button(onClick = {
            if (!estaEjecutandose) {
                estaEjecutandose = true
                textoPantalla = "Iniciando Cliente autonomo..."

                // Lanzar tarea en segundo plano
                corrutina.launch {
                    // Funcion que permite actualizar el texto de la pantalla desde el bucle
                    ejecutarBucleAutonomo(context) {
                        mensaje -> textoPantalla = mensaje
                    }
                }
            }
        }) {
            Text(if (estaEjecutandose) "Ejecutando en 2º plano..." else "Iniciar Cliente Autonomo")
        }
    }
}

suspend fun ejecutarBucleAutonomo(context: Context, actualizarPantalla: (String) -> Unit) {
    withContext(Dispatchers.IO) {
        val cliente = OkHttpClient()
        var rondaCompletada = 0

        var experimentoActual = ""

        while (true) {
            try {
                // 1. PREGUNTAR AL SERVIDOR (/config)
                val reqConfig = Request.Builder().url("http://10.0.2.2:8000/config").build()
                val respConfig = cliente.newCall(reqConfig).execute()

                if (respConfig.isSuccessful) {
                    // Extraemos ronda actual usando libreria JSON nativa de Android
                    val json = JSONObject(respConfig.body?.string() ?: "{}")
                    val rondaServidor = json.getInt("ronda_actual")
                    val epochsLocales = json.optInt("epochs_locales", 1)

                    // Leemos el ID del experimento del Servidor
                    val experimentoServidor = json.optString("experimento_id", "")

                    // Leemos la variable que indica si hay un experimento activo
                    val rondasObjetivo = json.optInt("rondas_objetivo", -1)

                    // DROPOUT SIMULATION
                    // El servidor nos enviará un booleano (ej. "participa_Android_1": false). Si no lo envía, asumimos true.
                    val miID = "Android_1" // OJO: Cambiar esto en cada móvil si se usan varios
                    val meTocaParticipar = json.optBoolean("participa_$miID", true)

                    if (rondasObjetivo == 0) {
                        actualizarPantalla("Experimento finalizado. Cliente en estado de reposo...")
                        // El código no entra a descargar nada. Al terminar el if,
                        // esperará 3 segundos y volverá a preguntar silenciosamente.
                    } else if (!meTocaParticipar) {
                        // Si el servidor ha simulado que hemos perdido la red, no entrenamos esta ronda
                        actualizarPantalla("Pérdida de red simulada (Dropout). Saltando ronda $rondaServidor...")
                        delay(5000)
                    } else {
                        // Logica de reseteo si el experimento cambia
                        if (experimentoServidor != experimentoActual) {
                            experimentoActual = experimentoServidor
                            rondaCompletada = 0 // Borramos la memoria de rondas del experimento anterior
                            actualizarPantalla("Nuevo experimento detectado. Preparando cliente...")
                            delay(1500)
                        }

                        if (rondaServidor > rondaCompletada) {
                            if (rondaServidor <= rondasObjetivo) {

                                actualizarPantalla("¡Ronda $rondaServidor de $rondasObjetivo! Descargando modelo global...")

                                // 2. DESCARGAR MODELO (/model/download)
                                val reqDownload = Request.Builder().url("http://10.0.2.2:8000/model/download/bin").build()
                                val respDownload = cliente.newCall(reqDownload).execute()

                                // Guardamos el archivo binario en el almacenamiento interno privado de la app
                                val archivoLocal = File(context.filesDir, "modelo_global.bin")
                                val fos = FileOutputStream(archivoLocal)
                                fos.write(respDownload.body?.bytes() ?: byteArrayOf())
                                fos.close()

                                actualizarPantalla("Modelo guardado. Iniciando LiteRT...")

                                // 3. ENTRENAMIENTO REAL EN ON-DEVICE CON TFLite

                                // Cargamos el modelo TFLite de la carpeta assets
                                val tfliteBuffer = cargarModeloDesdeAssets(context, "modelo_entrenable.tflite")
                                val interpreter = Interpreter(tfliteBuffer)

                                // -----------------------------------------------------------------------
                                // INICIALIZAR MEMORIA (Firma 'restore')
                                actualizarPantalla("Asignando memoria interna...")
                                val bytesModelo = archivoLocal.readBytes()
                                val bufferModelo = ByteBuffer.wrap(bytesModelo).order(ByteOrder.LITTLE_ENDIAN)

                                val pesos0 = Array(784) { FloatArray(128) { bufferModelo.float } }
                                val sesgos0 = FloatArray(128) { bufferModelo.float }
                                val pesos1 = Array(128) { FloatArray(10) { bufferModelo.float } }
                                val sesgos1 = FloatArray(10) { bufferModelo.float }

                                val entradasRestore = mapOf(
                                    "p0" to pesos0,
                                    "s0" to sesgos0,
                                    "p1" to pesos1,
                                    "s1" to sesgos1
                                )
                                val salidasRestore = mapOf("status" to FloatArray(1)) // Recibimos el 1.0 de Python
                                interpreter.runSignature(entradasRestore, salidasRestore, "restore")
                                // -----------------------------------------------------------------------

                                // CARGAR DATOS REALES DE MNIST DESDE ASSETS
                                actualizarPantalla("Cargando dataset local MNIST...")
                                val isX = context.assets.open("mnist_x.bin")
                                val bufferX = ByteBuffer.wrap(isX.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
                                val datosX = Array(100) { FloatArray(784) { bufferX.float } }
                                isX.close()

                                val isY = context.assets.open("mnist_y.bin")
                                val bufferY = ByteBuffer.wrap(isY.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
                                val datosY = IntArray(100) { bufferY.int }
                                isY.close()
                                // -----------------------------------------------------------------------

                                // Preparamos las variables de entrada/salida para la signature "train"
                                val entradasEntrenamiento = mapOf("x" to datosX, "y" to datosY)
                                val valorLoss = FloatArray(1)
                                val salidasEntrenamiento = mapOf("loss" to valorLoss)

                                // Iniciar cronometro
                                val tiempoInicioEntrenamiento = System.currentTimeMillis()

                                // Bucle de entrenamiento usando las epochs del servidor
                                for (epoch in 1..epochsLocales) {
                                    interpreter.runSignature(entradasEntrenamiento, salidasEntrenamiento, "train")
                                    actualizarPantalla("Epoch $epoch | Error (Loss): ${String.format("%.4f", valorLoss[0])}")
                                    delay(500)
                                }

                                // Parar cronómetro y calcular segundos
                                val tiempoFinEntrenamiento = System.currentTimeMillis()
                                val tiempoEntrenamientoLocal = (tiempoFinEntrenamiento - tiempoInicioEntrenamiento) / 1000.0 // en segundos

                                // Extraemos los nuevos pesos
                                actualizarPantalla("Extrayendo pesos actualizados...")
                                val entradaDummy = mapOf("dummy" to floatArrayOf(0.0f))

                                val salidasGuardar = mapOf(
                                    "pesos_0" to Array(784) { FloatArray(128) },
                                    "sesgos_0" to FloatArray(128),
                                    "pesos_1" to Array(128) { FloatArray(10) },
                                    "sesgos_1" to FloatArray(10)
                                )

                                interpreter.runSignature(
                                    entradaDummy,
                                    salidasGuardar,
                                    "save"
                                ) // Pasamos el dummy en vez del emptyMap
                                interpreter.close() // Cerramos para no saturar la RAM del móvil

                                // 4. SUBIR MODELO (/model/upload)
                                actualizarPantalla("Entrenamiento Finalizado. Enviando resultados al servidor...")

                                // Creamos el archivo puente
                                val archivoPesos = File(context.filesDir, "pesos_entrenados.bin")
                                guardarPesosBinario(
                                    archivoPesos,
                                    salidasGuardar["pesos_0"] as Array<FloatArray>,
                                    salidasGuardar["sesgos_0"] as FloatArray,
                                    salidasGuardar["pesos_1"] as Array<FloatArray>,
                                    salidasGuardar["sesgos_1"] as FloatArray
                                )

                                val fileBody = archivoPesos.asRequestBody("application/octet-stream".toMediaTypeOrNull())

                                // Preparamos el Formulario Multiparte con los metadatos exigidos por el servidor
                                val multipartBody = MultipartBody.Builder()
                                    .setType(MultipartBody.FORM)
                                    .addFormDataPart("cliente_id", "Android_1")
                                    .addFormDataPart("num_muestras", "100") // Las 100 del batch
                                    .addFormDataPart("ronda_cliente", rondaServidor.toString())
                                    .addFormDataPart("tiempo_entrenamiento", tiempoEntrenamientoLocal.toString())
                                    .addFormDataPart("file", "pesos_entrenados.bin", fileBody) // Enviamos el .bin
                                    .build()

                                val reqUpload = Request.Builder()
                                    .url("http://10.0.2.2:8000/model/upload")
                                    .post(multipartBody)
                                    .build()

                                val respUpload = cliente.newCall(reqUpload).execute()

                                if (respUpload.isSuccessful) {
                                    rondaCompletada = rondaServidor
                                    delay(2000)

                                    if (rondaServidor == rondasObjetivo) {
                                        actualizarPantalla("¡Experimento completado con éxito! (Total: $rondasObjetivo rondas)")
                                    } else {
                                        actualizarPantalla("Ronda $rondaServidor/$rondasObjetivo completada. Esperando la siguiente...")
                                    }
                                } else {
                                    // Si falla la subida, no actualizamos rondaCompletada para poder reintentar
                                    actualizarPantalla("Error al enviar modelo al servidor. Código: ${respUpload.code}")
                                }

                            } // Si el servidor envía una ronda mayor al objetivo (ej. 6/5)
                            else {
                                // Aseguramos que la pantalla siga mostrando el mensaje de éxito
                                // y evitamos que haga falsos entrenamientos.
                                actualizarPantalla("¡Experimento completado con éxito! (Total: $rondasObjetivo rondas)")
                                rondaCompletada = rondaServidor // Lo igualamos para que no vuelva a entrar al IF
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                actualizarPantalla("Buscando al servidor... (${e.message})")
            }

            // Espera de 3seg antes de volver a preguntar para no saturar
            delay(3000)
        }
    }
}