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

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PantallaPrincipal(context = this)
        }
    }
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

        while (true) {
            try {
                // 1. Preguntar al servidor (/config)
                val reqConfig = Request.Builder().url("http://10.0.2.2:8000/config").build()
                val respConfig = cliente.newCall(reqConfig).execute()

                if (respConfig.isSuccessful) {
                    // Extraemos ronda actual usando libreria JSON nativa de Android
                    val json = JSONObject(respConfig.body?.string() ?: "{}")
                    val rondaServidor = json.getInt("ronda_actual")

                    if (rondaServidor > rondaCompletada) {
                        actualizarPantalla("¡Ronda $rondaServidor! Descargando modelo...")

                        // 2. Descargar modelo (/model/download)
                        val reqDownload = Request.Builder().url("http://10.0.2.2:8000/model/download").build()
                        val respDownload = cliente.newCall(reqDownload).execute()

                        // Guardamos el archivo binario en el almacenamiento interno privado de la app
                        val archivoLocal = File(context.filesDir, "modelo_android.npz")
                        val fos = FileOutputStream(archivoLocal)
                        fos.write(respDownload.body?.bytes() ?: byteArrayOf())
                        fos.close()

                        actualizarPantalla("Modelo guardado. Simulando entrenamiento...")

                        // Simulamos que entrenamos esperando 3seg
                        delay(3000)

                        // 3. Subir modelo (/model/upload)
                        actualizarPantalla("Enviando resultados al servidor...")

                        val fileBody = archivoLocal.asRequestBody("application/octet-stream".toMediaTypeOrNull())

                        // Preparamos el Formulario Multiparte con los metadatos exigidos por el servidor
                        val multipartBody = MultipartBody.Builder()
                            .setType(MultipartBody.FORM)
                            .addFormDataPart("cliente_id", "Android_1")
                            .addFormDataPart("num_muestras", "1500") // Muestras falsas
                            .addFormDataPart("ronda_cliente", rondaServidor.toString())
                            .addFormDataPart("file", "pesos_android.npz", fileBody)
                            .build()

                        val reqUpload = Request.Builder()
                            .url("http://10.0.2.2:8000/model/upload")
                            .post(multipartBody)
                            .build()

                        val respUpload = cliente.newCall(reqUpload).execute()
                        if (respUpload.isSuccessful) {
                            rondaCompletada = rondaServidor
                            actualizarPantalla("Ronda $rondaServidor completada. Esperando la siguiente...")
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