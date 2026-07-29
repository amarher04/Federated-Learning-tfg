import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Configuración de Rutas y Carpetas
# ==========================================
ARCHIVO_CSV = 'resultados_2_3_F16ZIP.csv'

# Ruta de la nueva carpeta (sube un nivel desde 'server' y crea 'graphs')
CARPETA_GRAFICAS = os.path.join('..', 'graphs')

# Creamos la carpeta automáticamente si no existe
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# Verificamos si el archivo CSV existe antes de continuar
if not os.path.exists(ARCHIVO_CSV):
    print(f"Error: No se encontró el archivo '{ARCHIVO_CSV}'.")
    exit()

# ==========================================
# 2. Cargar y preparar los datos
# ==========================================
df = pd.read_csv(ARCHIVO_CSV)
df.columns = df.columns.str.strip() # Limpiar posibles espacios en cabeceras

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# GRÁFICA 1: Evolución del Accuracy
# ==========================================
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='Ronda', y='Accuracy', hue='Experimento_ID', marker='o', linewidth=2)

plt.title('Evolución del Accuracy Global del Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Ronda de Comunicación', fontsize=12)
plt.ylabel('Precisión (Accuracy)', fontsize=12)
plt.legend(title='Experimento', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.ylim(0, 1)
plt.ylim(0.88, 0.95) # CAMBIAR EJE Y
plt.tight_layout()

ruta_acc = os.path.join(CARPETA_GRAFICAS, 'grafica_accuracy_2_3_F16ZIP.png')
plt.savefig(ruta_acc, dpi=300)
print(f"Gráfica guardada en: {ruta_acc}")

# ==========================================
# GRÁFICA 2: Evolución del Loss (Error)
# ==========================================
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='Ronda', y='Loss', hue='Experimento_ID', marker='s', linewidth=2)

plt.title('Disminución del Error (Loss) durante el Entrenamiento', fontsize=14, fontweight='bold')
plt.xlabel('Ronda de Comunicación', fontsize=12)
plt.ylabel('Pérdida (Loss)', fontsize=12)
plt.legend(title='Experimento', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

ruta_loss = os.path.join(CARPETA_GRAFICAS, 'grafica_loss_2_3_F16ZIP.png')
plt.savefig(ruta_loss, dpi=300)
print(f"Gráfica guardada en: {ruta_loss}")

# ==========================================
# GRÁFICA 3: Tráfico de Red Acumulado (KB)
# ==========================================
# Cambiado a lineplot usando 'Trafico_Acumulado_KB' para ver la carga de red a lo largo del tiempo
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='Ronda', y='Trafico_Acumulado_KB', hue='Experimento_ID', marker='^', linewidth=2)

plt.title('Tráfico de Red Acumulado a lo largo del Entrenamiento', fontsize=14, fontweight='bold')
plt.xlabel('Ronda de Comunicación', fontsize=12)
plt.ylabel('Tráfico Acumulado (KB)', fontsize=12)
plt.legend(title='Experimento', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

ruta_trafico_acum = os.path.join(CARPETA_GRAFICAS, 'grafica_trafico_acumulado_2_3_F16ZIP.png')
plt.savefig(ruta_trafico_acum, dpi=300)
print(f"Gráfica guardada en: {ruta_trafico_acum}")

# ==========================================
# GRÁFICA 4: Tiempo Empleado por Ronda (Segundos)
# ==========================================
# Muestra el coste computacional de cada iteración del experimento
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='Ronda', y='Tiempo_Ronda_Seg', hue='Experimento_ID', marker='d', linewidth=2)

plt.title('Tiempo de Ejecución por Ronda', fontsize=14, fontweight='bold')
plt.xlabel('Ronda de Comunicación', fontsize=12)
plt.ylabel('Tiempo (Segundos)', fontsize=12)
plt.legend(title='Experimento', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(5, 6.3) # CAMBIAR EJE Y
plt.tight_layout()

ruta_tiempo = os.path.join(CARPETA_GRAFICAS, 'grafica_tiempo_ronda_2_3_F16ZIP.png')
plt.savefig(ruta_tiempo, dpi=300)
print(f"Gráfica guardada en: {ruta_tiempo}")

print(f"\n¡Todas las gráficas se han generado con éxito en la carpeta '{CARPETA_GRAFICAS}'!")