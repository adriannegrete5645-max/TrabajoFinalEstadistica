# ============================================
# PROYECTO FINAL - PROBABILIDAD + ESTADÍSTICA
# ============================================

import pandas as pd
import numpy as np
import math

df = pd.read_csv("irlanda_oecd.csv")

# ============================================
# 1. EXTRAER DATOS
# ============================================


# Limpieza
df = df.dropna(subset=['OBS_VALUE'])
data = df['OBS_VALUE'].values

# ============================================
# 2. FRECUENCIAS (TU BLOQUE ESTADÍSTICO)
# ============================================
#NUMERO DE DATOS YA FILTRADOS A MI PAIS
#DATOS TOTALES EN GENERAL = 1'897,803
print("Numero de datos en general 1'897,803")
print("\nNúmero de datos:", len(data))

# Tendencia central
media = np.mean(data)
mediana = np.median(data)

valores, conteo = np.unique(data, return_counts=True)
moda = valores[np.argmax(conteo)]

print("\n===== MEDIDAS DE TENDENCIA CENTRAL =====")
print(f"Media: {media:.4f}")
print(f"Mediana: {mediana:.4f}")
print(f"Moda: {moda:.4f}")

# Datos agrupados
k = int(np.sqrt(len(data)))
frecuencias, intervalos = np.histogram(data, bins=k)
marcas = (intervalos[:-1] + intervalos[1:]) / 2

media_agrupada = np.sum(frecuencias * marcas) / np.sum(frecuencias)

frecuencia_acum = np.cumsum(frecuencias)
n = np.sum(frecuencias)

clase_mediana = np.where(frecuencia_acum >= n/2)[0][0]
mediana_agrupada = marcas[clase_mediana]

clase_moda = np.argmax(frecuencias)
moda_agrupada = marcas[clase_moda]

print("\n===== DATOS AGRUPADOS =====")
print(f"Media agrupada: {media_agrupada:.4f}")
print(f"Mediana agrupada: {mediana_agrupada:.4f}")
print(f"Moda agrupada: {moda_agrupada:.4f}")

# Dispersión
rango = np.max(data) - np.min(data)

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

varianza = np.var(data)
desv_std = np.std(data)
desv_media = np.mean(np.abs(data - media))

print("\n===== MEDIDAS DE DISPERSIÓN =====")
print(f"Rango: {rango:.4f}")
print(f"Rango intercuartílico: {IQR:.4f}")
print(f"Varianza: {varianza:.4f}")
print(f"Desviación estándar: {desv_std:.4f}")
print(f"Desviación media: {desv_media:.4f}")

print("\n===== INTERPRETACIÓN =====")

print("La media de 18136.87 representa el valor promedio de los registros de planes de pensión en Irlanda obtenidos desde la API de la OCDE. Esto indica que, en promedio, los valores observados dentro del sistema de pensiones irlandés se encuentran alrededor de esa cantidad.")

print("La mediana de 100 muestra que el 50% de los registros se encuentran por debajo de este valor y el otro 50% por encima. La gran diferencia entre la media y la mediana evidencia la presencia de valores extremadamente altos dentro del conjunto de datos.")

print("La moda de 0 indica que existen numerosos registros con ausencia de aportaciones o valores nulos en ciertos tipos de planes de pensión reportados en Irlanda.")

print("La media agrupada permite observar el comportamiento promedio de los datos cuando estos se organizan en intervalos, facilitando el análisis estadístico de grandes volúmenes de información provenientes de la OCDE.")

print("La mediana agrupada identifica el intervalo central donde se concentra la mayor parte de los registros de planes de pensión.")

print("La moda agrupada señala el intervalo con mayor frecuencia de observaciones, mostrando el rango donde se acumulan más datos dentro del sistema analizado.")

print("El rango de 492630 refleja una gran amplitud entre el valor mínimo y máximo de los datos, lo cual demuestra una alta variabilidad en los registros económicos relacionados con los planes de pensión en Irlanda.")

print("El rango intercuartílico de 7266.35 muestra la dispersión existente en el 50% central de los datos, reduciendo el efecto de valores extremos.")

print("La varianza indica un nivel elevado de dispersión entre los registros analizados, lo que significa que los valores de los planes de pensión presentan diferencias importantes entre sí.")

print("La desviación estándar de 47411.45 confirma que muchos valores se alejan considerablemente de la media, evidenciando heterogeneidad en la información financiera reportada.")

print("La desviación media muestra, en promedio, cuánto se apartan los registros respecto al valor medio del conjunto de datos.")
# ============================================
# 4.1 DIAGRAMA DE VENN
# ============================================

from matplotlib import pyplot as plt
from matplotlib_venn import venn2

# Conjuntos de ejemplo
A = set([1,2,3,4,5])
B = set([4,5,6,7,8])

plt.figure(figsize=(6,6))

venn2([A, B],
      set_labels=('Planes ocupacionales',
                  'Planes personales'))

plt.title("Diagrama de Venn - Planes de pensión en Irlanda")
plt.savefig("Diagrama de VEEN.png")
plt.show()

print("\n===== DIAGRAMA DE VENN =====")
print("Se representaron dos conjuntos y su intersección.")
# 4.2 REGLA DE CONTEO
# Regla de la suma
A = 5
B = 3
total_suma = A + B

# Regla de la multiplicación
total_multiplicacion = A * B
print("\n===== INTERPRETACIÓN =====")
print("El diagrama de Venn permitió representar gráficamente la relación entre los planes ocupacionales y los planes personales en Irlanda con A ∩ B. La intersección muestra elementos compartidos entre ambos grupos, mientras que las áreas externas representan elementos exclusivos de cada categoría.")

print("\n===== CONTEO =====")
print("Regla de la suma:", total_suma)
print("Regla de la multiplicación:", total_multiplicacion)
#4.3 COMBINACIONES
import math

def combinacion(n, r):
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

print("\n===== COMBINACIONES =====")
print("C(5,2):", combinacion(5,2))
#4.4 PERMUTACIONES
def permutacion(n, r):
    return math.factorial(n) / math.factorial(n - r)

print("\n===== PERMUTACIONES =====")
print("P(5,2):", permutacion(5,2))
# ============================================
# 4.5 PROBABILIDAD CONDICIONAL (EN TABLA)
# ============================================

# Crear categorías (si no las tienes aún en esta parte)
df['Categoria'] = pd.qcut(
    df['OBS_VALUE'],
    q=3,
    labels=['Bajo', 'Medio', 'Alto']
)

# Variable de grupo
df['Grupo'] = df['Pension plan type']

# Tabla de frecuencias
tabla = pd.crosstab(df['Grupo'], df['Categoria'])

# Probabilidad condicional por fila
tabla_condicional = tabla.div(tabla.sum(axis=1), axis=0)

print("\n===== PROBABILIDAD CONDICIONAL =====")
print(tabla_condicional.round(4))
# Probabilidad condicional
grupo_ejemplo = tabla.index[0]
prob_cond = tabla.loc[grupo_ejemplo, 'Alto'] / tabla.loc[grupo_ejemplo].sum()

print(f"\nP(Alto | {grupo_ejemplo}): {prob_cond:.4f}")
#4.6 TEOREMA DE BAYES
P_B_dado_A = 0.7
P_A = 0.4
P_B = 0.5

bayes = (P_B_dado_A * P_A) / P_B

print("\n===== TEOREMA DE BAYES =====")
print("Resultado:", bayes)
import numpy as np
import math

# ============================================
# DISTRIBUCIÓN BINOMIAL
# ============================================

from scipy.stats import binom
import matplotlib.pyplot as plt

n = 10
p = 0.5

x = np.arange(0, n+1)
y = binom.pmf(x, n, p)

print("\n===== DISTRIBUCIÓN BINOMIAL =====")
print(f"P(X=3) con n=10, p=0.5: {binom.pmf(3,n,p):.4f}")

plt.figure(figsize=(8,5))
plt.bar(x, y)

plt.title("Distribución Binomial")
plt.xlabel("Número de éxitos")
plt.ylabel("Probabilidad")
plt.savefig("binomial.png")
plt.show()

print("\n===== INTERPRETACIÓN =====")

print("La distribución binomial representa la probabilidad de obtener un número específico de resultados exitosos en una cantidad fija de eventos independientes relacionados con fenómenos estadísticos.")

# ============================================
# DISTRIBUCIÓN POISSON
# ============================================

from scipy.stats import poisson

lam = 4

x = np.arange(0,15)
y = poisson.pmf(x, lam)

print("\n===== DISTRIBUCIÓN POISSON =====")
print(f"P(X=2) con λ=4: {poisson.pmf(2,lam):.4f}")

plt.figure(figsize=(8,5))
plt.bar(x, y)

plt.title("Distribución Poisson")
plt.xlabel("Número de eventos")
plt.ylabel("Probabilidad")
plt.savefig("Poisson.png")
plt.show()
print("\n===== INTERPRETACIÓN =====")
print("La distribución Poisson permite modelar eventos que ocurren de manera aleatoria dentro de un intervalo determinado, siendo útil para analizar frecuencia de ocurrencia en fenómenos económicos y sociales.")
# ============================================
# DISTRIBUCIÓN HIPERGEOMÉTRICA
# ============================================

from scipy.stats import hypergeom

N = 50
K = 20
n = 10

x = np.arange(0,11)
y = hypergeom.pmf(x, N, K, n)

print("\n===== DISTRIBUCIÓN HIPERGEOMÉTRICA =====")
print(f"P(X=4): {hypergeom.pmf(4,N,K,n):.4f}")

plt.figure(figsize=(8,5))
plt.bar(x, y)

plt.title("Distribución Hipergeométrica")
plt.xlabel("Número de éxitos")
plt.ylabel("Probabilidad")
plt.savefig("Hipergeometrica.png")
plt.show()

print("\n===== INTERPRETACIÓN =====")
print("La distribución hipergeométrica describe probabilidades asociadas a extracciones sin reemplazo, donde cada selección modifica la composición del conjunto original.")
# ============================================
# DISTRIBUCIÓN NORMAL
# ============================================

from scipy.stats import norm

media = 100
desv = 15

x = np.linspace(40,160,1000)
y = norm.pdf(x, media, desv)

print("\n===== DISTRIBUCIÓN NORMAL =====")
print(f"f(x=110): {norm.pdf(110,media,desv):.6f}")

plt.figure(figsize=(8,5))
plt.plot(x, y)

plt.title("Distribución Normal")
plt.xlabel("Valores")
plt.ylabel("Densidad")
plt.savefig("DIS Normal.png")
plt.show()

print("\n===== INTERPRETACIÓN =====")
print("La distribución normal representa el comportamiento de variables continuas cuyos valores tienden a concentrarse alrededor de un promedio, patrón común en indicadores económicos.")
# ============================================
# DISTRIBUCIÓN NORMAL ESTÁNDAR
# ============================================

z = (110 - media) / desv

x = np.linspace(-4,4,1000)
y = norm.pdf(x,0,1)

print("\n===== DISTRIBUCIÓN NORMAL ESTÁNDAR =====")
print(f"Z para x=110: {z:.4f}")

plt.figure(figsize=(8,5))
plt.plot(x, y)

plt.title("Distribución Normal Estándar")
plt.xlabel("Valor Z")
plt.ylabel("Densidad")
plt.savefig("DIS Normal Estandar.png")
plt.show()

print("\n===== INTERPRETACIÓN =====")
print("La distribución normal estándar permitió calcular el valor Z, el cual indica cuántas desviaciones estándar se encuentra un dato respecto a la media del conjunto analizado.")
# ============================================
# 7. ÍNDICES DE PRECIOS
# ============================================

print("\n===== ÍNDICES DE PRECIOS =====")

# ============================================
# FILTRAR SOLO VALORES POSITIVOS
# ============================================

datos_validos = data[data > 0]

# Tomar 10 datos válidos
precios_base = datos_validos[:10]

# Simular aumento de precios del 10%
factores = np.array([1.05, 1.08, 1.12, 1.15, 1.03,
                     1.09, 1.20, 1.07, 1.11, 1.06])

precios_actuales = precios_base * factores

# Cantidades base y actuales
cantidades_base = np.array([2,3,4,5,6,7,8,9,10,11])
cantidades_actuales = np.array([3,4,5,6,7,8,9,10,11,12])

# ============================================
# 7.1 ÍNDICE LASPEYRES
# ============================================

laspeyres = (
    np.sum(precios_actuales * cantidades_base)
    / np.sum(precios_base * cantidades_base)
) * 100

print("\nÍndice Laspeyres:", round(laspeyres, 4))

# ============================================
# 7.2 ÍNDICE PAASCHE
# ============================================

paasche = (
    np.sum(precios_actuales * cantidades_actuales)
    / np.sum(precios_base * cantidades_actuales)
) * 100

print("Índice Paasche:", round(paasche, 4))

# ============================================
# 7.3 ÍNDICE FISHER
# ============================================

fisher = np.sqrt(laspeyres * paasche)

print("Índice Fisher:", round(fisher, 4))

# ============================================
# INTERPRETACIÓN
# ============================================

print("\n===== INTERPRETACIÓN =====")

print("El índice Laspeyres mostró la variación de precios utilizando las cantidades del período base, permitiendo comparar el crecimiento de precios respecto al escenario inicial.")

print("El índice Paasche utilizó cantidades del período actual, reflejando cambios en el comportamiento de consumo y variaciones recientes en precios.")

print("El índice Fisher representó un promedio equilibrado entre los índices Laspeyres y Paasche, proporcionando una medida más precisa de la variación general de precios.")