import numpy as np

def simular_lanzamientos(n):
    # Generar n lanzamientos de moneda (0: cara, 1: cruz)
    lanzamientos = np.random.randint(0, 2, n)
    return lanzamientos

def calcular_estadisticas(n_lanzamientos):
    # Simular n_lanzamientos
    resultados = simular_lanzamientos(n_lanzamientos)
    
    # Calcular media y desviación estándar
    media = np.mean(resultados)
    desviacion_estandar = np.std(resultados)
    
    return media, desviacion_estandar

# Definir los valores de n
valores_n = [100, 1000, 10000]

# Crear tabla para reportar los valores de media y desviación estándar
print(f"{'N Lanzamientos':<15} {'Media':<15} {'Desv. Estándar':<15}")
print("=" * 45)

for n in valores_n:
    media, desviacion_estandar = calcular_estadisticas(n)
    print(f"{n:<15} {media:<15.5f} {desviacion_estandar:<15.5f}")


