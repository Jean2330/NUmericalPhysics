# Jean Marroquin 

import numpy as np

def metodo_chi_2(aleatorios, num_intervalos):
    # Calcular las frecuencias observadas
    frec_obs, limites = np.histogram(aleatorios, bins=num_intervalos)

    # Calcular la frecuencia esperada para una distribución uniforme
    frec_esp = len(aleatorios) // num_intervalos

    # Calcular la estadística de Chi-cuadrada
    V= np.sum(((frec_obs - frec_esp) ** 2) / frec_esp)
    
    # Calcular el valor crítico de Chi-cuadrada para un nivel de significancia del 0.05 (9 grados de libertad)
    valor_tabla = 31.41
    
    diferencia = abs((V-valor_tabla)/V *100)

    # Compara la diferencia
    if (diferencia >1 and diferencia<5) or (diferencia>95 and diferencia < 99):
        return "Los numeros parecen no seguir una distribución uniforme."
    elif (diferencia >1 and diferencia<5) or (diferencia>95 and diferencia < 99):
        return "Los números son sospechosos"
    elif (diferencia >5 and diferencia<10) or (diferencia>90 and diferencia < 95):
        return "Los numeros son casi sospechosos"
    else:
        return "Los numeros parecen seguir una distribución uniforme."

# Generar 1000 números aleatorios entre 0 y 1
aleatorios = np.random.random(1000000)

# Definir el número de intervalos
num_intervalos = 30

print(metodo_chi_2(aleatorios, num_intervalos))