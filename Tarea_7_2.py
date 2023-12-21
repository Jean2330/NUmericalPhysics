# Jean Marroquin 

import numpy as np

c = 3e8
m = 139.6 #MeV/c2
tau = 2.6e-8 #s
lambda1 = 0.4   # Actividad

def decaimiento(cinetica, distancia):
    # Calculamos el factor de Lorentz
    #gamma = cinetica/m +1
    gamma = 5
    # Calculamos la velocidad del meson
    velocidad = c*np.sqrt(1 - 1/(gamma**2))    
    # Calculamos el tiempo de recorrido en el sistema laboratorio
    t = distancia/(velocidad)
    sobrevive = 0
    for tiempo in np.arange(0, t + tau, tau):
        decaimiento = np.random.rand()
        if decaimiento < lambda1:
            sobrevive = 0
            break
        else:
            sobrevive = 1
    return sobrevive

total = 0
numero_inicial = 1e6
media = 200
std = 50
e_cinetica = np.random.normal(loc = media, scale= std, size=int(numero_inicial))

for i in range(len(e_cinetica)):
    total += decaimiento(e_cinetica[i], 20)
    
print(f"El numero de mesones que sobrevivieron es: {total}, es decir, el {total/numero_inicial *100:.2f}% de los iniciales.")