import numpy as np
from pylab import *

# Definicion de parametros
N=2000           # Número de pasos
tau=30            # Tiempo en segundos de la simulación
h=(tau/float(N-1)) # Paso del tiempo
M=1          # Masa de los bloques [kg]
k=1              # Constante de resorte 1
k1=1.2             # Constante de resorte 2

# Definimos condiciones iniciales
x1_0 = 0.0  # Posicion inicial de x1
v1_0 = 0.0  # Velocidad inicial vx1
x2_0 = 1.0  # Posicion inicial de x2
v2_0 = 0.0  # Velocidad inicial vx2

alpha1 = (-1/M) * (k1 + k)
alpha2 = (k/M)
alpha3 = (k/M)
alpha4 = (-1/M) * (k1 + k )

# Definimos nuestra ecuación diferencial
def f1(t, y):
  f0 = y[1]
  f1 = alpha1*(y[0]+0.1*y[0]**3) + alpha2*(y[2]+0.1*y[2]**3)
  f2 = y[3]
  f3 = alpha3*(y[0]+0.1*y[0]**3) + alpha4*(y[2]+0.1*y[2]**3)
  return [f0, f1, f2, f3]

def f(t, y):
  f0 = y[1]
  f1 = alpha1*(y[0]) + alpha2*(y[2])
  f2 = y[3]
  f3 = alpha3*(y[0]) + alpha4*(y[2])
  return [f0, f1, f2, f3]

k1 = np.zeros(N)
k2 = np.zeros(N)
k3 = np.zeros(N)
k4 = np.zeros(N)

def rk4(y,t,h,f) :
  k1 = np.multiply(h, f(t, y))
  k2 = np.multiply(h, f(t + h/2, np.add(y, np.divide(k1, 2))))
  k3 = np.multiply(h, f(t + h/2, np.add(y, np.divide(k2, 2))))
  k4 = np.multiply(h, f(t + h, np.add(y, k3)))
  return np.add(y, np.divide(np.add(np.add(np.add(k1, np.multiply(2, k2)), np.multiply(2, k3)), k4), 6))


# Generamos un arreglo de Nx4 para almacenar posiciones y velocidades
y=np.zeros([N,4])
# Arreglo para el caso de resorte no lineal
y1=np.zeros([N,4])
# Tomamos los valores del estado inicial
y[0,0]=x1_0    
y[0,1]=v1_0    
y[0,2]=x2_0    
y[0,3]=v2_0     

y1[0,0]=x1_0    
y1[0,1]=v1_0    
y1[0,2]=x2_0    
y1[0,3]=v2_0 

# Generamos tiempos igualmente espaciados
tiempo=linspace(0,tau,N)

for i in range(N-1):
   y[i+1] = rk4(y[i], tiempo[i], h, f)
   y1[i+1] = rk4(y1[i], tiempo[i], h, f1)

# Calculamos las frecuencias de los modos normales de vibracion, primero definimos la matriz A
A = np.array([[alpha1, alpha2], [alpha3, alpha4]])
eigenvalores, eigenvectores = np.linalg.eig(A)
# Extraemos las frecuencias de los modos normales de vibracion
omega = np.sqrt(-eigenvalores)
# Presentamos las frecuencias de los modos normales de vibracion
print(f"Frecuencias de los modos normales de vibracion: {omega}")

#Graficamos los resultados

x1_datos = np.array([y[j,0] for j in range(N)])
v1_datos = np.array([y[j,1] for j in range(N)])
x2_datos = np.array([y[j,2] for j in range(N)])
v2_datos = np.array([y[j,3] for j in range(N)])
tiempo = np.array([tiempo[j] for j in range(N)])

# Para el caso no lineal
nl_x1_datos = np.array([y1[j,0] for j in range(N)])
nl_v1_datos = np.array([y1[j,1] for j in range(N)])
nl_x2_datos = np.array([y1[j,2] for j in range(N)])
nl_v2_datos = np.array([y1[j,3] for j in range(N)])

figure(figsize=(10, 5))
plot(tiempo, x1_datos, label='x1(t)')
plot(tiempo, x2_datos, label='x2(t)')
plot(tiempo, nl_x1_datos, label='x1(t) [No lineal]')
plot(tiempo, nl_x2_datos, label='x2(t) [No lineal]')
xlabel('Tiempo [s]')
ylabel('Desplazamiento [m]')
legend()
title('Sistema de resortes acoplados')
show()