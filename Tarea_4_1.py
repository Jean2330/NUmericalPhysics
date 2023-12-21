import numpy as np
from pylab import *

N=2000           # Número de pasos
x0=0.0           # Posición inicial x [m]
y0=2.0           # Posición inicial y [m]
tau=5            # Tiempo en segundos de la simulación
h=tau/float(N-1) # Paso del tiempo
g=9.81           # Aceleración de gravedad [m/s^2]
R=0.06           # Radio del martillo en [m]
m=7.26           # Masa del martillo [kg]
A = np.pi*R**2   # Seción transersal del martillo [m^2]
rho = 1.2        # Densidad del aire [kg/m^2]
CD = 0.75        # Coeficientes de rozamiento (ajustar dependiendo del régimen)

#Para encontrar la velocidad inicial usamos el método de bisección
Imax=20 # Número máximo de iteraciones
v0_max = 23 # Cota superior
v0_min = 19 # Cota inferior

# Definimos nuestra ecuación diferencial
def f( t , y):
  A = np.pi * (R**2)
  v = (y[1]**2 + y[3]**2)**0.5
  f0 = y[1]
  f1 = -1/(2*m) * rho * A * CD * y[1] * v
  f2 = y[3]
  f3 = -g - 1/(2*m) * rho * A * CD * y[3] * v
  return np.array([f0,f1,f2,f3])

k1 = np.zeros(N)
k2 = np.zeros(N)
k3 = np.zeros(N)
k4 = np.zeros(N)

# Metodo de Runge Kutta 4
def rk4(y,t,h,f) :
	k1 = h*f(t,y)
	k2 = h*f(t+h/2,y+k1/2)
	k3 = h*f(t+h/2,y+k2/2)
	k4 = h*f(t+h,y+k3)
	y=y+(k1+2*(k2+k3)+k4)/6
	return y

for i in range(Imax):

    # Generamos un arreglo de Nx4 para almacenar posición y velocidad
    y=zeros([N,4])

    v0=(v0_max+v0_min)/2.0 # Punto medio
    v = np.sqrt(v0**2+v0**2) # Magnitud de velocidad inicial

    # tomamos los valores del estado inicial
    y[0,0]=x0       # Posición inicial x
    y[0,1]=v0     # Velocidad inicial x
    y[0,2]=y0       # Posición inicial y
    y[0,3]=v0     # Velocidad inicial y

    # Generamos tiempos igualmente espaciados
    tiempo=linspace(0,tau,N)
    vx = linspace(20,21,N)

    # Ahora calculamos!
    for j in range(N-1):
        y[j+1]=rk4(y[j],tiempo[j],h,f)
        if y[j+1,2]<0: # Cuando ya haya pasado la altura del suelo (0 m)
            suelo=j
            if y[j,0]>86.74: # Si la distancia recorrida es mayor que la deseada
                v0_max = v0
                break
            elif y[j,0]<86.74: # Si la distancia recorrida es menor que la deseada
                v0_min = v0
                break

    if abs(y[suelo,0]-86.74)<1e-8: # Criterio de paro
        break

print('Iteración número =', i)
print('\nVelocidad inicial [m/s] =', v)
print('Distancia recorrida [m] =', y[suelo,0])
print('Altura sobre el suelo [m] =', y[suelo,2])
print('Tiempo transcurrido [s] =',tiempo[suelo])

#Graficamos los resultados

x_datos = np.array([y[j,0] for j in range(suelo+1)])
vx_datos = np.array([y[j,1] for j in range(suelo+1)])
y_datos = np.array([y[j,2] for j in range(suelo+1)])
vy_datos = np.array([y[j,3] for j in range(suelo+1)])
tiempo = np.array([tiempo[j] for j in range(suelo+1)])

subplots(figsize=(8, 7))
figure(1)
subplot(2,1,1)
xlabel('Tiempo [s]'); ylabel('Posición y [m]')
plot(tiempo,y_datos, '-', lw=2,color='r')

subplot(2,1,2)
xlabel('Posición x [m]'); ylabel('Posición y [m]')
plot(x_datos,y_datos,'-',lw=2,color='b')

