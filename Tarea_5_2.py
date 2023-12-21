# Jean Marroquin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Importamos los datos de un archivo txt
data = pd.read_csv('BretWigner.txt',header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
E = data.iloc[:,1]
f = data.iloc[:,2]
sigma = data.iloc[:,3]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
x = np.float64(E)
y = np.float64(f)
z = np.float64(sigma)

# Definimos arreglos donde vayan los datos de la interpolacion
x1 = np.arange(0, 205, 5)


# Crear el spline cúbico
cs = CubicSpline(x, y)

# Evaluamos el spline en los puntos de x1
y_cs = cs(x1)

# Función cuyas raíces intersectan a los splines a la mitad de su altura máxima
def f(x):
    return cs(x)-y_cs[np.argmax(y_cs)]/2

# Método de Newton Rhapson para encontrar raíces
def NewtonR(x,h,err,Nmax):
    for i in range(0,Nmax+1):
        F=f(x)
        if (abs(F)<=err):         # ¿Es raíz dentro del error?  
            #print('\nRaíz encontrada, Raíz=', x) 
            break
        #print('Iteracion=',i,'x=',x,'f(x)=',F)
        df=(f(x+h/2)-f(x-h/2))/h  # Central difference
        dx=-F/df 
        x+=dx                         # Nueva propuesta
    if i==Nmax: 
        print('\n Newton no encontró raíz para Nmax=',Nmax) 
    return x

h=0.1           # Paso de la derivada
err=1e-8         # Precisión de la raíz
Nmax=100         # Número máximo de iteracciones


print(f'\nEnergía de resonancia Er = {x1[np.argmax(y_cs)]} MeV')  
print(f'\nAncho a la mitad del máximo \u0393 ={NewtonR(100,h,err,Nmax)-NewtonR(50,h,err,Nmax)} Mev')


# Graficamos
plt.figure(figsize=(8, 6), dpi=120)
plt.plot(x, z, 'b*', label='Datos originales')
plt.plot(x1, y_cs, 'r', label='Interpolacion por splines cubicos')
plt.title('Sección eficaz vs Energía')  
plt.xlabel(r'Energia $E$ [mV]')
plt.ylabel(r'Seccion eficaz $\sigma$')
plt.text(125, 60, r'$E_r = 75$ MeV, $\Gamma$ = 58.487 MeV')
plt.errorbar(x, y, yerr=z, fmt = '.',color = 'red', ecolor = 'gray', elinewidth = 1, capsize=8, label = 'Incertidumbre')
plt.legend()
plt.grid(True)
plt.show()
