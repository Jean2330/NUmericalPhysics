# Jean Marroquin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
x1 = np.arange(0, 201, 5)

# Definimos una funcion que calcule el valor de la interpolacion en un punto
def Lagrange(x1):
    z1 = 0
    for i in range(len(x)):
        prod = 1
        for j in range(len(x)):
            if i != j:
                prod *= (x1 - x[j]) / (x[i] - x[j])
        z1 += prod * z[i]
    return z1

y1=[0] * len(x1)

for k, punto in enumerate(x1):
    y1[k] = Lagrange(punto)


# Función cuyas raíces intersectan al polinomio a la mitad de su altura máxima
def f(x): 
    return Lagrange(x)-Lagrange(75)/2

# Método de Newton Rhapson para encontrar raíces
def NewtonR(x,h,err,Nmax):
    for i in range(0,Nmax+1):
        F=f(x)
        if (abs(F)<=err):         # ¿Es raíz dentro del error?  
            #print('\nRaíz encontrada. Raíz=',x) 
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

print(f'\nEnergía de resonancia Er = {x1[np.argmax(y1)]} MeV')  
print(f'\nAncho a la mitad del máximo \u0393 ={NewtonR(106,h,err,Nmax)-NewtonR(45,h,err,Nmax)} Mev')

# Graficamos
plt.figure(figsize=(10, 6), dpi=180)
plt.plot(x, z, 'b*', label='Datos originales')
plt.plot(x1, y1, 'g', label='Interpolacion') 
plt.title('Sección eficaz vs Energía')  
plt.xlabel(r'Energia $E$ [mV]')
plt.ylabel(r'Seccion eficaz $\sigma$')
plt.errorbar(x, y, yerr=z, fmt = '.',color = 'red', ecolor = 'gray', elinewidth = 1, capsize=8, label = 'Incertidumbre')
plt.legend()
plt.grid(True)
plt.show()