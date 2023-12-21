# Jean Marroquin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importamos los datos de un archivo txt
data = pd.read_csv('COBE.txt',header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
nu = data.iloc[:,0]
I = data.iloc[:,1]
delta = data.iloc[:,2]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
x = np.float64(nu)
y = np.float64(I)
z = np.float64(delta)*0.001
N = len(x)

x1 = np.arange(0,22,0.1 )

alpha = 39.72891714
beta = 1.43877688

# Definición de g(x)
def g(x, a1):
    return (alpha* (x ** 3)) / (np.exp((beta*x)/a1) - 1)

# Función f_1(a1)
def f1(a1, x, y, z):
    suma = 0
    for i in range(N):
        suma += ((y[i]-g(x[i],a1))/z[i]**2)*((np.exp(beta*x[i]/a1)*(x[i]**4))/((a1**2)*(np.exp(x[i]*beta/a1)-1)**2))
    return suma

def jacobiana(a1, x, y, z):
    dx = 1e-5
    return (f1(a1 + dx, x, y, z) - f1(a1, x, y, z)) / dx

def newton_raphson(x, y, z, semilla, tol=1e-5, max_iter=100):
    a1 = semilla
    
    for i in range(max_iter):
        f1_val = f1(a1, x, y, z)
        paso = -f1_val / jacobiana(a1, x, y, z)  # Actualización de Newton-Raphson para una variable
        
        # Actualizar el valor de a1
        a1 += paso
        
        # Verificar la convergencia
        if abs(paso) < tol:
            break
    
    return a1

# Definimos a chi^2
def chi2(a1,x,y,z):
    chi_2=0
    for i in range(N):
        chi_2+=((y[i]-g(x[i],a1))/z[i])**2
    return chi_2


# Valor inicial para a1
semilla = 3.0

# Calculamos T_max
a1 = newton_raphson(x, y, z, semilla)
print(f'El valor de la temperatura es {a1} K')

# Calculamos a chi^2
print(f'El valor de chi^2 es {chi2(a1,x,y,z)}')

plt.figure(figsize=(8, 6), dpi=120)
plt.plot(x, y, 'b*', label='Datos originales')
plt.plot(x1, g(x1, a1), 'g', label='Ajuste')
plt.title('Frecuencia vs Intensidad (COBE)')  
plt.xlabel(r'Frecuencia $\nu$ [1/cm]')
plt.ylabel(r'Intensidad $I$ [MJy/sr]')
plt.text(15, 300, r'$T_{Max}=$2.725 K')
plt.errorbar(x, y, yerr=100*z, fmt = '.',color = 'red', ecolor = 'blue', elinewidth = 1, capsize=8, label='Incertidumbre*100') 
plt.legend()
plt.grid(True)
plt.show()
