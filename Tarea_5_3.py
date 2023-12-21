# Jean Marroquin

import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd

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

N = len(x)
x1 = np.arange(0,201, 0.1)

# Semilla
a1=70000 
a2=70 
a3=700 

dx=0.00005 # Paso para el cálculo de las derivadas

def g(a1,a2,a3,i): # Función para hacer ajuste 
    return a1/((x[i]-a2)**2+a3)

# Definimos las funciones para minimizar chi cuadrada

def f1(a1,a2,a3): 
    f1=0
    for i in range(N):
        f1+=(y[i]-g(a1,a2,a3,i))/(((x[i]-a2)**2+a3)*z[i]**2)
    return f1

def f2(a1,a2,a3):
    f2=0
    for i in range(N):
        f2+=((y[i]-g(a1,a2,a3,i))*(x[i]-a2))/((((x[i]-a2)**2+a3)**2)*(z[i]**2))
    return f2

def f3(a1,a2,a3): 
    f3=0
    for i in range(N):
        f3+=(y[i]-g(a1,a2,a3,i))/((((x[i]-a2)**2+a3)**2)*(z[i]**2))
    return f3

# Definimos la matriz de derivadas parciales

def matriz_jacobiana(a1,a2,a3):
    jacobian = np.zeros((3, 3))
    # Aproximacion por forward difference
    for i in range(3):
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = dx
            if i == 0:
                jacobian[i][j] = (f1(a1 + delta[0], a2 + delta[1] , a3 + delta[2]) - f1(a1, a2, a3)) / dx
            elif i == 1:
                jacobian[i][j] = (f2(a1 + delta[0], a2 + delta[1] , a3 + delta[2]) - f2(a1, a2, a3)) / dx
            else:
                jacobian[i][j] = (f3(a1 + delta[0], a2 + delta[1] , a3 + delta[2]) - f3(a1, a2, a3)) / dx
    return jacobian

def chi2(a1,a2,a3): # Definimos cómo calcular chi^2
    chi_2=0
    for i in range(N):
        chi_2+=((y[i]-g(a1,a2,a3,i))/sigma[i])**2
    return chi_2

Imax=100 # Número máximo de iteraciones

def NewtonRaphson(a1,a2,a3):
    for i in range(Imax):
        
        f=np.array([f1(a1,a2,a3),f2(a1,a2,a3),f3(a1,a2,a3)])
        
        F=matriz_jacobiana(a1, a2, a3)
        paso = np.linalg.solve(F,-f)
        
        p = abs(paso) # Arreglo con los valores absolutos de paso
        if sum(p) < 0.00001: # Criterio de paro
            break
        
        a1 += paso[0]
        a2 += paso[1]
        a3 += paso[2]
    return a1, a2, a3

  
a1, a2, a3 = NewtonRaphson(a1, a2, a3)
print(f'Valor de fr = {a1} MeV')
print(f'\nEnergía de resonancia Er = {a2} MeV')
print(f'\nAncho a la mitad del máximo \u0393 = {np.sqrt(4*a3)} MeV')
print(f'\nEl valor de \u03C7^2 es {chi2(a1,a2,a3)}')

#Graficamos el ajuste

plt.figure(figsize=(8, 6), dpi=120)
plt.plot(x, y, 'b*', label='Datos originales')
plt.plot(x1, (a1)/((x1-a2)**2+a3), 'g', label='Ajuste')
plt.title('Sección eficaz vs Energía')  
plt.xlabel(r'Energia $E$ [mV]')
plt.ylabel(r'Seccion eficaz $\sigma$')
plt.text(80, 130, r'$E_r = 78.18$ MeV, $\Gamma$ = 59.16 MeV, $f_r =$ 70878.19 MeV')
plt.errorbar(x, y, yerr=z, fmt = '.',color = 'red', ecolor = 'gray', elinewidth = 1, capsize=8, label = 'Incertidumbre')
plt.legend()
plt.grid(True)
plt.show()