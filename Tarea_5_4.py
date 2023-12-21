# Jean Marroquin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importamos los datos de un archivo txt
data = pd.read_csv('tVdatos.txt',header=0,delim_whitespace = True)

# Recopilamos los datos de las dos columnas
t = data.iloc[:,0]
V = data.iloc[:,1]
delta = data.iloc[:,2]

# Transformamos los datos a dos arreglos x,y para su mejor manejo
x = np.float64(t)
y = np.float64(V)
z = np.float64(delta)
N=len(x)

x1 = np.arange(0,500,0.1 )
# Definir la función g(x)
def g(x, a1, a2):
    return a1 * np.exp(-a2 * x)

# Definir las funciones f1 y f2
def f1(a1, a2, x, y, z):
    sum_result = 0
    for i in range(N):
        sum_result += ((y[i] - g(x[i], a1, a2)) / z[i]**2) * np.exp(-a2 * x[i])
    return sum_result

def f2(a1, a2, x, y, z):
    sum_result = 0
    for i in range(N):
        sum_result += ((y[i] - g(x[i], a1, a2)) / z[i]**2) * x[i] * np.exp(-a2 * x[i])
    return sum_result

def jacobiana(a1, a2, x, y, z):
    dx = 1e-5
    jacobian = np.zeros((2, 2))
    
    # Aproximacion por forward difference
    for i in range(2):
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = dx
            if i == 0:
                jacobian[i][j] = (f1(a1 + delta[0], a2 + delta[1], x, y, z) - f1(a1, a2, x, y, z)) / dx
            else:
                jacobian[i][j] = (f2(a1 + delta[0], a2 + delta[1], x, y, z) - f2(a1, a2, x, y, z)) / dx
    
    return jacobian

def newton_raphson(x, y, z, semilla, tol=1e-5, max_iter=100):
    a1, a2 = semilla
    
    for i in range(max_iter):
        f1_val = f1(a1, a2, x, y, z)
        f2_val = f2(a1, a2, x, y, z)
        f = np.array([f1_val, f2_val])
        
        # Calcular la matriz jacobiana
        jacobian = jacobiana(a1, a2, x, y, z)
        
        # Resolver el sistema lineal para la actualización
        paso = np.linalg.solve(jacobian, -f)
        
        # Actualizar los valores de a1 y a2
        a1 += paso[0]
        a2 += paso[1]
        
        # Verificar la convergencia
        if np.linalg.norm(paso) < tol:
            break
    
    return a1, a2

# Definimos a chi^2
def chi2(a1,a2,x,y,z):
    chi_2=0
    for i in range(N):
        chi_2+=((y[i]-g(x[i],a1,a2))/z[i])**2
    return chi_2

# Determinamos incertidumbres de V_o y Gamma 

def Sxn(sigma_log,x,n):
    sxn=0
    for i in range(N):
        sxn+=(x[i]**n)/sigma_log[i]
    return sxn

def Sum(sigma_log):
    Sum=0
    for i in range(N):
        Sum += 1/sigma_log[i]
    return Sum

# Valores iniciales para a1 y a2
semilla = [5.0, 0.01]

a1, a2 = newton_raphson(x, y, z, semilla)
print(f'Los valores ajustados son V_o = {a1} y \u0393 = {a2}')

z_log = abs(1/y)*z 
delta = Sum(z_log)*Sxn(z_log,x,2)-(Sxn(z_log,x,1))**2 # Delta
sgm_a1 = Sxn(z_log,x,2)/delta # Sigma cuadrada de log(V0)
sgm_a2 = Sum(z_log)/delta # Sigma cuadrada de Gamma

print(f'\nCon incetidumbre en V_o = {np.exp(np.log(a1))*np.sqrt(sgm_a1)} y con Incetidumbre en \u0393 = {np.sqrt(sgm_a2)}')

# Calculamos el valor de chi^2
print(f'\nEl valor de chi^2 es {chi2(a1,a2,x,y,z)}')

# Graficamos
plt.figure(figsize=(10, 6), dpi=90)
plt.plot(x, y, 'b*', label='Datos originales')
plt.plot(x1, g(x1, a1, a2), 'g', label='Interpolacion') 
plt.title('Voltaje vs Tiempo')
plt.xlabel(r'Tiempo $t$ [ns]')
plt.ylabel(r'Voltaje $V(t)$ [V]')
plt.text(300, 3, r'$V(t)=5.01368 e^{-0.0122t}$')
plt.errorbar(x, y, yerr=z, fmt = '.',color = 'red', ecolor = 'gray', elinewidth = 1, capsize=8, label = 'Incertidumbre') 
plt.legend()
plt.grid(True)
plt.show()

# Grafica semi-log
plt.figure()

log_y=np.log(y)
sigma_log = abs(1/y)*z


#Gáfica de recta de ajuste
logy=-a2*x1+np.log(a1)

plt.plot(x1, logy, color='b', label = 'Semi-log')
plt.title('ln(Voltaje) vs Tiempo') 
plt.xlabel('Tiempo $t$ [ns]')
plt.ylabel(r'$\ln(V)$ [V]')
plt.errorbar(x, log_y, yerr=sigma_log, fmt = '.',color = 'red', ecolor = 'red', elinewidth = 1, capsize=8, label = 'Incertidumbre')
plt.legend()
plt.grid(True)
plt.show()