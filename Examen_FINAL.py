# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:06:28 2023

@author: Carlos Higuera
"""
from numpy import *
import matplotlib.pyplot as plt  

'''
a) Encontrar el valor de mu
'''

A=0.   # E_min
B=2.0  # E_max
N=1001 #Número de puntos para integrar

def f(mu,E): # Definimos f_FD
    return 1/(exp((E-mu)/0.025)+1) 

def simpson(A,B,N,mu): # integración por Simpson (N impar)
    h=(B-A)/(N-1)      # tamaño del paso
    sum=(f(mu,A)+f(mu,B))/3    # (primero+último)/3 (0,N-1)
    for i in range(1,N-1,2):   #Términos intermedios
        sum+=(4/3)*f(mu,A+i*h)     
    for i in range(2,N-2,2): 
        sum+=(2/3)*f(mu,A+i*h)     
    return h*sum # Resultado final

def fn(mu): # Función que queremos encontrar su raiz variando mu
    return simpson(A,B,N,mu)-1
    
def NewtonR(x,h,err,Nmax): # x = mu
    for i in range(0,Nmax+1):
        F=fn(x)
        if (abs(F)<=err):         # ¿Es raíz dentro del error?  
            print('\nRaíz encontrada, f(raíz)=',F,'Raíz=',x,', error=',err) 
            break
        print('Iteracion=',i,'x=',x,'f(x)=',F)
        df=(fn(x+h/2)-fn(x-h/2))/h  # Central difference
        dx=-F/df 
        x+=dx                         # Nueva propuesta
    if i==Nmax: 
        print('\n Newton no encontró raíz para Nmax=',Nmax) 
    return x

h=0.05           # Paso de la derivada
err=1.e-14       # Precisión de la raíz
Nmax=100         # Número máximo de iteracciones (Newton Rhapson)

mu=NewtonR(0,h,err,Nmax)

print('\nValor de mu encontrado =',mu)

'''
Inciso b) Graficar f con el valor de mu encontrado
'''

X = linspace(0,2,1000) # Dominio
Y = f(mu,X) # Evaluamos la función f en el dominio

plt.xlabel('Energía [eV]'); plt.ylabel('f_FD')
plt.plot(X,Y,'-',c='r')      # Graficamos
plt.grid(True)               # Ponemos cuadrícula
plt.show()                   # Mostramos la gráfica  
