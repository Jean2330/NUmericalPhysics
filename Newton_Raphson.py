from math import cos
x0 = 1111.0; dx=3.e-4;err=0.001;Nmax=100;  # Parámetros 

def f(x):
    return 2*cos(x)-x    # Función

def NewtonR(x,dx,err,Nmax):
    for it in range(0,Nmax+1):
        F=f(x)
        if (abs(F)<=err):         # ¿Es raíz dentro del error?  
            print('\n Raíz encontrada, f(raíz)=',F,'Raíz=',x,', error=',err) 
            break
        print('Iteracion=',it,'x=',x,'f(x)=',F)
        df=(f(x+dx/2)-f(x-dx/2))/dx  # Central difference
        dx=-F/df 
        x+=dx                         # Nueva propuesta
    if it==Nmax+1: 
        print('\n Newton no encontró raíz para Nmax=',Nmax) 
    return x

NewtonR(x0,dx,err,Nmax)