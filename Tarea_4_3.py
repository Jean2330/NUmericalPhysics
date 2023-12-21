import numpy as np
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Tamaño de la barra [m] y tiempo de evolución [s]
L=1 ; t_f=6
# Número de pasos y tamaño de paso
Nx=500; Nt=3000; Dx=L/Nx; Dt=t_f/Nt
# Constantes de la cuerda
T=5 ; rho=0.1
#Vel. de propagacion [m/s]
c=np.sqrt(T/rho)
#Vel. de la malla
c_p=Dx/Dt


print('Condicion de Courant (<=1)=', c/c_p)
print("Tamaño de paso \u0394x: ", Dx);
print("Tamaño de paso \u0394t: ", Dt);

y=np.zeros((Nt+1,Nx+1),float)

#Condiciones de frontera
for j in range(0,Nt+1):
    y[j,0]=0 
    y[j,Nx]=0 

#Condiciones iniciales
for i in range(1,Nx):
    y[0,i]=np.sin(i*Dx*np.pi*4)
    y[1,i]=y[i,0]

# Calculamos la solución en el resto de puntos de la malla
f=(c**2)/(c_p**2)
for j in range(1,Nt):
    for i in range (1,Nx):
        y[j+1,i]=2*y[j,i]-y[j-1,i]+f*(y[j,i+1]+y[j,i-1]-2*y[j,i])

# Graficacion
fig, ax = plt.subplots()
x=np.arange(Nx+1) # Definimos un arreglo para x (espacio)
x=x*Dx # Normalizamos los valores x=iΔx (0<=x<=L)
X=list(range(0,Nx+1))   #Rescatador de valores
line,=ax.plot(x,y[0,X])

def animate(i):
    line.set_ydata(y[i,X])  #Actualización de datos 'y'
    return line,

ani = animation.FuncAnimation(fig,animate,interval=1,blit=False)
plt.xlim([0,L])
plt.ylim(-10,10)
ax.set_ylabel("Amplitud y(x,t) (m)")
ax.set_xlabel("Eje x (m)")
plt.title("Vibraciones en cuerda con extremos fijos")
plt.grid(visible=True,which="both")
plt.show()