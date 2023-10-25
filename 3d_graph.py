import numpy as np
import matplotlib.pyplot as plt

# Define the 2D function you want to plot
def f(x, y):
    return x**2 + y**2

# Create a grid of x and y values
x = np.linspace(-5, 5, 100)  # Adjust the range and resolution as needed
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create a contour plot
plt.contourf(X, Y, Z, cmap='viridis')  # You can choose a different colormap

# Add color bar for reference
plt.colorbar()

# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Contour Plot of Your 2D Function')

# Show the plot
plt.show()
