import numpy as np
import matplotlib.pyplot as plt

# Define the 1D function you want to plot
def f(x):
    return x**2

# Generate x values
x = np.linspace(-5, 5, 100)  # Adjust the range and resolution as needed

# Calculate the corresponding y values
y = f(x)

# Create a line plot
plt.plot(x, y)

# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Your 1D Function')

# Show the plot
plt.grid(True)  # Add a grid for better visualization (optional)
plt.show()