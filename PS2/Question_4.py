import numpy as np
import matplotlib.pyplot as plt

# # Question 4


N = 50 #NxN grid
iteration = 100

x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)


x_grid, y_grid = np.meshgrid(x, y, indexing='xy')

c_grid = x_grid + y_grid*1j

z = c_grid
  


for i in range(iteration):
    z_p = z**2 + c_grid
    z = z_p
    
ugh = (abs(z)<=2)


plt.imshow(ugh, cmap='gray',extent = [-2,2,-2,2])
