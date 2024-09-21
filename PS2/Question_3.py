import numpy as np
import struct
import time
import matplotlib.pyplot as plt

# # Question 3 For Loops

# Madelung constant derrivation while keeping all other constants 1
start = time.time()

L = 100 # the number of atoms on each side

M = 0

i = 0
j = 0
k = 0



for i in range(-L, L+1):
    for j in range(-L, L+1):
        for k in range(-L,L+1):
            if i== j == k == 0:
                continue   
            M += ((-1)**abs(i+j+k))/np.sqrt((i**2)+(j**2)+(k**2))
            
            
print(M)
end = time.time()

print(f'Time taken = {end-start}')

            

# # Question 3 without for Loops 

start2 = time.time()
L = 100  # Number of atoms on each side

# Create arrays of i, j, and k values ranging from -L to L excluding 0
i_vals = np.arange(-L, L + 1)
j_vals = np.arange(-L, L + 1)
k_vals = np.arange(-L, L + 1)



# Use meshgrid to create 3D grids for i, j, k values
i_grid, j_grid, k_grid = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')

# Calculate the squared distance for each point (i^2 + j^2 + k^2)
dist_sq = np.float64(i_grid**2 + j_grid**2 + k_grid**2)

# Calculate the alternating sign (-1)^(i + j + k)
signs = (-1)**abs(i_grid + j_grid + k_grid)

# Create a mask to exclude the (0, 0, 0) point
mask = (i_grid == 0) & (j_grid == 0) & (k_grid == 0)

# Remove the zero distance points from the calculation (set dist_sq to avoid division by zero)
dist_sq[mask] = np.inf  # Set the value to inf so that value becomes 0 and isnt added to the sum

result = signs / np.sqrt(dist_sq)

# Sum all the values in the result array to get M
M = np.sum(result)

end2 = time.time() 
print(M)
print(f'Time taken = {end2-start2}')
