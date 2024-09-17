

# # Question 1



import numpy as np
import struct
import time
import matplotlib.pyplot as plt



og = 100.98763

bit32 = np.float(og)



# 'f' is the format for 32-bit float, 'I' is for unsigned int
binary_rep = ''.join(f'{c:08b}' for c in struct.pack('>f', bit32))

# print(binary_rep)

# Binary back to float
float_again = struct.unpack('>f', struct.pack('>I', int(binary_rep, 2)))[0]

# print(float_again)

difference = og - float_again

# print (difference)


print(f"Original number: {og}")
print(f"32-bit float number: {bit32}")
print(f"Binary representation: {binary_rep}")
print(f"32 bit float number: {float_again}")
print(f"Difference: {difference}")


# Note: np when i subtracted bit32 from the original number i got 0.0 as output

# # Question 2

# theoretical numbers

print(np.finfo(np.float32()))

print(np.finfo(np.float64()))



smallest_float = np.float32(np.float32(2**(-126))*np.float32(2**(-23)))

np.set_printoptions(suppress= True ,precision= 50)

print(f"{smallest_float:.68f}")




# Precision Test (largest value added to 1 such that result is different from 1)
def find_epsilon(dtype):
    epsilon = dtype(1)
    while dtype(1) + epsilon != dtype(1):
        epsilon /= dtype(2) # divided by 2 since thats the binary value each digit is incremented by
    return epsilon * 2  # The last epsilon before the sum becomes equal to 1

# Finding epsilon for 32-bit and 64-bit floats
epsilon_32 = find_epsilon(np.float32)
epsilon_64 = find_epsilon(np.float64)

# Displaying results
print("Precision Test:")
print(f"Smallest epsilon for np.float32: {epsilon_32}")
print(f"Smallest epsilon for np.float64: {epsilon_64}")



# Start with a large number and keep multiplying by 2 until overflow (approx inf)
# make sure everything is calculated in 32 bits and 64 bits respectively
print("Dynamic range test")

large_32 = np.float32(1e30)
large_64 = np.float64(1e30)


while not np.isinf(np.float64(large_64)): #the function checks if the value in the bracket is infinity
    max_64 = np.float64(large_64) # To store the value before it becomes inf
    large_64 *= 2

while not np.isinf(np.float32(large_32)): #the function checks if the value in the bracket is infinity
    max_32 = np.float32(large_32) # To store the value before it becomes inf
    large_32 *= 2


print(f"Maximum for np.float32: {max_32}")
print(f"Maximum for np.float64: {max_64}")
print(f"Minimum for np.float32: {-max_32}")
print(f"Minimum for np.float64: {-max_64}")


# Start with a small number and keep dividing by 2 until underflow (approx 0)
small_32 = np.float32(1e-30)
small_64 = np.float64(1e-30)

while np.float32(np.float32(small_32)) > 0:
    smallest_32 = np.float32(small_32)
    small_32 /= 2
    
while np.float64(np.float64(small_64)) > 0:
    smallest_64 = np.float64(small_64)
    small_64 /= 2

print(f"Smallest normal for np.float32: {smallest_32}")
print(f"Smallest normal for np.float64: {smallest_64}")


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


# # Question 4
# 


# del j
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


# # Question 5




print("Please enter the value for a")
a = float(input())
print("Please enter the value for b")
b = float(input())
print("Please enter the value for c")
c = float(input())


x1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
x2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)
print(f'The two roots are : x1 = {x1} and x2 = {x2}')




x3 = 2*c/(-b-np.sqrt(b**2-4*a*c))
x4 = 2*c/(-b+np.sqrt(b**2-4*a*c))

print(f'The two roots are : x3 = {x3} and x4 = {x4}')


# #The problem is that when subtracting numbers, the precision of floating-point representation is reduced because the computer can only store a finite number of digits (typically 16). Subtraction between close numbers can lead to significant accuracy loss



def quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    
    # Check if the discriminant is negative, which means complex roots
    if discriminant < 0:
        real_part = -b / (2*a)
        imaginary_part = np.sqrt(-discriminant) / (2*a)
        return (complex(real_part, imaginary_part), complex(real_part, -imaginary_part))
    
   
    sqrt_discriminant = np.sqrt(discriminant)
    
    if b >= 0:
        root1 = (-b - sqrt_discriminant) / (2*a)
        root2 = (2*c) / (-b - sqrt_discriminant)  # Using an alternate form
    else:
        root1 = (-b + sqrt_discriminant) / (2*a)
        root2 = (2*c) / (-b + sqrt_discriminant)  # Using an alternate form
    
    return (root1, root2)




root1, root2 = quadratic_roots(0.001, 1000, 0.001)
print(f"The roots of the quadratic equation are: {root1} and {root2}")







