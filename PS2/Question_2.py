import numpy as np
import struct
import time
import matplotlib.pyplot as plt

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
