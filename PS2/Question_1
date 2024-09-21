import numpy as np
import struct
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
