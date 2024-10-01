import numpy as np
import matplotlib.pyplot as plt
from gaussian_func import *


# Define the anharmonic function with parameter 'a'
def anharmonic_func(x=None, a=0):
    return 1 / np.sqrt(a**4 - x**4)


a_vals = np.linspace(0.01, 2, 500) #at 0 it goes to inf so for a decent graph going from 0.01 is better

N_val = 20

# Loop over the a_vals array and compute the integral for each value of 'a'
results = []
for a in a_vals:
    result = np.sqrt(8) * func_gaussian_intg(lambda x: anharmonic_func(x=x, a=a), 0, a, N_val)
    results.append(result)

# # Print the results
# print("Approximate integrals:", results)



plt.figure(figsize=(10, 6))
plt.title("Change of period with amplitude")
plt.plot(a_vals,results)
plt.xlabel('Amplitude / m')
plt.ylabel('Period / s')
plt.show()