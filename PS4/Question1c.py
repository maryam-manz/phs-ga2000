import numpy as np
import matplotlib.pyplot as plt
from gaussian_func import *

def temp_func(x=None):
    return (x**4)*np.exp(x)/(np.exp(x)-1)**2


T = np.linspace(5, 20, 5)                      #values of T to evaluate on
b = 426/T
constant = (9*6.022*1.3806*10**2)/b**3



# Evaluate the integral for each N
N_values = [10, 20, 30,40, 50, 60, 70]
results_test = []
for N in N_values:
    result_test = constant * func_gaussian_intg(temp_func, 0, b, N)
    results_test.append(result_test)

# Plot the convergence
plt.figure(figsize=(12, 8))
plt.plot(N_values, results_test, marker = 'o')
plt.title("Convergence of Gaussian Quadrature")
plt.xlabel("Number of Nodes (N)")
plt.ylabel("Approximate Integral Value")
plt.grid(True)
plt.xticks(N_values)
plt.legend()
plt.show()