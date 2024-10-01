import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import roots_hermite
from math import factorial
from gaussian_func import *


def H(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H0 = np.ones_like(x)
        H1 = 2 * x
        for i in range(1, n):
            H_next = 2 * x * H1 - 2 * i * H0
            H0, H1 = H1, H_next
        return H1

def psi(n, x):
    return (1/np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))) * np.exp(-x**2 / 2) * H(n, x)

def uncertainty_func(n=0, x=None):
    psi_squared = psi(n, x)**2
    return x**2 * psi_squared

N_val = 100


result = np.sqrt(func_gaussian_intg(lambda x: uncertainty_func(n=5, x=x), -10, 10, N_val))


# Print the result
print("Uncertainity using Gauss-Legendre Quadrature:", result)



# Perform Gauss-Hermite quadrature for the given function f over the range (-inf, inf)
def gauss_hermite_integrate(f, n_points):
    # Get the roots and weights for the Gauss-Hermite quadrature
    nodes, weights = roots_hermite(n_points)
    
    # Compute the integral
    integral = sum(weights * f(nodes))
    return integral


N_val = 20  # Number of quadrature points

# Compute the integral using Gauss-Hermite quadrature
result = gauss_hermite_integrate(lambda x: uncertainty_func(n=5, x=x) * np.exp(-x**2), N_val)

# Print the result
print("Uncertainty value using Gauss-Hermite quadrature:", result)