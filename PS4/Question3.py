import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import roots_hermite
from math import factorial
from gaussian_func import *


# Hermite polynomial using recurrence relation
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

# Harmonic oscillator wavefunction
def psi(n, x):
    return (1/np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))) * np.exp(-x**2 / 2) * H(n, x)

# Set the range of x values
x_vals = np.linspace(-4, 4, 400)

# Plot the wavefunctions for n = 0, 1, 2, 3
plt.figure(figsize=(8, 6))
for n in range(4):
    plt.plot(x_vals, psi(n, x_vals), label=f'n = {n}')

plt.title("Harmonic Oscillator Wavefunctions")
plt.xlabel('x')
plt.ylabel(r'$\psi_n(x)$')
plt.legend()
plt.grid(True)
plt.show()

# Set the range of x values
x_vals_b = np.linspace(-10, 10, 400)

n_b = 30

# Plot the wavefunctions for n = 0, 1, 2, 3
plt.figure(figsize=(8, 6))

plt.plot(x_vals_b, psi(n_b, x_vals_b), label=f'n = {n_b}')

plt.title("Harmonic Oscillator Wavefunctions")
plt.xlabel('x')
plt.ylabel(r'$\psi_n(x)$')
plt.legend()
plt.grid(True)
plt.show()
