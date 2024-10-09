import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

def gamma_f(a, x):
    return np.exp(-x)*x**(a-1)

# Generate x values in the range 0 <= x <= 5
x_values = np.linspace(0, 5, 100)


a_values = [2,3,4]

plt.figure(figsize = [10,6] )

for a in a_values:
    integrand_vals = [gamma_f(a, x) for x in x_values]
    plt.plot(x_values, integrand_vals, label=f'a={a}')
    
    
    
plt.xlabel('x')
plt.ylabel('Integrand Value')
plt.title('Integrand Values for Different a')
plt.legend()
plt.grid(True)
plt.show()