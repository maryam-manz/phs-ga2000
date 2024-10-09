import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad



# the function is only defined on the real positive values

# Define the integrand after the change of variables
def integrand(z, a):
    c = a - 1
    x = c * z / (1 - z)  # Change of variables: z = x / (c + x), hence x = c * z / (1 - z)
    dx_dz = c / (1 - z)**2  # Derivative of x with respect to z for the Jacobian
    return (x**(a - 1)) * np.exp(-x) * dx_dz



# Define the gamma function using numerical integration
def gamma_function(a):
    # If a < 1, use the recurrence relation: Gamma(a) = Gamma(a+1) / a 
    if a < 1 and a!=0 :
        return gamma_function(a + 1) / a
    if a == 0:
        return 0
    # If a >= 1, proceed with integration
    integral_value, error = quad(integrand, 0, 1, args=(a))
    return integral_value



# Test the function with different values of a, including a < 1
gamma_3_2 = gamma_function(3/2)
gamma_1 = gamma_function(1)
gamma_0 = gamma_function(0)
gamma_1_2 = gamma_function(1/2)

# Known values for comparison
exact_gamma_3_2 = 0.5 * math.sqrt(math.pi)
exact_gamma_1 = 0  # Gamma(1) is 0
exact_gamma_0 = 0  # Gamma(0) is 0
exact_gamma_1_2 = math.sqrt(math.pi)

# Print results
print(f"Calculated Gamma(3/2) = {gamma_3_2}, Exact Gamma(3/2) = {exact_gamma_3_2}")
print(f"Calculated Gamma(1) = {gamma_1}, Exact Gamma(1) = {exact_gamma_1}")
print(f"Calculated Gamma(0) = {gamma_0}, Exact Gamma(0) = {exact_gamma_0}")
print(f"Calculated Gamma(1/2) = {gamma_1_2}, Exact Gamma(1/2) = {exact_gamma_1_2}")




gamma_3 = gamma_function(3)
gamma_6 = gamma_function(6)
gamma_10 = gamma_function(10)

# Known factorial values for comparison
exact_gamma_3 = 2  # 2! = 2
exact_gamma_6 = 120  # 5! = 120
exact_gamma_10 = 362880  # 9! = 362880

# Print results
print(f"Calculated Gamma(3) = {gamma_3}, Exact Gamma(3) = {exact_gamma_3}")
print(f"Calculated Gamma(6) = {gamma_6}, Exact Gamma(6) = {exact_gamma_6}")
print(f"Calculated Gamma(10) = {gamma_10}, Exact Gamma(10) = {exact_gamma_10}")
