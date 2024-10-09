import numpy as np
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as jnp


def f(x):
    return 1 + 0.5*math.tanh(2*x)

# Central difference method for numerical derivative
def central_difference(x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def analytic_der(x):
    return 1/(math.cosh(2*x)**2)

def f_jax(x):
    return 1 + 0.5 * jnp.tanh(2*x)


# Generate x values in the range -2 <= x <= 2
x_values = np.linspace(-2, 2, 100)


# Autodiff using JAX
f_prime_autodiff = jax.grad(f_jax)
autodiff_derivative = [f_prime_autodiff(x) for x in x_values]
# Compute numerical and analytic derivatives
numerical_derivative = [central_difference(x) for x in x_values]
analytic_derivative = [analytic_der(x) for x in x_values]

# Plot the results
plt.figure(figsize = (12,6))
plt.plot(x_values, analytic_derivative, label='Analytic Derivative', color='blue', linestyle='-')
plt.plot(x_values, numerical_derivative, label='Numerical Derivative', color='red', linestyle='None', marker='o')
plt.plot(x_values, autodiff_derivative, label='Autodiff Derivative (JAX)', color='green', linestyle='--')

# Labeling the graph
plt.title('Numerical, Analytic, and Autodiff Derivative of f(x) = 1 + 0.5tanh(2x)')
plt.xlabel('x')
plt.ylabel('Derivative f\'(x)')
plt.legend()
plt.grid(True)
plt.show()