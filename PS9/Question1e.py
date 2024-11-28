import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define the system of first-order equations for the van der Pol oscillator
def van_der_pol(t, y, mu, w):
    x, v = y  # y[0] = x, y[1] = v = dx/dt
    dxdt = v
    dvdt = mu * (1 - x**2) * v - w**2 * x
    return [dxdt, dvdt]

# Parameters
w = 1.0  # Angular frequency
t_span = (0, 20)  # Time range
initial_conditions = [1.0, 0.0]  # Initial conditions: x=1, dx/dt=0

# Solve for different values of mu
mu_values = [1.0, 2.0, 4.0]
colors = ['blue', 'red', 'green']
labels = [r'$\mu = 1$', r'$\mu = 2$', r'$\mu = 4$']

# Plot phase space for each value of mu
plt.figure(figsize=(8, 8))
for mu, color, label in zip(mu_values, colors, labels):
    solution = solve_ivp(
        van_der_pol, t_span, initial_conditions, args=(mu, w), dense_output=True, max_step=0.01
    )
    t_vals = np.linspace(t_span[0], t_span[1], 5000)
    x_vals, v_vals = solution.sol(t_vals)
    plt.plot(x_vals, v_vals, label=label, color=color)

# Plot formatting
plt.title("Phase Space Plot of the van der Pol Oscillator")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (dx/dt)")
plt.grid()
plt.legend()
plt.show()