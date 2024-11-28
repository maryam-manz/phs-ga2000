import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Part a

# Define the system of equations
def harmonic_oscillator(t, y, w):
    x, v = y  # y[0] = x, y[1] = v = dx/dt
    dxdt = v
    dvdt = -w**2 * x
    return [dxdt, dvdt]

# Parameters
w = 1.0  # Angular frequency
t_span = (0, 50)  # Time range
initial_conditions_1 = [1.0, 0.0]  # Initial conditions: x=1, dx/dt=0

# Solve the system using scipy's solve_ivp
solution_1 = solve_ivp(harmonic_oscillator, t_span, initial_conditions_1, args=(w,), dense_output=True)

# Generate points for plotting
t_vals_1 = np.linspace(t_span[0], t_span[1], 1000)
x_vals_1 = solution_1.sol(t_vals_1)[0]

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t_vals_1, x_vals_1, label='x(t)', color='blue')
plt.title("Harmonic Oscillator: x vs. t")
plt.xlabel("Time (t)")
plt.ylabel("Displacement (x)")
plt.grid()
plt.legend()
plt.show()



# Part b
# Solve for initial conditions x=1 and x=2
initial_conditions_2 = [2.0, 0.0]  # Initial conditions: x=2, dx/dt=0

solution_2 = solve_ivp(harmonic_oscillator, t_span, initial_conditions_2, args=(w,), dense_output=True)

# Generate points for plotting
x_vals_2 = solution_2.sol(t_vals_1)[0]

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t_vals_1, x_vals_1, label='x(0) = 1', color='blue')
plt.plot(t_vals_1, x_vals_2, label='x(0) = 2', color='red')
plt.title("Harmonic Oscillator: Comparing Different Initial Amplitudes")
plt.xlabel("Time (t)")
plt.ylabel("Displacement (x)")
plt.grid()
plt.legend()
plt.show()
