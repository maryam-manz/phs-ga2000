import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of first-order equations for the anharmonic oscillator
def anharmonic_oscillator(t, y, w):
    x, v = y  # y[0] = x, y[1] = v = dx/dt
    dxdt = v
    dvdt = -w * x**3
    return [dxdt, dvdt]

# Parameters
w = 1.0  # Anharmonic coefficient
t_span = (0, 50)  # Time range

# Solve for different initial amplitudes
initial_conditions_1 = [1.0, 0.0]  # Initial conditions: x=1, dx/dt=0
initial_conditions_2 = [2.0, 0.0]  # Initial conditions: x=2, dx/dt=0
initial_conditions_3 = [0.5, 0.0]  # Initial conditions: x=0.5, dx/dt=0

solution_1 = solve_ivp(anharmonic_oscillator, t_span, initial_conditions_1, args=(w,), dense_output=True)
solution_2 = solve_ivp(anharmonic_oscillator, t_span, initial_conditions_2, args=(w,), dense_output=True)
solution_3 = solve_ivp(anharmonic_oscillator, t_span, initial_conditions_3, args=(w,), dense_output=True)

# Generate points for phase space plots
t_vals = np.linspace(t_span[0], t_span[1], 1000)
x_vals_1, v_vals_1 = solution_1.sol(t_vals)
x_vals_2, v_vals_2 = solution_2.sol(t_vals)
x_vals_3, v_vals_3 = solution_3.sol(t_vals)

# Plot the phase space diagram for all three solutions
plt.figure(figsize=(8, 8))
plt.plot(x_vals_1, v_vals_1, label='x(0) = 1', color='blue')
plt.plot(x_vals_2, v_vals_2, label='x(0) = 2', color='red')
plt.plot(x_vals_3, v_vals_3, label='x(0) = 0.5', color='green')
plt.title("Phase Space Plot of the Anharmonic Oscillator")
plt.xlabel("Position (x)")
plt.ylabel("Velocity (dx/dt)")
plt.grid()
plt.legend()
plt.show()