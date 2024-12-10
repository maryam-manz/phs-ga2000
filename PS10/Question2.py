import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dcst import dst, idst  # Ensure dcst.py is in the working directory

# Constants
L = 1.0          # Length of the box
N = 1000         # Number of slices
m = 1.0          # Particle mass (arbitrary units)
hbar = 1.0       # Reduced Planck's constant (arbitrary units)
time_step = 1e-3      # Time interval between frames (femto second)

# Discretize the box
x = np.linspace(0, L, N, endpoint=False)


# # Initial wavefunction: real and imaginary parts (example function ground state)
# psi_real = np.sin(np.pi * x / L)  # Example real part
# psi_imag = np.zeros_like(x)       # Example imaginary part (purely real initial wavefunction)


# Parameters for Gaussian wave packet
x_0 = L / 2  # Center of the wave packet
sigma = L / 10  # Width of the wave packet
k_0 = 5 * np.pi / L  # Initial wavenumber

# Initial wavefunction: real and imaginary parts
psi_real = np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.cos(k_0 * x)  # Real part
psi_imag = np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.sin(k_0 * x)  # Imaginary part


# Perform discrete sine transforms
alpha_k = dst(psi_real)  # DST for the real part
eta_k = dst(psi_imag)    # DST for the imaginary part

# Energy levels (E_k = hbar^2 k^2 pi^2 / (2mL^2))
k_values = np.arange(1, N)  # k = 1 to N-1
E_k = (hbar**2 * k_values**2 * np.pi**2) / (2 * m * L**2)

# Function to compute real part of the wavefunction at time t
def compute_wavefunction(t):
    alpha_k_t = alpha_k[:N-1] * np.cos(E_k * t / hbar) - eta_k[:N-1] * np.sin(E_k * t / hbar)
    eta_k_t = alpha_k[:N-1] * np.sin(E_k * t / hbar) + eta_k[:N-1] * np.cos(E_k * t / hbar)
    b_k_t = alpha_k_t + 1j * eta_k_t  # Updated coefficients
    psi_real_t = idst(b_k_t.real)  # Inverse sine transform for real part
    return np.pad(psi_real_t, (0, 1), mode='constant', constant_values=0)  # Adjust to match grid

# Create the figure and axis for the animation
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, np.zeros_like(x), label="Re($\psi(x, t)$)")
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)  # Adjust based on expected wavefunction amplitude
ax.set_xlabel("x (10-8 m)")
ax.set_ylabel("Re($\psi(x, t)$)")
ax.set_title("Time Evolution of the Wavefunction (initial condition described as a gaussian)")
ax.legend()
ax.grid()

# Animation update function
def update(frame):
    t = frame * time_step
    psi_real_t = compute_wavefunction(t)
    line.set_ydata(psi_real_t)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=True)


# Save or display the animation
print("Saving animation...")
ani.save("question2_wavefunction_animation.gif", writer=PillowWriter(fps=20))
print("Animation saved. Open question2_wavefunction_animation.gif to view it")

plt.show()
