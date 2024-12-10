import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import solve_banded

# Constants
L = 1.0  # Length of the box (arbitrary units)
N = 1000  # Number of spatial slices
a = L / N  # Spatial step size
h = 1e-3  # Time step (femto seconds)
M = 1.0  # Particle mass (arbitrary units)
hbar = 1.0  # Reduced Planck's constant (arbitrary units)

# # Discretization coefficients
b = 1 - 1j * h * hbar / (2 * M * a**2)
b2 = 1j * h * hbar / (4 * M * a**2)
# scaling_factor = 1e6  # Amplify for testing and animation
# b = 1 - 1j * scaling_factor * h * hbar / (2 * M * a**2)
# b2 = 1j * scaling_factor * h * hbar / (4 * M * a**2)

# Spatial grid
x = np.linspace(0, L, N+1)


# Initial example wavefunction (Gaussian wave packet)
sigma = 0.1  # Width of the wave packet
x0 = L / 2  # Initial position of the wave packet
psi = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * 5 * np.pi * x / L)
psi[0] = psi[-1] = 0  # Boundary conditions (psi = 0 at walls)

# Matrix A and B setup
main_diag_A = np.full(N-1, b)
off_diag_A = np.full(N-2, b2)
main_diag_B = np.full(N-1, np.conj(b))
off_diag_B = np.full(N-2, np.conj(b2))

# Crank-Nicolson step
def crank_nicolson_step(psi):
    v = np.zeros(N-1, dtype=complex)
    for i in range(1, N-2):
        v[i] = (
            main_diag_B[i] * psi[i+1]
            + off_diag_B[i-1] * psi[i]
            + off_diag_B[i] * psi[i+2]
        )
    v[0] = main_diag_B[0] * psi[1] + off_diag_B[0] * psi[2]
    v[-1] = main_diag_B[-1] * psi[-2] + off_diag_B[-1] * psi[-3]

    ab = np.zeros((3, N-1), dtype=complex)
    ab[0, 1:] = off_diag_A  # Upper diagonal
    ab[1, :] = main_diag_A  # Main diagonal
    ab[2, :-1] = off_diag_A  # Lower diagonal
    psi_next = solve_banded((1, 1), ab, v)

    psi[1:-1] = psi_next
    psi[0] = 0  # Boundary conditions
    psi[-1] = 0

    return psi




# Animation setup
fig, ax = plt.subplots()
line, = ax.plot(x, np.real(psi), label="Re(psi)", color="blue")
ax.set_xlim(0, L)

# Dynamically adjust the y-axis limits based on the initial wavefunction
y_scale_factor = 1.5
y_max = np.max(np.abs(psi)) * y_scale_factor
ax.set_ylim(-y_max, y_max)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Real Part of Wavefunction")
ax.set_title("Time Evolution of the Wavefunction")
ax.legend()

# Animation update function
def update(frame):
    global psi
    # print(f"Animating frame {frame}")
    psi = crank_nicolson_step(psi)  # Update the wavefunction
    line.set_ydata(np.real(psi))  # Update line with real part of psi
    # print(f"Frame {frame}: Real Part of Psi: {np.real(psi[:5])}")  # Debug output
    return line,

# Create animation
# NOTE: at this current frame rate it takes a minute or two to render and save and then show the animation. For quicker computation reduce the frame rate
ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=True)
print("Saving animation...")
ani.save("question1_wavefunction_animation.gif", writer=PillowWriter(fps=20))
print("Animation saved. Open question1_wavefunction_animation.gif to view it")

plt.show()
