import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Load the signal data
file_path = 'signal.csv'
signal_data = pd.read_csv(file_path)

# Clean the column names
signal_data.columns = signal_data.columns.str.strip()

# Extract time and signal data
time = signal_data['time']
signal = signal_data['signal'].astype(float)


# Scale the time to avoid numerical issues
time_scaled = (time - np.mean(time)) / np.std(time)

# Create the matrix A for the third-order polynomial (1, t, t^2, t^3)
# A = np.vstack([np.ones(len(time_scaled)), time_scaled, time_scaled**2, time_scaled**3]).T

A = np.zeros((len(time_scaled), 4)) #4 since its 3rd order polynomial
A[:, 0] = 1.
A[:, 1] = time_scaled 
A[:, 2] = time_scaled**2
A[:, 3] = time_scaled**3
# print(A)



# Perform SVD on A
U, W, Vt = np.linalg.svd(A, full_matrices=False)


# Compute the pseudo-inverse of the diagonal matrix of singular values
W_inv = np.diag(1 / W)

# Use SVD components to solve for the coefficients
x_svd = Vt.T @ W_inv @ U.T @ signal

signal_svd = A.dot(x_svd)

# Plot the original signal and the polynomial fit
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='b', label='Original Signal')
# plt.plot(time, signal, '.', label='data')
plt.plot(time, signal_svd, '.', color='r', label='3rd-order Polynomial SVD Fit')
plt.title('Signal vs Time with 3rd-Order Polynomial SVD Fit')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)
plt.show()