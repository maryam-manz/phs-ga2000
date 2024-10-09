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

# Calculate the residuals
residuals = signal - signal_svd

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(time, residuals, color='purple', label='Residuals')
plt.title('Residuals of the Data with Respect to the 3rd-Order SVD Polynomial Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Calculate summary statistics of the residuals
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)

print(f'Mean of residuals: {residual_mean}')
print(f'Standard deviation of residuals: {residual_std}')