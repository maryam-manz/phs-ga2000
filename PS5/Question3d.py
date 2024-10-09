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

# Scale the time to avoid numerical issues
time_scaled = (time - np.mean(time)) / np.std(time)


A = np.zeros((len(time_scaled), 17))

# Use a loop to fill in the columns of A
for i in range(17):
    A[:, i] = time_scaled**i


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
plt.plot(time, signal_svd, '.', color='r', label='16th-order Polynomial SVD Fit')
plt.title('Signal vs Time with 16th-Order Polynomial SVD Fit')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)
plt.show()


# Calculate the residuals
residuals = signal - signal_svd


# Calculate summary statistics of the residuals
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)

print(f'Mean of residuals: {residual_mean}')
print(f'Standard deviation of residuals: {residual_std}')



# Function to evaluate polynomials and their condition numbers
def evaluate_polynomials(max_degree, time_scaled, signal):
    condition_numbers = []

    for degree in range(1, max_degree + 1):
        # Create the design matrix for the current polynomial degree
        A = np.vstack([time_scaled**i for i in range(degree + 1)]).T
        
        # Compute the condition number of the design matrix
        condition_number = np.linalg.cond(A)
        condition_numbers.append((degree, condition_number))
        
        # Stop if condition number becomes too large (ill-conditioned matrix)
        if condition_number > 1e6:
            break

    return condition_numbers

# Set maximum polynomial degree to evaluate
max_degree = 16

# Evaluate the condition numbers for polynomials up to max_degree
condition_numbers = evaluate_polynomials(max_degree, time_scaled, signal)

# Output the condition numbers for each polynomial degree
for degree, cond_number in condition_numbers:
    print(f'Degree: {degree}, Condition Number: {cond_number}')

