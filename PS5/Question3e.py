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

# Sample function to define a Fourier series with sine and cosine terms
def fourier_series(t, *coeffs):
    n = len(coeffs) // 2  # Number of sine and cosine terms
    result = coeffs[0]  # Initial offset
    for i in range(n):
        result += coeffs[2*i + 1] * np.sin(2 * np.pi * (i + 1) * t / period)  # Sine term
        result += coeffs[2*i + 2] * np.cos(2 * np.pi * (i + 1) * t / period)  # Cosine term
    return result


# Set the period to half of the time span to represent the simplist curve first
time_span = time[len(time)-1] - time[0]
period = time_span / 2

# Initial guess for coefficients [offset, sin1, cos1, sin2, cos2, ...]
initial_guess = [0] + [1] * (6)  # 0 offset 2 sine and 2 cosine terms

# Fit the model to the data
params, _ = curve_fit(fourier_series, time, signal, p0=initial_guess)

# Generate fitted data
fitted_signal = fourier_series(time, *params)

# Calculate residuals
residuals = signal - fitted_signal



# Plot original data and fitted model
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, label='Original Data', color='blue', s=10)
plt.plot(time, fitted_signal, '.', label='Fitted Fourier Series', color='red')
plt.title('Signal and Fitted Fourier Series')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()

plt.show()

# Output the estimated parameters and typical periodicity
print("Estimated Parameters:", params)
print("Typical Periodicity (from fitted model):", period)
