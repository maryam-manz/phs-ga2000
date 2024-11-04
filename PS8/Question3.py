import numpy as np
import matplotlib.pyplot as plt

# Part a: Read in data and plot
data = np.loadtxt("dow.txt")  # Assuming dow.txt is in the same directory
days = np.arange(len(data))

plt.figure(figsize=(12, 6))
plt.plot(days, data, label="Original Data", color="blue")
plt.title("Dow Jones Industrial Average Daily Closing Value")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.show()

# Part b: Calculate the Fourier coefficients using rfft
fourier_coeffs = np.fft.rfft(data)
# Part c: Set all but the first 10% of elements to zero
N = len(fourier_coeffs)
filtered_10 = fourier_coeffs.copy()
filtered_10[int(0.1 * N):] = 0

# Calculate inverse Fourier transform for filtered data
smoothed_data_10 = np.fft.irfft(filtered_10, n=len(data))
# Part d: Plot the smoothed data with 10% coefficients
plt.figure(figsize=(12, 6))
plt.plot(days, data, label="Original Data", color="blue")
plt.title("Dow Jones Industrial Average Daily Closing Value")
plt.xlabel("Days")
plt.ylabel("Closing Value")

plt.plot(days, smoothed_data_10, label="10% Coefficients", color="orange")
plt.legend()
plt.show()

# Part e: Set all but the first 2% of elements to zero
filtered_2 = fourier_coeffs.copy()
filtered_2[int(0.02 * N):] = 0

# Calculate inverse Fourier transform for 2% filtered data
smoothed_data_2 = np.fft.irfft(filtered_2, n=len(data))

# Plot the smoothed data with 2% coefficients
plt.figure(figsize=(12, 6))
plt.plot(days, data, label="Original Data", color="blue")
plt.plot(days, smoothed_data_2, label="2% Coefficients", color="green")
plt.title("Fourier Smoothing with 2% of Coefficients")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.legend()
plt.show()


