import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import time


# Part A

# Open the FITS file
hdu_list = fits.open('specgrid.fits')

# Extract log10 of wavelength (logwave) and flux (spectra)
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

# Convert logwave back to linear scale (wavelength in Angstroms)
wavelength = 10**logwave



# Part B
normalizations = []

# Normalize the flux for each galaxy
normalized_flux = np.zeros_like(flux)
for i in range(flux.shape[0]):
    # Compute the integral of the flux using the trapezoidal rule
    integral = np.trapz(flux[i], x=wavelength)
    
    # Store the normalization factor
    normalizations.append(integral)
    
    # Normalize the flux
    normalized_flux[i] = flux[i] / integral
    
    

    
    
# Part C
# Compute the mean spectrum across all galaxies
mean_spectrum = np.mean(normalized_flux, axis=0)

# Subtract the mean spectrum from each galaxy's spectrum to get residuals
residuals = normalized_flux - mean_spectrum


# Now, 'residuals' contains the spectra of each galaxy after the mean has been subtracted.


    
    
# Part D

# Number of galaxies (Ngal) and wavelengths (Nwave)
Ngal = residuals.shape[0]
Nwave = residuals.shape[1]

# Compute the covariance matrix: C = (1 / Ngal) * (R @ R.T)
cov_matrix = (1 / Ngal) * np.dot(residuals.T, residuals)  # R.T @ R gives a Nwave x Nwave matrix


start_time_eig = time.time()
# Find the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort the eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]



end_time_eig = time.time()
time_eig = end_time_eig - start_time_eig

# Part G

Nc = 20  # Number of components to keep
mean_spectrum = np.mean(normalized_flux, axis=0)

# Step 1: Rotate the spectra into the eigenspectrum basis to get the coefficients
coefficients = np.dot(residuals, eigenvectors[:, :Nc])

# Step 2: Reconstruct the approximate spectra using only the first Nc components
approx_spectra = np.dot(coefficients, eigenvectors[:, :Nc].T) + mean_spectrum


# Part I

# Initialize lists to store the sum of squared residuals (SSR) and RMS residuals
squared_residuals = []

# Loop over Nc = 1 to 20
for Nc in range(1, 21):  # Nc should range from 1 to 20 (not starting at 0)
    # Reconstruct the spectra using Nc components
    # coefficients[:, :Nc] is (9713, Nc), eigenvectors[:, :Nc] is (4001, Nc)
    approx_spectra = np.dot(coefficients[:, :Nc], eigenvectors[:, :Nc].T) + mean_spectrum  # Ensure Nc eigenvectors are used

    # Compute the squared residuals (SSR = sum of squared differences)
    residuals = normalized_flux - approx_spectra
    squared_residual = np.sum(residuals**2)
    squared_residuals.append(squared_residual)



# Convert squared residuals to RMS residuals
rms_residuals = np.sqrt(np.array(squared_residuals) / normalized_flux.size)

# Plot the sum of squared residuals as a function of Nc
plt.figure(figsize=(8, 6))
plt.plot(range(1, 21), rms_residuals, marker='o', color='blue')
plt.xlabel('Number of Principal Components (Nc)')
plt.ylabel('Root-Mean-Squared (RMS) Residual')
plt.title('RMS Residuals vs. Number of Principal Components')
plt.grid(True)
plt.show()

# Print the RMS residual for Nc = 20
print(f"RMS residual for Nc = 20: {rms_residuals[19]:.6f}")

