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


# Part H

# Plot c0 vs c1
plt.figure(figsize=(8, 6))
plt.scatter(coefficients[:, 0], coefficients[:, 1], s=10, color='blue', alpha=0.7)
plt.xlabel('c0 (1st PCA coefficient)')
plt.ylabel('c1 (2nd PCA coefficient)')
plt.title('Plot of c0 vs c1')
plt.grid(True)
plt.show()

# Plot c0 vs c2
plt.figure(figsize=(8, 6))
plt.scatter(coefficients[:, 0], coefficients[:, 2], s=10, color='green', alpha=0.7)
plt.xlabel('c0 (1st PCA coefficient)')
plt.ylabel('c2 (3rd PCA coefficient)')
plt.title('Plot of c0 vs c2')
plt.grid(True)
plt.show()


