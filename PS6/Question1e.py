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

# Part E

start_time_svd = time.time()


# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(residuals)

# The right singular vectors (V) are the eigenvectors of R^T * R
eigenvectors_svd = Vt.T  # Transpose to get them in the correct format

# Eigenvalues from S are the squares of the singular values
eigenvalues_svd = S**2  # Square the singular values to get eigenvalues


end_time_svd = time.time()
time_svd = end_time_svd - start_time_svd


# Plot the first five eigenvectors obtained from SVD
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(wavelength, eigenvectors_svd[:, i], label=f'SVD Eigenvector {i+1}')
for i in range(3):
    plt.plot(wavelength, eigenvectors[:, i], label=f'Eigenvector {i+1}')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Eigenvector Value')
plt.title('First 5 Eigenvectors from PCA')


plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Eigenvector Value')
plt.title('First 5 Eigenvectors from SVD')
plt.legend()
plt.show()


print(f"Eigenvalue Decomposition took {time_eig:.4f} seconds")
print(f"SVD Decomposition took {time_svd:.4f} seconds")
