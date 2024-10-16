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
    
    

    
plt.figure(figsize = (12,8))
# Plot a handful of normalized spectra
for i in range(5):  # Plot first 5 galaxies
    plt.plot(wavelength, normalized_flux[i], label=f'Galaxy {i+1}')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Normalized Flux')
plt.title('Normalized Spectra of Nearby Galaxies')
plt.legend()
plt.show()
    
    
    
# Part C
# Compute the mean spectrum across all galaxies
mean_spectrum = np.mean(normalized_flux, axis=0)

# Subtract the mean spectrum from each galaxy's spectrum to get residuals
residuals = normalized_flux - mean_spectrum


# Now, 'residuals' contains the spectra of each galaxy after the mean has been subtracted.


    
    
  