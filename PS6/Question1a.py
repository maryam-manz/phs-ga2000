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

plt.figure(figsize = (12,8))

# Plot a handful of the spectra
for i in range(5):  # Plot first 5 galaxies
    plt.plot(wavelength, flux[i], label=f'Galaxy {i+1}')


plt.axvline(x=6563, color='k', linestyle='--', label='Hα (6563 Å)')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (10^-17 erg s^-1 cm^-2 A^-1)')
plt.title('Spectra of Nearby Galaxies')
plt.legend()
plt.show()