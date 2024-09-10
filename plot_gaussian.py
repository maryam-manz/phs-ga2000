
# # #Creating a Gaussian


import matplotlib.pyplot as plt
import numpy as np



iteration = 200
x = np.linspace(-10, 10, iteration)
mu , sd = 0 , 3

a = 1/np.sqrt(2*np.pi*sd**2)



# print(mu , sd)
gaussian = a*np.exp((-(x-mu)**2)/(2*(sd**2)))




plt.figure(figsize = (10,7))
# plt.plot(x, ".")

plt.plot(x, gaussian)

plt.show

plt.savefig('gaussian.png')






