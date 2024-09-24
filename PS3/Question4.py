
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm



#number of itterations for each N
num_samples = 100000

def give_y(N, num_samples):
    x = np.random.exponential(1,(num_samples, N))    #creates exponential array of size n_samples*len(N) so for each N we will have n_sample value
    y = np.mean(x, axis = 1)         # we want to calculate mean of the N numbers not all the samples thus axis 1
    return y

N_num = [25, 50, 100, 250, 500]                            # value of N we want to test

plt.figure(figsize=(12, 8))

for N in N_num:
    y = give_y(N, num_samples)
    plt.hist(y, bins=50, density=True, alpha=0.6, label=f'N ={N}')  # density is true since its a probability density
    
    #Comparrison Gaussian
    mean = 1
    std= np.sqrt(1/N)
    x_gauss = np.linspace(0,3,1000)
    plt.plot(x_gauss, norm.pdf(x_gauss, mean, std), '--')

plt.title('Distribution of y for different N')
plt.xlabel('y')
plt.ylabel('Probability Density')
plt.legend()
plt.show()



# Show as a function of N how does mean, varience, skewness and distribution change
means_plot = []
variences_plot = []
skews_plot = []
kurtoses_plot = []

for N in N_num:
    y = give_y(N, num_samples)
    means_plot.append(np.mean(y))
    variences_plot.append(np.var(y))
    skews_plot.append(skew(y))
    kurtoses_plot.append(kurtosis(y))


    
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(N_num, means_plot, 'o-')
plt.title('Mean vs N')
plt.xlabel('N')
plt.ylabel('Mean')
 
plt.subplot(2, 2, 2)
plt.plot(N_num, variences_plot, 'o-')
plt.title('Variance vs N')
plt.xlabel('N')
plt.ylabel('Variance')

plt.subplot(2, 2, 3)
plt.plot(N_num, skews_plot, 'o-')
plt.title('Skewness vs N')
plt.xlabel('N')
plt.ylabel('Skewness')

plt.subplot(2, 2, 4)
plt.plot(N_num, kurtoses_plot, 'o-')
plt.title('Kurtosis vs N')
plt.xlabel('N')
plt.ylabel('Kurtosis')

plt.tight_layout()
plt.show()

