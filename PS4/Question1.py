import numpy as np
import matplotlib.pyplot as plt
from gaussian_func import *

def temp_func(x=None):
    return (x**4)*np.exp(x)/(np.exp(x)-1)**2


T = np.linspace(5, 500, 250)                      #values of T to evaluate on
b = 426/T
constant = (9*6.022*1.3806*10**2)/b**3



result = constant * np.array(func_gaussian_intg(temp_func, 0, b, 50))
# print(f"Approximate integral: {result}")


# plt.figure(figsize=(10, 6))
plt.title("Cv as a function of T")
plt.plot(T,result)
plt.xlabel('Temperature, K')
plt.ylabel('C_v,  J/(kg⋅K)')
plt.show()
