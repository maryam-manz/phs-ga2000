
import numpy as np
import matplotlib.pyplot as plt


n_atoms = 1000
n_sample = 10000
tau = 3.053*60 
u = np.random.uniform(0, 1, n_atoms)
t = np.sort(-tau*np.log(1-u)/np.log(2))              # the decay times sorted

#arrays for plotting
time_points = np.linspace(0, max(t)*1.1, n_sample)

remaining_atoms = np.arange(n_atoms, 0, -1)          # range from 1-n_atom


plt.plot(t, remaining_atoms, color='blue', label='Undecayed Atoms')
plt.xlabel('Time (s)')
plt.ylabel('Number of Atoms Remaining')
plt.title('Decay of 1000 Atoms of $^{208}$Tl (Half-life: 3.053 min)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

