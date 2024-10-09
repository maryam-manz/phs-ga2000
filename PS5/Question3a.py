import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the signal data
file_path = 'signal.csv'
signal_data = pd.read_csv(file_path)

# Clean the column names
signal_data.columns = signal_data.columns.str.strip()

# Extract time and signal data
time = signal_data['time']
signal = signal_data['signal'].astype(float)

# Plot the signal as a function of time
plt.figure(figsize=(10, 6))
plt.scatter(time, signal, color='b')
plt.title('Signal vs Time (Scatter Plot)')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.grid(True)
plt.show()