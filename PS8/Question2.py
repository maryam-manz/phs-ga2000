import numpy as np
import matplotlib.pyplot as plt

# Load waveform data from file
def load_waveform(file_name):
    return np.loadtxt(file_name)

# Plot the waveform
def plot_waveform(data, title="Waveform"):
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.show()

# Perform FFT and plot the magnitudes of the first 10000 Fourier coefficients
def plot_fft(data, title="FFT Magnitude"):
    fft_result = np.fft.fft(data)
    fft_magnitudes = np.abs(fft_result)[:10000]  # Take the magnitude of first 1000 coefficients
    plt.figure(figsize=(10, 4))
    plt.plot(fft_magnitudes)
    plt.title(f"{title} - First 10000 FFT Coefficients")
    plt.xlabel("Frequency Component")
    plt.ylabel("Magnitude")
    plt.show()
    


# Load, plot, and analyze data
for instrument in ["piano.txt", "trumpet.txt"]:
    data = load_waveform(instrument)
    plot_waveform(data, title=f"{instrument} Waveform")
    plot_fft(data, title=f"{instrument} FFT Magnitude")


    
    
# Perform FFT and find fundamental frequency
def get_fundamental_frequency(data, sample_rate=44100):
    fft_result = np.fft.fft(data)
    fft_magnitudes = np.abs(fft_result)

    # Find index of the peak in the first half of the FFT result (positive frequencies)
    peak_index = np.argmax(fft_magnitudes[:len(fft_magnitudes) // 2])
    
    # Convert the index to frequency
    fundamental_freq = peak_index * sample_rate / len(data)
    return fundamental_freq

# Convert frequency to the nearest musical note
def frequency_to_note_name(freq):
    # Reference frequency for A4
    A4 = 440.0
    # MIDI note calculation
    midi_note = 69 + 12 * np.log2(freq / A4)
    # Round to nearest MIDI note and convert to note name
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_number = int(round(midi_note))
    octave = (note_number // 12) - 1
    note_name = note_names[note_number % 12]
    return f"{note_name}{octave}"

# Analyze both files
for instrument in ["piano.txt", "trumpet.txt"]:
    data = load_waveform(instrument)
    fundamental_freq = get_fundamental_frequency(data)
    note = frequency_to_note_name(fundamental_freq)
    print(f"{instrument} fundamental frequency: {fundamental_freq:.2f} Hz, closest musical note: {note}")
