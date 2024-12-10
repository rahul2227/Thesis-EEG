# readerEEG.py - Gaze_Classification/Data_Reader_EEG
# Developed by: Rahul Sharma

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch
from utils import load_mat_file
from filters import bandpass_filter

eeg_file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'
eeg_data = load_mat_file(eeg_file_path)

# Accessing the 'EEG' field
eeg_struct = eeg_data['EEG']
eeg_content = eeg_struct[0, 0]

# Listing fields in 'EEG' struct
eeg_field_names = eeg_content.dtype.names
print("Fields in 'EEG' struct:")
print(eeg_field_names)

# Accessing EEG signals
eeg_data_array = eeg_content['data']
print(f"Type of eeg_data_array: {type(eeg_data_array)}")
print(f"Shape of eeg_data_array: {eeg_data_array.shape}")  # (channels, samples)
print(f"Field names eeg_data_array: {eeg_data_array}")  # (channels, samples)

# Extracting the EEG signals (corrected)
eeg_signals = eeg_data_array
print(f"EEG signals type: {type(eeg_signals)}")
print(f"EEG signals shape: {eeg_signals.shape}")

# Verifying the data
print(f"EEG signals mean: {np.mean(eeg_signals)}")
print(f"EEG signals std: {np.std(eeg_signals)}")
print("First few samples of the first channel:")
print(eeg_signals[0, :10])

# Accessing sampling rate
srate = eeg_content['srate']
print(f"Sampling rate: {srate} Hz")

# Extracting scalar value
srate_scalar = srate[0, 0]
print(f"Sampling rate: {srate_scalar} Hz")

# Number of samples
num_samples = eeg_signals.shape[1]

# Creating a time vector based on the number of samples and sampling rate
time_vector = np.arange(num_samples) / srate_scalar

# Defining filter parameters
lowcut = 1.0  # Low cutoff frequency in Hz
highcut = 40.0  # High cutoff frequency in Hz
order = 4  # Order of the filter

# Using the sampling rate you've extracted
fs = srate_scalar  # Sampling rate in Hz

filtered_eeg = bandpass_filter(eeg_signals, lowcut, highcut, fs, order)

# Plot raw vs. filtered signal for the first channel
channel = 0
time_vector = np.arange(eeg_signals.shape[1]) / fs

# Plotting a segment
segment_duration = 5
segment_samples = int(segment_duration * fs)

# Parameters for PSD computation
nperseg = int(fs * 2)

# Compute PSD for each channel
psd_list = []
freqs = None

for ch in range(filtered_eeg.shape[0]):
    f, Pxx = welch(filtered_eeg[ch, :], fs=fs, nperseg=nperseg)
    psd_list.append(Pxx)
    if freqs is None:
        freqs = f

psd_array = np.array(psd_list)  # (channels, frequencies)

# Defining frequency bands
bands = { 'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 40)}

# Computing band powers
band_powers = {}

for band_name, (low, high) in bands.items():
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    band_power = np.mean(psd_array[:, idx_band], axis=1)
    band_powers[band_name] = band_power  # Shape: (channels,)

# Computing Theta/Beta Ratio
theta_power = band_powers['theta']
beta_power = band_powers['beta']
theta_beta_ratio = theta_power / beta_power

# Average over channels
avg_theta_beta_ratio = np.mean(theta_beta_ratio)
print(f"Average Theta/Beta Ratio: {avg_theta_beta_ratio}")

channel_index = 0
raw_signal = eeg_signals[channel_index, :]
filtered_signal = filtered_eeg[channel_index, :]
bands = { 'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8), 'Alpha (8-13 Hz)': (8, 13), 'Beta (13-30 Hz)': (13, 30)}
band_signals = {}

# Extracting the band signals using bandpass filters
for band_name, (low, high) in bands.items():
    # Apply bandpass filter for each band
    band_signal = bandpass_filter(filtered_signal.reshape(1, -1), low, high, fs, order)
    band_signals[band_name] = band_signal[0, :]

# Creating a time vector for plotting
num_samples = raw_signal.shape[0]
time_vector = np.arange(num_samples) / fs
duration_to_plot = 5
samples_to_plot = int(duration_to_plot * fs)

# Slicing the signals for plotting
time_vector = time_vector[:samples_to_plot]
raw_signal = raw_signal[:samples_to_plot]
filtered_signal = filtered_signal[:samples_to_plot]
for band_name in band_signals:
    band_signals[band_name] = band_signals[band_name][:samples_to_plot]


plt.figure(figsize=(15, 12))

# Plot raw signal
plt.subplot(6, 1, 1)
plt.plot(time_vector, raw_signal, color='blue')
plt.title('Raw EEG Signal (Channel {})'.format(channel_index + 1))
plt.ylabel('Amplitude (µV)')
plt.xlim([time_vector[0], time_vector[-1]])

# Plot filtered signal
plt.subplot(6, 1, 2)
plt.plot(time_vector, filtered_signal, color='green')
plt.title('Filtered EEG Signal (1-40 Hz)')
plt.ylabel('Amplitude (µV)')
plt.xlim([time_vector[0], time_vector[-1]])

# Plot each band signal
subplot_index = 3
for band_name, band_signal in band_signals.items():
    plt.subplot(6, 1, subplot_index)
    plt.plot(time_vector, band_signal)
    plt.title('{} Band Signal'.format(band_name))
    plt.ylabel('Amplitude (µV)')
    plt.xlim([time_vector[0], time_vector[-1]])
    if subplot_index == 6:
        plt.xlabel('Time (s)')
    subplot_index += 1

plt.tight_layout()
plt.show()

num_samples = eeg_signals.shape[1]
srate_scalar = srate[0, 0]
total_time_seconds = num_samples / srate_scalar
total_time_minutes = total_time_seconds / 60

print(f"Total recording time: {total_time_seconds:.2f} seconds")
print(f"Total recording time: {total_time_minutes:.2f} minutes")