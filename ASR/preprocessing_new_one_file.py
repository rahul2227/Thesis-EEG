# main.py

import os
import numpy as np
import mne
from pyprep.prep_pipeline import PrepPipeline
from data_reader import load_all_eeg_data
import matplotlib.pyplot as plt
import torch

# If you intend to use GPU acceleration via PyTorch (optional)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Base directories
base_dir_task_1 = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR'
base_dir_of_code = '/Users/rahul/PycharmProjects/Thesis-EEG/ASR'  # Where this script/code resides

def preprocess_data(raw, montage_name='GSN-HydroCel-128'):
    """Apply montage, set reference, and run PREP pipeline on the raw data."""
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"Could not set {montage_name} montage: {e}")
        montage = None

    # Set EEG reference
    try:
        raw = raw.set_eeg_reference(['Cz'])
    except ValueError:
        print("Cz not found in channel names. Using average reference instead.")
        raw = raw.set_eeg_reference('average')

    # PREP pipeline parameters
    prep_params = {
        'ref_chs': 'eeg',
        'reref_chs': 'eeg',
        'line_freqs': [50],
        'max_bad_channels': 0.1,
        'filter_kwargs': {
            'l_freq': 1.0,
            'h_freq': 40.0
        }
    }

    # Run PREP pipeline
    prep = PrepPipeline(raw, prep_params, montage=montage)
    prep.fit()
    raw_clean = prep.raw
    return raw_clean

def visualize_data(raw_clean, output_dir, base_filename, num_channels_to_plot=5, duration=5):
    """Create and save various EEG data visualizations."""
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Time-domain plot of the first few EEG channels
    picks = raw_clean.copy().pick('eeg').ch_names[:num_channels_to_plot]
    fig_time = raw_clean.plot(start=0, duration=duration, picks=picks, title='Preprocessed EEG (Time-Domain)', show=False)
    fig_time_file = os.path.join(output_dir, f"{base_filename}_time.png")
    fig_time.savefig(fig_time_file)
    plt.close(fig_time)
    print(f"Time-domain visualization saved to {fig_time_file}")

    # PSD plot
    fig_psd = raw_clean.plot_psd(fmax=50, show=False)
    fig_psd_file = os.path.join(output_dir, f"{base_filename}_psd.png")
    fig_psd.savefig(fig_psd_file)
    plt.close(fig_psd)
    print(f"PSD visualization saved to {fig_psd_file}")

    # Sensor layout plot (topomap of electrode positions)
    fig_sensors = raw_clean.plot_sensors(show=False, kind='topomap')
    fig_sensors_file = os.path.join(output_dir, f"{base_filename}_sensors.png")
    fig_sensors.savefig(fig_sensors_file)
    plt.close(fig_sensors)
    print(f"Sensor layout visualization saved to {fig_sensors_file}")

def preprocess_and_save_single_file(data_entry, base_dir_code):
    """Preprocess a single EEG file and save the preprocessed data and visualizations."""
    eeg_signals = data_entry['eeg_signals']
    s_rate = data_entry['sampling_rate']
    file_path = data_entry['file_path']  # The original .mat file path
    file_name = data_entry['file_name']

    # Create MNE Raw object
    n_channels, n_samples = eeg_signals.shape
    ch_names = [f'E{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=s_rate, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_signals, info)

    # Preprocess data
    raw_clean = preprocess_data(raw, montage_name='GSN-HydroCel-128')
    print(f"Preprocessing completed for {file_name}")

    # Construct output path
    # Input example:
    # /Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat
    # Output:
    # /Users/rahul/PycharmProjects/Thesis-EEG/ASR/task1 - NR/Preprocessed data/YAC/YAC_NR1_EEG.fif
    split_path = file_path.split('osfstorage-archive/')
    if len(split_path) < 2:
        print("Unexpected file path structure:", file_path)
        return

    relative_path_from_task = split_path[1].replace('Raw data', 'Preprocessed data')
    relative_path_from_task = os.path.splitext(relative_path_from_task)[0] + '.fif'
    output_full_path = os.path.join(base_dir_code, relative_path_from_task)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
    raw_clean.save(output_full_path, overwrite=True)
    print(f"Saved preprocessed data to {output_full_path}")

    # Visualization
    # Base filename for visualization (without extension)
    base_filename = os.path.splitext(os.path.basename(output_full_path))[0]
    visualize_data(raw_clean, os.path.dirname(output_full_path), base_filename, num_channels_to_plot=5, duration=5)

# Main script
if __name__ == "__main__":
    all_eeg_data_task_1 = load_all_eeg_data(base_dir_task_1)
    print(f"Total number of EEG files loaded: {len(all_eeg_data_task_1)}")

    for idx, data_entry in enumerate(all_eeg_data_task_1):
        print(f"Processing file {idx+1}/{len(all_eeg_data_task_1)}: {data_entry['file_name']}")
        try:
            preprocess_and_save_single_file(data_entry, base_dir_of_code)
        except Exception as e:
            print(f"Error during preprocessing {data_entry['file_name']}: {e}")