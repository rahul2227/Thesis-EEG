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

# Directories
base_dir_task_1 = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR'
base_dir_of_code = '/Users/rahul/PycharmProjects/Thesis-EEG/ASR'  # Directory where this script resides

# Load all EEG data (for task 1)
all_eeg_data_task_1 = load_all_eeg_data(base_dir_task_1)
print(f"Total number of EEG files loaded: {len(all_eeg_data_task_1)}")

for idx, data_entry in enumerate(all_eeg_data_task_1):
    print(f"Processing file {idx + 1}/{len(all_eeg_data_task_1)}: {data_entry['file_name']}")

    eeg_signals = data_entry['eeg_signals']
    s_rate = data_entry['sampling_rate']
    subject = data_entry['subject']
    task = data_entry['task']
    file_path = data_entry['file_path']  # The original .mat file path

    # Create MNE Raw object
    n_channels, n_samples = eeg_signals.shape
    ch_names = [f'E{i + 1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=s_rate, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_signals, info)

    # Load the GSN-HydroCel-128 montage
    try:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"Could not set GSN-HydroCel-128 montage: {e}")
        montage = None

    # Set the EEG reference
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

    # Run the PREP pipeline
    try:
        prep = PrepPipeline(raw, prep_params, montage=montage)
        prep.fit()
        raw_clean = prep.raw
        print(f"Preprocessing completed for {data_entry['file_name']}")

        # Construct output path
        split_path = file_path.split('osfstorage-archive/')
        if len(split_path) < 2:
            print("Unexpected file path structure:", file_path)
            continue

        # Example: "task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat"
        relative_path_from_task = split_path[1]

        # Replace "Raw data" with "Preprocessed data"
        relative_path_from_task = relative_path_from_task.replace('Raw data', 'Preprocessed data')

        # Change extension from .mat to .fif
        relative_path_from_task = os.path.splitext(relative_path_from_task)[0] + '.fif'

        # Construct full output path in the code directory structure
        output_full_path = os.path.join(base_dir_of_code, relative_path_from_task)

        # Ensure directories exist
        os.makedirs(os.path.dirname(output_full_path), exist_ok=True)

        # Save the preprocessed data
        raw_clean.save(output_full_path, overwrite=True)
        print(f"Saved preprocessed data to {output_full_path}")

        # Create and save visualization
        # We'll plot the first 5 EEG channels for the first 5 seconds
        picks = raw_clean.copy().pick('eeg').ch_names[:5]
        tmin = 0
        tmax = 5
        fig = raw_clean.plot(start=tmin, duration=tmax, picks=picks, title='Preprocessed EEG Data', show=False)

        # Construct visualization file path (replace .fif with .png)
        vis_file_path = output_full_path.replace('.fif', '.png')
        fig.savefig(vis_file_path)
        print(f"Visualization saved to {vis_file_path}")

        plt.close(fig)  # Close figure to prevent memory leaks

    except Exception as e:
        print(f"Error during preprocessing {data_entry['file_name']}: {e}")
        continue