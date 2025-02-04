# Preprocessing.py

import os
import numpy as np
import mne
from pyprep.prep_pipeline import PrepPipeline
from data_reader import load_all_eeg_data
import torch

# If you intend to use GPU acceleration via PyTorch (optional)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Base directories
# base_dir_task_1 = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR'
base_dir_task_2 = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task2 - TSR'
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
        raw.set_eeg_reference(['Cz'])
    except ValueError:
        print("Cz not found in channel names. Using average reference instead.")
        raw.set_eeg_reference('average')

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
    prep = PrepPipeline(raw, prep_params, montage=montage)
    prep.fit()
    raw_clean = prep.raw
    return raw_clean

def preprocess_and_save_single_file(data_entry, base_dir_code):
    """Preprocess a single EEG file and save the preprocessed data."""
    eeg_signals = data_entry['eeg_signals']
    s_rate = data_entry['sampling_rate']
    file_path = data_entry['file_path']
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
    split_path = file_path.split('osfstorage-archive/')
    if len(split_path) < 2:
        print("Unexpected file path structure:", file_path)
        # Free memory before return
        del raw_clean, raw, eeg_signals
        return

    relative_path_from_task = split_path[1].replace('Raw data', 'Preprocessed data')
    relative_path_from_task = os.path.splitext(relative_path_from_task)[0] + '.fif'
    output_full_path = os.path.join(base_dir_code, relative_path_from_task)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_full_path), exist_ok=True)

    # Save preprocessed data directly to disk
    raw_clean.save(output_full_path, overwrite=True)
    print(f"Saved preprocessed data to {output_full_path}")

    # Free memory by deleting large variables
    del raw_clean, raw, eeg_signals

# Main script
if __name__ == "__main__":
    # Process files one by one as they are yielded by load_all_eeg_data
    for idx, data_entry in enumerate(load_all_eeg_data(base_dir_task_2), start=1):
        print(f"Processing file {idx}: {data_entry['file_name']}")
        try:
            preprocess_and_save_single_file(data_entry, base_dir_of_code)
        except Exception as e:
            print(f"Error during preprocessing {data_entry['file_name']}: {e}")
        finally:
            # Delete data_entry to free memory
            del data_entry