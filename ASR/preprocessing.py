# main.py

import numpy as np
import mne
from pyprep.prep_pipeline import PrepPipeline
from data_reader import load_all_eeg_data
import matplotlib.pyplot as plt

# Base directory of your dataset
base_dir_task_1 = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR'
base_dir_task_2 = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task2 - TSR'

# Load all EEG data
all_eeg_data_task_1 = load_all_eeg_data(base_dir_task_1) # This only contains the dataset for task 1
print(f"Total number of EEG files loaded: {len(all_eeg_data_task_1)}")

# Process each EEG data entry
for idx, data_entry in enumerate(all_eeg_data_task_1):
    print(f"Processing file {idx+1}/{len(all_eeg_data_task_1)}: {data_entry['file_name']}")

    eeg_signals = data_entry['eeg_signals']
    s_rate = data_entry['sampling_rate']
    subject = data_entry['subject']
    task = data_entry['task']

    # Create MNE Raw object
    n_channels, n_samples = eeg_signals.shape
    ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=s_rate, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_signals, info)

    # Set montage (Optional)
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)
    except Exception as e:
        print(f"Could not set montage: {e}")
        montage = None

    # Store the raw data
    data_entry['raw'] = raw.copy()

    # Prepare PREP pipeline parameters
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
        print("############# I AM HERE")

        # Store the preprocessed data
        data_entry['raw_clean'] = raw_clean
        print(f"Preprocessing completed for {data_entry['file_name']}")
    except Exception as e:
        print(f"Error during preprocessing {data_entry['file_name']}: {e}")
        data_entry['raw_clean'] = None

# Visualization of the first data entry
print(all_eeg_data_task_1)
data_entry = all_eeg_data_task_1[0]

raw = data_entry.get('raw')
raw_clean = data_entry.get('raw_clean')

if raw_clean is None:
    print("Preprocessed data not available for visualization.")
else:
    # Plot settings
    picks = mne.pick_types(raw.info, eeg=True)[:5]
    tmin = 0
    tmax = 5

    # Plot raw data
    fig_raw = raw.plot(start=tmin, duration=tmax, picks=picks, title='Raw EEG Data', show=False)

    # Plot preprocessed data
    fig_clean = raw_clean.plot(start=tmin, duration=tmax, picks=picks, title='Preprocessed EEG Data (ASR Applied)', show=False)

    plt.show()