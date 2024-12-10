# data_reader.py

import os
import numpy as np
import scipy.io as sio
import h5py

def load_mat_file(file_path):
    try:
        # Try to load using scipy.io.loadmat
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        print(f"Loaded {file_path} using scipy.io.loadmat")
        return mat
    except NotImplementedError:
        # If loadmat fails (e.g., for MATLAB v7.3 files), use h5py
        try:
            mat = h5py.File(file_path, 'r')
            print(f"Loaded {file_path} using h5py.File")
            return mat
        except Exception as e:
            print(f"Error loading {file_path} with h5py: {e}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_eeg_data(base_dir):
    all_eeg_data = []
    for root, dirs, files in os.walk(base_dir):
        if 'Raw data' in root:
            for file in files:
                # Only process files ending with '_EEG.mat'
                if file.endswith('_EEG.mat'):
                    file_path = os.path.join(root, file)
                    mat_data = load_mat_file(file_path)

                    if mat_data is not None:
                        if 'EEG' in mat_data:
                            EEG = mat_data['EEG']
                            print(f"Processing file: {file_path}")
                            # Extract EEG fields
                            eeg_signals, srate, chanlocs, events = extract_eeg_fields(EEG, mat_data)
                            if eeg_signals is None:
                                print(f"Could not extract EEG fields from {file_path}")
                                continue

                            # Transpose if necessary (MNE expects channels x samples)
                            if eeg_signals.shape[0] > eeg_signals.shape[1]:
                                eeg_signals = eeg_signals.T

                            # Additional metadata
                            file_name = file
                            subject = extract_subject(file)
                            task = extract_task(file)

                            data_entry = {
                                'eeg_signals': eeg_signals,
                                'sampling_rate': srate,
                                'chanlocs': chanlocs,
                                'events': events,
                                'file_name': file_name,
                                'subject': subject,
                                'task': task,
                            }
                            all_eeg_data.append(data_entry)
                        else:
                            print(f"'EEG' variable not found in {file_path}")
                    else:
                        print(f"Failed to load {file_path}")
    return all_eeg_data

def extract_eeg_fields(EEG, mat_data=None):
    # Try accessing fields via different methods
    eeg_signals = None
    srate = None
    chanlocs = None
    events = None

    try:
        # Try accessing as simple namespace (when 'squeeze_me=True', 'struct_as_record=False')
        eeg_signals = EEG.data
        srate = EEG.srate
        chanlocs = EEG.chanlocs
        events = EEG.event
        return eeg_signals, srate, chanlocs, events
    except AttributeError:
        # Maybe 'EEG' is a numpy.void or numpy structured array
        if isinstance(EEG, np.ndarray) and EEG.dtype.names is not None:
            # Structured array
            EEG = EEG.item()
            eeg_signals = EEG['data']
            srate = EEG['srate']
            chanlocs = EEG['chanlocs']
            events = EEG['event']
            return eeg_signals, srate, chanlocs, events
        elif isinstance(EEG, np.void):
            # numpy.void, access via field names
            eeg_signals = EEG['data']
            srate = EEG['srate']
            chanlocs = EEG['chanlocs']
            events = EEG['event']
            return eeg_signals, srate, chanlocs, events
        elif isinstance(EEG, dict):
            # Access via dictionary keys
            eeg_signals = EEG['data']
            srate = EEG['srate']
            chanlocs = EEG['chanlocs']
            events = EEG['event']
            return eeg_signals, srate, chanlocs, events
        elif isinstance(EEG, h5py.Group) and mat_data is not None:
            # Loaded using h5py, need to navigate HDF5 structure
            eeg_signals = mat_data[EEG['data'][0, 0]][()]
            srate = EEG['srate'][0, 0]
            chanlocs = EEG['chanlocs']
            events = EEG['event']
            return eeg_signals, srate, chanlocs, events
        else:
            print(f"Unknown EEG data structure: {type(EEG)}")
            return None, None, None, None

def extract_subject(file_name):
    # Assuming file name format 'SUBJECT_TASK_EEG.mat'
    subject = file_name.split('_')[0]
    return subject

def extract_task(file_name):
    # Assuming file name format 'SUBJECT_TASK_EEG.mat'
    task = file_name.split('_')[1]
    return task

# Test for data reader
if __name__ == "__main__":
    eeg_file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'

    mat_data = load_mat_file(eeg_file_path)

    if mat_data is not None:
        if 'EEG' in mat_data:
            EEG = mat_data['EEG']
            print(f"EEG type: {type(EEG)}")

            eeg_signals, srate, chanlocs, events = extract_eeg_fields(EEG, mat_data)
            if eeg_signals is not None:
                # Ensure eeg_signals is a NumPy array
                if not isinstance(eeg_signals, np.ndarray):
                    eeg_signals = np.array(eeg_signals)

                # Print statements to verify data
                print(f"EEG signals shape: {eeg_signals.shape}")
                print(f"Sampling rate: {srate} Hz")
                print(f"Number of channels: {eeg_signals.shape[0]}")
                print(f"Number of events: {len(events)}")
            else:
                print(f"Could not extract EEG fields from {eeg_file_path}")
        else:
            print("'EEG' variable not found in the .mat file.")
    else:
        print("Failed to load the .mat file.")