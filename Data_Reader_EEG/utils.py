# utils.py - Gaze_Classification/Data_Reader_EEG
# Developed by: Rahul Sharma

import scipy.io
import h5py


def load_mat_file(file_path):
    try:
        # Try to load using scipy.io.loadmat
        mat = scipy.io.loadmat(file_path)
        print(f"Loaded {file_path} using scipy.io.loadmat")
        return mat
    except NotImplementedError:
        # If loadmat fails, use h5py for MATLAB v7.3 files
        mat = h5py.File(file_path, 'r')
        print(f"Loaded {file_path} using h5py.File")
        return mat
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def list_mat_fields(mat):
    fields = [key for key in mat.keys() if not key.startswith('__')]
    return fields


def list_h5py_fields(mat):
    def recurse_keys(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    keys = []
    mat.visititems(recurse_keys)
    return keys
