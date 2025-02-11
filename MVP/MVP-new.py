#!/usr/bin/env python3
"""
MVP EEG Application

Features:
- Data Loading using MNE and scipy.io from a .mat file
- Preprocessing (bandpass & notch filtering)
- (Optional) Event extraction and epoching
- Feature extraction:
    • Time-domain: mean, standard deviation, peak-to-peak amplitude
    • Frequency-domain: band power in delta (1–4 Hz), theta (4–8 Hz),
      alpha (8–13 Hz), beta (13–30 Hz) computed with Welch’s method
- Training a simple SVM classifier (scikit-learn)
- Real-time simulation:
    • Visualizes one EEG channel in real time using PyQtGraph
    • Simulates online feature extraction and classification on a sliding epoch

Author: Your Name
Date: YYYY-MM-DD
"""

import sys
import os
import time
import numpy as np
import scipy.io as sio
from scipy.signal import welch
import mne

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# PyQtGraph & Qt imports for real-time visualization
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


# =============================================================================
# 1. Data Loading & Preprocessing
# =============================================================================
def load_mat_eeg(file_path):
    """
    Load raw EEG data from a MATLAB .mat file.

    Expects the file to contain a structure 'EEG' with at least:
      - EEG.data : the EEG signals (either shape (n_channels, n_samples)
        or (n_samples, n_channels))
      - EEG.srate: the sampling rate
      - (optional) EEG.event: event structure array

    Returns:
      raw         : mne.io.RawArray object of the EEG data
      events_struct: the event structure (if available, else None)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    EEG = mat_data['EEG']

    # Get EEG data and transpose if needed (we want shape: n_channels x n_samples)
    eeg_signals = EEG.data
    if eeg_signals.shape[0] < eeg_signals.shape[1]:
        # Already n_channels x n_samples
        pass
    else:
        # Transpose if needed (n_samples x n_channels)
        eeg_signals = eeg_signals.T

    srate = float(EEG.srate)

    # Create channel info
    n_channels = eeg_signals.shape[0]
    ch_names = [f"ch_{i}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

    # Create a RawArray with the EEG data
    raw = mne.io.RawArray(eeg_signals, info, verbose=False)

    # Try to get event information if available
    events_struct = getattr(EEG, 'event', None)

    print(f"Loaded '{os.path.basename(file_path)}': shape={eeg_signals.shape}, sfreq={srate} Hz")
    return raw, events_struct


def preprocess_raw(raw, bandpass=(1., 40.), notch=50.0):
    """
    Apply minimal preprocessing: bandpass and notch filtering.

    Parameters:
      raw     : mne.io.Raw object
      bandpass: tuple (l_freq, h_freq) in Hz
      notch   : notch filter frequency (e.g. 50 Hz) or None

    Returns:
      raw     : preprocessed mne.io.Raw object
    """
    print("Preprocessing: applying bandpass filter...")
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
    if notch is not None:
        print(f"Preprocessing: applying notch filter at {notch} Hz...")
        raw.notch_filter(freqs=[notch], verbose=False)
    return raw


# =============================================================================
# 2. Event Extraction and Epoching (Optional)
# =============================================================================
def extract_events_from_struct(events_struct):
    """
    Convert MATLAB EEG.event structures to an MNE-compatible events array.

    Returns:
      mne_events  : numpy array of shape (n_events, 3) with [sample, 0, code]
      event_id_map: dictionary mapping event label to integer code
    """
    mne_events = []
    event_id_map = {}
    current_code = 1

    for e in events_struct:
        latency = int(e.latency)
        etype = e.type
        if isinstance(etype, str):
            if etype not in event_id_map:
                event_id_map[etype] = current_code
                current_code += 1
            code = event_id_map[etype]
        else:
            code = int(etype)
            if code not in event_id_map.values():
                event_id_map[f"code_{code}"] = code
        mne_events.append([latency, 0, code])

    mne_events = np.array(mne_events, dtype=int)
    print(f"Extracted {len(mne_events)} events with mapping: {event_id_map}")
    return mne_events, event_id_map


def epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0):
    """
    Create epochs from raw data using the provided events.

    Parameters:
      raw         : mne.io.Raw object
      mne_events  : numpy array of events (n_events x 3)
      event_id_map: dictionary mapping event labels to codes
      tmin, tmax  : time window for each epoch (in seconds)

    Returns:
      epochs      : mne.Epochs object
    """
    epochs = mne.Epochs(
        raw,
        events=mne_events,
        event_id=event_id_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )
    print(f"Created {len(epochs)} epochs, each {tmax - tmin:.1f}s long.")
    return epochs


# =============================================================================
# 3. Feature Extraction
# =============================================================================
def compute_band_power(signal, srate, band):
    """
    Compute the average power in a given frequency band using Welch's method.

    Parameters:
      signal: 1D numpy array (time series)
      srate : sampling rate in Hz
      band  : tuple (low_freq, high_freq)

    Returns:
      power : average power within the frequency band
    """
    freqs, psd = welch(signal, fs=srate, nperseg=min(256, len(signal)))
    # Select frequencies within the band
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    return band_power


def extract_features_from_epoch(epoch_data, srate):
    """
    For a single epoch (n_channels x n_times), compute features per channel:
      - Time-domain: mean, std, peak-to-peak amplitude
      - Frequency-domain: band power for delta, theta, alpha, beta bands

    Returns:
      feat_vector: concatenated feature vector (for all channels)
    """
    n_channels, _ = epoch_data.shape
    features = []

    # Frequency bands (Hz)
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }

    for ch in range(n_channels):
        sig = epoch_data[ch, :]
        # Time-domain features
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        ptp_val = np.ptp(sig)  # peak-to-peak amplitude

        # Frequency-domain features (band power)
        band_powers = []
        for band in bands.values():
            bp = compute_band_power(sig, srate, band)
            band_powers.append(bp)

        # Combine features for this channel (order can be adjusted)
        ch_features = [mean_val, std_val, ptp_val] + band_powers
        features.extend(ch_features)

    return np.array(features)


def extract_features(epochs):
    """
    Extract features from all epochs.

    For each epoch, compute a feature vector using both time and frequency-domain features.

    Returns:
      X: Feature matrix (n_epochs x n_features)
      y: Labels (if available) from the events
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    srate = epochs.info['sfreq']
    n_epochs = data.shape[0]
    features = []
    for i in range(n_epochs):
        feat_vec = extract_features_from_epoch(data[i], srate)
        features.append(feat_vec)
    X = np.array(features)
    y = epochs.events[:, 2]  # event codes as labels
    print(f"Extracted features for {n_epochs} epochs; feature vector size: {X.shape[1]}")
    return X, y


# =============================================================================
# 4. Modeling
# =============================================================================
def train_classifier(X, y):
    """
    Train an SVM classifier using the extracted features.

    Splits data into training and test sets, prints accuracy, and returns the trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained SVM classifier; Test Accuracy = {acc:.2f}")
    return clf


# =============================================================================
# 5. Real-Time Simulation and Visualization
# =============================================================================
class RealTimeEEGSimulator:
    """
    Simulates real-time EEG streaming and visualization.

    Features:
      - Continuously plots the signal from a selected channel.
      - Maintains a buffer of incoming samples.
      - When enough samples accumulate (epoch_length in seconds),
        extracts features from the latest epoch and, if a model is provided,
        predicts a class.
    """

    def __init__(self, eeg_data, srate, model=None, chunk_size=50, epoch_length=2.0, channel_index=0):
        """
        Parameters:
          eeg_data     : numpy array of shape (n_channels, n_samples)
          srate        : sampling rate in Hz
          model        : trained classifier (optional)
          chunk_size   : number of samples to update at each timer tick
          epoch_length : length of data (in seconds) used for classification
          channel_index: index of the channel to visualize
        """
        self.eeg_data = eeg_data
        self.srate = srate
        self.model = model
        self.chunk_size = chunk_size
        self.epoch_length = epoch_length
        self.channel_index = channel_index

        self.n_channels, self.n_samples = eeg_data.shape
        self.current_idx = 0
        self.buffer = []  # to store incoming samples for classification

        # Set up PyQtGraph window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Simulation")
        self.win.resize(800, 500)
        self.plot = self.win.addPlot(title=f"EEG Channel {channel_index}")
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude', units='µV')
        self.curve = self.plot.plot(pen='y')

        self.x_data = []  # time axis for visualization
        self.y_data = []  # data for visualization

        # Timer interval in ms: (chunk_size/srate)*1000
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int((chunk_size / srate) * 1000))
        self.timer.timeout.connect(self.update_plot)

    def update_plot(self):
        start = self.current_idx
        end = start + self.chunk_size
        if end > self.n_samples:
            self.timer.stop()
            print("End of EEG data reached.")
            return

        # Get new chunk from the selected channel
        chunk = self.eeg_data[self.channel_index, start:end]
        self.current_idx = end

        # Update visualization data
        if self.x_data:
            last_time = self.x_data[-1]
        else:
            last_time = 0.0
        t_values = np.linspace(last_time, last_time + (self.chunk_size / self.srate), num=self.chunk_size,
                               endpoint=False)
        self.x_data.extend(t_values.tolist())
        self.y_data.extend(chunk.tolist())
        self.curve.setData(self.x_data, self.y_data)

        # Append chunk to buffer (for classification simulation)
        self.buffer.extend(chunk.tolist())
        epoch_sample_count = int(self.epoch_length * self.srate)
        if len(self.buffer) >= epoch_sample_count:
            # Take the latest epoch_window worth of samples for prediction
            epoch_chunk = np.array(self.buffer[-epoch_sample_count:])
            # For simplicity, assume one channel: reshape as (1, n_samples)
            # Create a dummy multi-channel epoch with shape (1, 1, n_samples)
            # so that our feature extraction function works (n_channels x n_times)
            epoch_data = epoch_chunk.reshape(1, -1)
            features = extract_features_from_epoch(epoch_data, self.srate).reshape(1, -1)
            if self.model is not None:
                pred = self.model.predict(features)[0]
                print(f"Predicted class for current epoch (last {self.epoch_length}s): {pred}")
            else:
                print("No model available: skipping prediction.")

    def start(self):
        print("\nStarting real-time EEG simulation...\n")
        self.timer.start()
        QtWidgets.QApplication.instance().exec_()


# =============================================================================
# 6. Main: End-to-End Pipeline
# =============================================================================
def main():
    # -----------------------
    # Parameters and file path
    # -----------------------
    # Adjust the file path to point to your .mat EEG file.
    file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'  # <-- CHANGE THIS to your file path

    # -----------------------
    # Data Loading & Preprocessing
    # -----------------------
    raw, events_struct = load_mat_eeg(file_path)
    raw = preprocess_raw(raw, bandpass=(1., 40.), notch=50.0)

    # -----------------------
    # Epoching and Feature Extraction (if events exist)
    # -----------------------
    model = None
    if events_struct is not None:
        mne_events, event_id_map = extract_events_from_struct(events_struct)
        # Create epochs (e.g., 2-second epochs)
        epochs = epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0)
        # Extract features and labels
        X, y = extract_features(epochs)
        # Train a classifier
        model = train_classifier(X, y)
    else:
        print("No event information found; skipping epoch-based training.")

    # -----------------------
    # Real-Time Simulation
    # -----------------------
    # Get raw data (n_channels x n_samples) for simulation
    data = raw.get_data()
    srate = raw.info['sfreq']
    # Set chunk_size to simulate updates every ~0.1 sec (adjust as needed)
    chunk_size = int(srate * 0.1)
    # Create and start the simulator.
    simulator = RealTimeEEGSimulator(
        eeg_data=data,
        srate=srate,
        model=model,  # if available, will perform classification
        chunk_size=chunk_size,
        epoch_length=2.0,  # simulate prediction on 2-second epochs
        channel_index=0  # choose which channel to visualize
    )
    simulator.start()


if __name__ == "__main__":
    main()