#!/usr/bin/env python3
"""
MVP EEG Application with 3D Topography Visualization and Multi-Channel Real-Time Display

Features:
- Data Loading using MNE and scipy.io from a .mat file
- Preprocessing (bandpass 1–40 Hz and notch filtering at 50 Hz)
- (Optional) Event extraction and epoching
- Feature Extraction:
    • Time-domain: mean, std, peak-to-peak amplitude
    • Frequency-domain: band power in delta (1–4 Hz), theta (4–8 Hz),
      alpha (8–13 Hz), beta (13–30 Hz) using Welch’s method
- Modeling: Simple SVM classifier using scikit-learn
- Real-time Simulation:
    • Multi-channel visualization using PyQtGraph (each channel is vertically offset)
    • Simulates online feature extraction and classification on a sliding epoch
- 3D Visualization:
    • Displays a 3D scalp topography (using Plotly) for the latest epoch
      by mapping sensor average amplitudes onto their 3D coordinates

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import sys
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

# Plotly for 3D visualization
import plotly.graph_objects as go


# =============================================================================
# 1. Data Loading & Preprocessing
# =============================================================================
def load_mat_eeg(file_path):
    """
    Load raw EEG data from a MATLAB .mat file.

    Expects the file to contain a structure 'EEG' with:
      - EEG.data : EEG signals (shape can be (n_channels, n_samples) or vice versa)
      - EEG.srate: sampling rate
      - (optional) EEG.event: event structure array

    Returns:
      raw         : mne.io.RawArray object
      events_struct: event structure (if available, else None)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    EEG = mat_data['EEG']

    # Ensure data is in shape (n_channels, n_samples)
    eeg_signals = EEG.data
    if eeg_signals.shape[0] >= eeg_signals.shape[1]:
        eeg_signals = eeg_signals.T

    srate = float(EEG.srate)

    # Create channel info using generic names (we will rename later if needed)
    n_channels = eeg_signals.shape[0]
    ch_names = [f"ch_{i}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

    # Create a RawArray with the EEG data
    raw = mne.io.RawArray(eeg_signals, info, verbose=False)
    events_struct = getattr(EEG, 'event', None)

    print(f"Loaded '{os.path.basename(file_path)}': shape={eeg_signals.shape}, sfreq={srate} Hz")
    return raw, events_struct


def preprocess_raw(raw, bandpass=(1., 40.), notch=50.0):
    """
    Apply bandpass and notch filtering to raw data.
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
      event_id_map: dict mapping event labels to integer codes
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
    Create epochs from raw data using events.
    """
    epochs = mne.Epochs(raw, events=mne_events, event_id=event_id_map,
                        tmin=tmin, tmax=tmax, baseline=None, preload=True,
                        verbose=False)
    print(f"Created {len(epochs)} epochs, each {tmax - tmin:.1f}s long.")
    return epochs


# =============================================================================
# 3. Feature Extraction
# =============================================================================
def compute_band_power(signal, srate, band):
    """
    Compute average power in a frequency band using Welch's method.
    """
    freqs, psd = welch(signal, fs=srate, nperseg=min(256, len(signal)))
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    # Use np.trapezoid (np.trapz is deprecated)
    band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
    return band_power


def extract_features_from_epoch(epoch_data, srate):
    """
    For one epoch (n_channels x n_times), compute:
      - Time-domain: mean, std, peak-to-peak amplitude
      - Frequency-domain: band power for delta, theta, alpha, beta bands
    """
    n_channels, _ = epoch_data.shape
    features = []
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    for ch in range(n_channels):
        sig = epoch_data[ch, :]
        ch_feats = [np.mean(sig), np.std(sig), np.ptp(sig)]
        ch_feats += [compute_band_power(sig, srate, band) for band in bands.values()]
        features.extend(ch_feats)
    return np.array(features)


def extract_features(epochs):
    """
    Extract features from all epochs.

    Returns:
      X: Feature matrix (n_epochs x n_features)
      y: Labels (from events)
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    srate = epochs.info['sfreq']
    X = np.array([extract_features_from_epoch(epoch, srate) for epoch in data])
    y = epochs.events[:, 2]  # event codes as labels
    print(f"Extracted features for {len(data)} epochs; each feature vector has size {X.shape[1]}.")
    return X, y


# =============================================================================
# 4. Modeling
# =============================================================================
def train_classifier(X, y):
    """
    Train an SVM classifier using a train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained SVM classifier; Test Accuracy = {acc:.2f}")
    return clf


# =============================================================================
# 5. Real-Time Simulation and Visualization (Multi-Channel)
# =============================================================================
class RealTimeMultiChannelEEGSimulator:
    """
    Simulates real-time EEG streaming and visualization for all channels.
    Each channel is displayed as a trace in a single plot with a vertical offset.
    Also performs online feature extraction and classification on sliding epochs.
    """
    def __init__(self, eeg_data, srate, model=None, chunk_size=50,
                 epoch_length=2.0, offset=100.0):
        self.eeg_data = eeg_data
        self.srate = srate
        self.model = model
        self.chunk_size = chunk_size
        self.epoch_length = epoch_length
        self.offset = offset  # vertical offset per channel

        self.n_channels, self.n_samples = eeg_data.shape
        self.current_idx = 0
        # Common time axis
        self.x_data = []
        # Data for each channel (list of lists)
        self.y_data = [[] for _ in range(self.n_channels)]

        # Set up PyQtGraph window for multi-channel visualization
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="Real-Time Multi-Channel EEG Simulation")
        self.win.resize(1200, 800)
        self.plot = self.win.addPlot(title="Multi-Channel EEG")
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude', units='µV')
        # Create a curve for each channel (using a different color)
        self.curves = []
        for ch in range(self.n_channels):
            # Using pg.intColor for distinct colors
            curve = self.plot.plot(pen=pg.mkPen(color=pg.intColor(ch, hues=self.n_channels), width=1))
            self.curves.append(curve)

        # Timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int((self.chunk_size / self.srate) * 1000))
        self.timer.timeout.connect(self.update_plot)

    def update_plot(self):
        start = self.current_idx
        end = start + self.chunk_size
        if end > self.n_samples:
            self.timer.stop()
            print("End of EEG data reached.")
            return

        # Update common time axis
        last_time = self.x_data[-1] if self.x_data else 0.0
        t_values = np.linspace(last_time, last_time + (self.chunk_size / self.srate),
                               num=self.chunk_size, endpoint=False)
        self.x_data.extend(t_values.tolist())

        # Update each channel's data and its curve
        for ch in range(self.n_channels):
            new_data = self.eeg_data[ch, start:end]
            # Add vertical offset per channel
            new_data_offset = new_data + ch * self.offset
            self.y_data[ch].extend(new_data_offset.tolist())
            self.curves[ch].setData(self.x_data, self.y_data[ch])

        self.current_idx = end

        # For classification: extract features from the full epoch window (all channels)
        epoch_sample_count = int(self.epoch_length * self.srate)
        if end >= epoch_sample_count:
            epoch_data = self.eeg_data[:, end - epoch_sample_count: end]
            features = extract_features_from_epoch(epoch_data, self.srate).reshape(1, -1)
            if self.model is not None:
                pred = self.model.predict(features)[0]
                print(f"Predicted class for current epoch (last {self.epoch_length}s): {pred}")
            else:
                print("No model available: skipping prediction.")

    def start(self):
        print("\nStarting real-time Multi-Channel EEG simulation...\n")
        self.win.show()
        self.timer.start()
        self.app.exec_()


# =============================================================================
# 6. 3D Visualization: Scalp Topography (Updated)
# =============================================================================
def get_sensor_coordinates(raw):
    """
    Retrieve 3D positions and labels for EEG channels.
    If no montage is set or if the channel names are generic,
    assign a standard montage based on channel count.
    """
    n_channels = len(raw.ch_names)
    if n_channels == 128:
        # You can choose a 128-channel montage (here using GSN-HydroCel-128)
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    else:
        montage = mne.channels.make_standard_montage('standard_1020')
    # If channel names are generic (e.g. "ch_0", "ch_1", ...), rename them using montage names.
    if all(name.startswith("ch_") for name in raw.ch_names):
        mapping = {old: new for old, new in zip(raw.ch_names, montage.ch_names)}
        raw.rename_channels(mapping)
    raw.set_montage(montage)
    sensor_pos = []
    sensor_labels = []
    for dig in montage.dig:
        if dig['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH:
            sensor_pos.append(dig['r'])
            sensor_labels.append(dig['ch_name'])
    # Convert list to numpy array and ensure it is 2D
    sensor_pos = np.array(sensor_pos)
    if sensor_pos.ndim == 1:
        sensor_pos = np.vstack(sensor_pos)
    return sensor_pos, sensor_labels


def plot_eeg_topography_3d(sensor_values, sensor_pos, sensor_labels, title="EEG 3D Topography"):
    """
    Plot a 3D scatter of sensor values using Plotly.

    Parameters:
      sensor_values: 1D array (one value per channel)
      sensor_pos   : 2D array of sensor positions (n_channels x 3)
      sensor_labels: list of sensor names
    """
    x, y, z = sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2]
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=8,
            color=sensor_values,
            colorscale='Viridis',
            colorbar=dict(title="Amplitude")
        ),
        text=sensor_labels,
        hoverinfo='text'
    )
    layout = go.Layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def visualize_latest_epoch_topography(raw, epoch_length=2.0):
    """
    Compute and display the 3D scalp topography for the latest epoch.
    Averages the data over the last epoch_length seconds.
    """
    srate = raw.info['sfreq']
    n_samples = int(epoch_length * srate)
    data = raw.get_data()  # shape: (n_channels, total_samples)
    epoch_data = data[:, -n_samples:] if data.shape[1] >= n_samples else data
    sensor_avg = np.mean(epoch_data, axis=1)
    sensor_pos, sensor_labels = get_sensor_coordinates(raw)
    plot_eeg_topography_3d(sensor_avg, sensor_pos, sensor_labels, title="Latest Epoch Topography")


# =============================================================================
# 7. Main: End-to-End Pipeline
# =============================================================================
def main():
    # -----------------------
    # Parameters and file path
    # -----------------------
    file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'  # <-- CHANGE THIS

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
        epochs = epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0)
        X, y = extract_features(epochs)
        model = train_classifier(X, y)
    else:
        print("No event information found; skipping epoch-based training.")

    # -----------------------
    # Real-Time Simulation (Multi-Channel)
    # -----------------------
    data = raw.get_data()  # shape: (n_channels, n_samples)
    srate = raw.info['sfreq']
    chunk_size = int(srate * 0.1)  # update ~ every 0.1 sec
    simulator = RealTimeMultiChannelEEGSimulator(
        eeg_data=data,
        srate=srate,
        model=model,
        chunk_size=chunk_size,
        epoch_length=2.0,
        offset=100.0  # adjust vertical offset as needed
    )

    # Schedule the 3D topography visualization to run after 5 seconds.
    QtCore.QTimer.singleShot(5000, lambda: visualize_latest_epoch_topography(raw, epoch_length=2.0))

    # Start the multi-channel simulation (runs on the main thread)
    simulator.start()


if __name__ == "__main__":
    main()