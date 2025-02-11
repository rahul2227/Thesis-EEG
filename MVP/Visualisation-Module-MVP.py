#!/usr/bin/env python3
"""
MVP EEG Application with 3D Topography Visualization and Multi-Channel Real-Time Display,
with Real-Time 3D Simulation in a Separate Window

Features:
- Data Loading, Preprocessing, Epoching, Feature Extraction, Modeling (SVM)
- Real-time Multi-Channel Visualization using PyQtGraph with a grouped colorbar legend
- A separate 3D simulation window that updates periodically (using Plotly) showing a 3D surface
  of the latest epoch data (similar to the plot_3dSurface_and_heatmap style)
- 3D scalp topography visualization (via a separate function)

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
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

# Import QWebEngineView for displaying Plotly HTML content
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    raise ImportError(
        "PyQt5.QtWebEngineWidgets is required for the 3D simulation window. Please install PyQtWebEngine.")

# Plotly for 3D visualization
import plotly.graph_objects as go
import plotly.io as pio  # for generating HTML

# pio.kaleido.scope.default_format = "png"  # optional: set default image format


# =============================================================================
# 1. Data Loading & Preprocessing
# =============================================================================
def load_mat_eeg(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    EEG = mat_data['EEG']

    # Ensure data is in shape (n_channels, n_samples)
    eeg_signals = EEG.data
    if eeg_signals.shape[0] >= eeg_signals.shape[1]:
        eeg_signals = eeg_signals.T

    srate = float(EEG.srate)
    n_channels = eeg_signals.shape[0]
    ch_names = [f"ch_{i}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_signals, info, verbose=False)
    events_struct = getattr(EEG, 'event', None)
    print(f"Loaded '{os.path.basename(file_path)}': shape={eeg_signals.shape}, sfreq={srate} Hz")
    return raw, events_struct


def preprocess_raw(raw, bandpass=(1., 40.), notch=50.0):
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
    epochs = mne.Epochs(raw, events=mne_events, event_id=event_id_map,
                        tmin=tmin, tmax=tmax, baseline=None, preload=True,
                        verbose=False)
    print(f"Created {len(epochs)} epochs, each {tmax - tmin:.1f}s long.")
    return epochs


# =============================================================================
# 3. Feature Extraction
# =============================================================================
def compute_band_power(signal, srate, band):
    freqs, psd = welch(signal, fs=srate, nperseg=min(256, len(signal)))
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
    return band_power


def extract_features_from_epoch(epoch_data, srate):
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
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    srate = epochs.info['sfreq']
    X = np.array([extract_features_from_epoch(epoch, srate) for epoch in data])
    y = epochs.events[:, 2]
    print(f"Extracted features for {len(data)} epochs; each feature vector has size {X.shape[1]}.")
    return X, y


# =============================================================================
# 4. Modeling
# =============================================================================
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained SVM classifier; Test Accuracy = {acc:.2f}")
    return clf
# =============================================================================
# Helper: Create a Grouped Colorbar Widget with Channel Ranges
# =============================================================================
def create_grouped_colorbar_widget(n_channels, channel_names, group_size=10, patch_width=20, patch_height=20):
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    header = QtWidgets.QLabel("Channel Legend (Groups)")
    header.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(header)
    n_groups = (n_channels + group_size - 1) // group_size
    for group in range(n_groups):
        start_idx = group * group_size
        end_idx = min((group + 1) * group_size - 1, n_channels - 1)
        rep_idx = (start_idx + end_idx) // 2
        rep_color = pg.intColor(rep_idx, hues=n_channels)
        h_layout = QtWidgets.QHBoxLayout()
        patch_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(patch_width, patch_height)
        pixmap.fill(rep_color)
        patch_label.setPixmap(pixmap)
        h_layout.addWidget(patch_label)
        group_label_text = f"{channel_names[start_idx]} - {channel_names[end_idx]}"
        channel_label = QtWidgets.QLabel(group_label_text)
        h_layout.addWidget(channel_label)
        layout.addLayout(h_layout)
    layout.addStretch()
    return widget


# =============================================================================
# 5. Real-Time Simulation and Visualization (Multi-Channel) with Separate 3D Simulation Window
# =============================================================================
class RealTimeMultiChannelEEGSimulator:
    def __init__(self, eeg_data, srate, channel_names=None, model=None, chunk_size=50,
                 epoch_length=2.0, offset=100.0, group_size=10, update_3d_interval=5000):
        self.eeg_data = eeg_data
        self.srate = srate
        self.model = model
        self.chunk_size = chunk_size
        self.epoch_length = epoch_length
        self.offset = offset  # vertical offset per channel
        self.group_size = group_size
        self.update_3d_interval = update_3d_interval  # in milliseconds

        self.n_channels, self.n_samples = eeg_data.shape
        if channel_names is None:
            self.channel_names = [f"ch_{i}" for i in range(self.n_channels)]
        else:
            self.channel_names = channel_names

        self.current_idx = 0
        self.x_data = []  # common time axis
        self.y_data = [[] for _ in range(self.n_channels)]

        # --- Setup multi-channel simulation plot ---
        self.win = pg.GraphicsLayoutWidget(title="Real-Time Multi-Channel EEG Simulation")
        self.win.resize(1200, 800)
        self.plot = self.win.addPlot(title="Multi-Channel EEG")
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude', units='µV')
        self.curves = []
        for ch in range(self.n_channels):
            pen = pg.mkPen(color=pg.intColor(ch, hues=self.n_channels), width=1)
            curve = self.plot.plot(pen=pen)
            self.curves.append(curve)

        # Create main widget layout with simulation plot and grouped colorbar
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.main_layout.addWidget(self.win)
        self.colorbar_widget = create_grouped_colorbar_widget(self.n_channels, self.channel_names,
                                                              group_size=self.group_size)
        self.main_layout.addWidget(self.colorbar_widget)

        # Timer for updating the simulation plot
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int((self.chunk_size / self.srate) * 1000))
        self.timer.timeout.connect(self.update_plot)

        # --- Setup 3D simulation window using QWebEngineView ---
        self.webview_window = QWebEngineView()
        self.webview_window.setWindowTitle("Real-Time 3D EEG Simulation")
        self.webview_window.resize(800, 600)

        # Timer for updating the 3D simulation window
        self.timer_3d = QtCore.QTimer()
        self.timer_3d.setInterval(self.update_3d_interval)
        self.timer_3d.timeout.connect(self.update_3d_simulation)

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

        # Update each channel's curve with vertical offset
        for ch in range(self.n_channels):
            new_data = self.eeg_data[ch, start:end]
            new_data_offset = new_data + ch * self.offset
            self.y_data[ch].extend(new_data_offset.tolist())
            self.curves[ch].setData(self.x_data, self.y_data[ch])

        self.current_idx = end

        # (Optional) Run classification on the full epoch window (all channels)
        epoch_sample_count = int(self.epoch_length * self.srate)
        if end >= epoch_sample_count:
            epoch_data = self.eeg_data[:, end - epoch_sample_count: end]
            features = extract_features_from_epoch(epoch_data, self.srate).reshape(1, -1)
            if self.model is not None:
                pred = self.model.predict(features)[0]
                print(f"Predicted class for current epoch (last {self.epoch_length}s): {pred}")
            else:
                print("No model available: skipping prediction.")

    def update_3d_simulation(self):
        """
        Update the 3D simulation window with a surface plot of the latest epoch.
        This function takes the last epoch window (all channels) and builds a 3D surface
        visualization similar to the 'plot_3dSurface_and_heatmap' style.
        """
        epoch_sample_count = int(self.epoch_length * self.srate)
        if self.current_idx < epoch_sample_count:
            return  # Not enough data yet
        epoch_data = self.eeg_data[:, self.current_idx - epoch_sample_count: self.current_idx]

        # Define x (time/sample indices) and y (channel indices) axes
        x_vals = list(range(epoch_sample_count))
        y_vals = list(range(self.n_channels))
        z_data = epoch_data.tolist()  # 2D list: one list per channel

        # Build the Plotly surface plot
        fig = go.Figure(data=[go.Surface(z=z_data, x=x_vals, y=y_vals, colorscale='Bluered')])
        fig.update_layout(title="Real-Time 3D EEG Simulation",
                          scene=dict(xaxis_title="Time (samples)",
                                     yaxis_title="Channel",
                                     zaxis_title="Sensor Value"))

        # Generate full HTML for the Plotly figure, loading Plotly.js from the CDN
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Update the QWebEngineView with the new HTML content and a base URL
        self.webview_window.setHtml(html_str, QtCore.QUrl("about:blank"))

    def start(self):
        print("\nStarting real-time Multi-Channel EEG simulation with separate 3D simulation window...\n")
        self.main_widget.show()
        self.webview_window.show()  # show the 3D simulation window
        self.timer.start()
        self.timer_3d.start()
        QtWidgets.QApplication.instance().exec_()


# =============================================================================
# 6. 3D Visualization: Scalp Topography (Unchanged)
# =============================================================================
def get_sensor_coordinates(raw):
    n_channels = len(raw.ch_names)
    if n_channels == 128:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    else:
        montage = mne.channels.make_standard_montage('standard_1020')
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
    sensor_pos = np.array(sensor_pos)
    if sensor_pos.ndim == 1:
        sensor_pos = np.vstack(sensor_pos)
    return sensor_pos, sensor_labels


def plot_eeg_topography_3d(sensor_values, sensor_pos, sensor_labels, title="EEG 3D Topography"):
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
    srate = raw.info['sfreq']
    n_samples = int(epoch_length * srate)
    data = raw.get_data()
    epoch_data = data[:, -n_samples:] if data.shape[1] >= n_samples else data
    sensor_avg = np.mean(epoch_data, axis=1)
    sensor_pos, sensor_labels = get_sensor_coordinates(raw)
    plot_eeg_topography_3d(sensor_avg, sensor_pos, sensor_labels, title="Latest Epoch Topography")


# =============================================================================
# 7. Main: End-to-End Pipeline
# =============================================================================
def main():
    file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'  # <-- CHANGE THIS
    raw, events_struct = load_mat_eeg(file_path)
    raw = preprocess_raw(raw, bandpass=(1., 40.), notch=50.0)
    model = None
    if events_struct is not None:
        mne_events, event_id_map = extract_events_from_struct(events_struct)
        epochs = epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0)
        X, y = extract_features(epochs)
        model = train_classifier(X, y)
    else:
        print("No event information found; skipping epoch-based training.")
    data = raw.get_data()
    srate = raw.info['sfreq']
    chunk_size = int(srate * 0.1)
    simulator = RealTimeMultiChannelEEGSimulator(
        eeg_data=data,
        srate=srate,
        channel_names=raw.ch_names,
        model=model,
        chunk_size=chunk_size,
        epoch_length=2.0,
        offset=100.0,
        group_size=10,
        update_3d_interval=5000  # update the 3D view every 5 seconds
    )
    QtCore.QTimer.singleShot(5000, lambda: visualize_latest_epoch_topography(raw, epoch_length=2.0))
    simulator.start()


if __name__ == "__main__":
    main()