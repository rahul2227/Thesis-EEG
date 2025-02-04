import sys
import os
import time
import numpy as np
import scipy.io as sio
import mne
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------
# 1. LOADING & PREPROCESSING
# ----------------------------------------------------------------
def load_mat_eeg(file_path):
    """
    Load raw EEG data from a .mat file.
    Returns an mne.Raw object and the event structs (if available).
    """
    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    EEG = mat_data['EEG']

    eeg_signals = EEG.data
    if eeg_signals.shape[0] < eeg_signals.shape[1]:
        pass  # shape = (n_channels, n_samples)
    else:
        # shape = (n_samples, n_channels), so transpose
        eeg_signals = eeg_signals.T

    srate = float(EEG.srate)
    ch_names = [f"ch_{i}" for i in range(eeg_signals.shape[0])]
    ch_types = ["eeg"] * eeg_signals.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

    raw = mne.io.RawArray(eeg_signals, info, verbose=False)
    events_struct = getattr(EEG, 'event', None)  # If it exists
    print(f"Loaded raw data from {os.path.basename(file_path)} "
          f"with shape={eeg_signals.shape}, sfreq={srate}")
    return raw, events_struct


def preprocess_raw(raw, bandpass=(1., 40.), notch=50.0):
    """
    Minimal preprocess: bandpass filter + optional notch
    """
    print("Preprocessing: bandpass filtering...")
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
    if notch is not None:
        print(f"Preprocessing: notch filtering at {notch} Hz...")
        raw.notch_filter(freqs=[notch], verbose=False)
    return raw


# ----------------------------------------------------------------
# 2. EPOCHING & FEATURE EXTRACTION (OPTIONAL)
# ----------------------------------------------------------------
def extract_real_events(events_struct):
    """
    Convert your EEG.event structs to MNE event array.
    Returns (mne_events, event_id_map).
    """
    if events_struct is None:
        return None, None

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
    return mne_events, event_id_map


def epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0):
    """
    Create MNE epochs from real events
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
    print(f"Created {len(epochs)} epochs, each {tmax - tmin}s long.")
    return epochs


def extract_features(epochs):
    """
    Extract simple features (mean, std) per channel.
    Return X, y
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs = len(data)
    y = epochs.events[:, 2]   # real label from the event code

    features = []
    for i in range(n_epochs):
        epoch_data = data[i]
        ch_means = epoch_data.mean(axis=1)
        ch_stds = epoch_data.std(axis=1)
        feat_vec = np.concatenate([ch_means, ch_stds])
        features.append(feat_vec)
    X = np.array(features)
    return X, y


# ----------------------------------------------------------------
# 3. TRAIN A SIMPLE MODEL
# ----------------------------------------------------------------
def train_classifier(X, y):
    """
    Train an SVM or any simple model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classification test accuracy: {acc:.2f}")
    return clf


# ----------------------------------------------------------------
# 4. REAL-TIME SIMULATION (CHUNK-BASED)
# ----------------------------------------------------------------
class RealTimeEEGSimulator:
    """
    A chunk-based 'real-time' simulator using PyQtGraph.
    Optionally, we can do model predictions each time we fetch a chunk.
    """
    def __init__(self, eeg_data, srate, model=None, chunk_size=50, channel_index=0):
        """
        eeg_data   : np.ndarray of shape (n_channels, n_samples)
        srate      : sampling rate
        model      : optional trained model. If provided, we can run .predict() on each chunk
        chunk_size : how many samples to read each "tick"
        """
        self.eeg_data = eeg_data
        self.srate = srate
        self.model = model
        self.chunk_size = chunk_size
        self.channel_index = channel_index

        self.current_idx = 0
        self.n_channels, self.n_samples = eeg_data.shape

        # Set up Qt application & window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="EEG Real-Time Simulation")
        self.win.resize(800, 500)

        self.plot = self.win.addPlot(title=f"Simulated EEG Channel {channel_index}")
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude', units='µV')
        self.curve = self.plot.plot(pen='y')

        self.x_data = []
        self.y_data = []

        # A QTimer that calls update_plot at intervals matching chunk_size
        ms_per_chunk = (chunk_size / self.srate) * 1000.0
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(ms_per_chunk))
        self.timer.timeout.connect(self.update_plot)

    def update_plot(self):
        start = self.current_idx
        end = start + self.chunk_size
        if end > self.n_samples:
            self.timer.stop()
            print("End of data reached.")
            return

        # fetch chunk of data from one channel
        chunk = self.eeg_data[self.channel_index, start:end]

        # build the time axis for the chunk
        if len(self.x_data) == 0:
            t_start = 0.0
        else:
            t_start = self.x_data[-1] + 1.0/self.srate

        t_values = [t_start + i*(1.0/self.srate) for i in range(self.chunk_size)]

        # Store in big arrays for plotting
        self.x_data.extend(t_values)
        self.y_data.extend(chunk)

        # Update PyQtGraph curve
        self.curve.setData(self.x_data, self.y_data)

        # If we have a model, we can do a tiny feature extraction here
        if self.model is not None:
            # For example, compute mean/std of the chunk
            chunk_mean = np.mean(chunk)
            chunk_std = np.std(chunk)
            feature_vec = np.array([chunk_mean, chunk_std]).reshape(1, -1)
            pred_class = self.model.predict(feature_vec)[0]
            print(f"Chunk {start}:{end} → Predicted class: {pred_class}")

        self.current_idx = end

    def start(self):
        self.timer.start()
        QtWidgets.QApplication.instance().exec_()


# ----------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------
def main():
    # ------- Load & Preprocess -------
    file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'
    raw, events_struct = load_mat_eeg(file_path)
    raw = preprocess_raw(raw, (1., 40.), 50.0)

    # ------- (Optional) Train a model on epoch-based features -------
    # If you want to do classification on each chunk, you can train a model using epochs.
    # Or skip if you just want to see real-time signals.

    # This part is optional demonstration:
    mne_events, event_id_map = extract_real_events(events_struct)
    if mne_events is not None:
        epochs = epoch_data(raw, mne_events, event_id_map, 0.0, 2.0)
        X, y = extract_features(epochs)
        clf = train_classifier(X, y)
    else:
        clf = None
        print("No events found. Will just show raw data without predictions.")

    # ------- Real-Time Simulation of raw signals -------
    # We take the raw data as (n_channels, n_samples)
    data = raw.get_data()
    srate = raw.info['sfreq']

    # Create a simulator with chunk_size=50 samples (example)
    simulator = RealTimeEEGSimulator(
        eeg_data=data,
        srate=srate,
        model=clf,            # pass your model if you want predictions
        chunk_size=50,
        channel_index=0       # which channel to visualize
    )

    print("\nStarting real-time simulation...\n")
    simulator.start()


if __name__ == "__main__":
    main()