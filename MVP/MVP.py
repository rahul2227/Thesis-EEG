import os
import time
import numpy as np
import scipy.io as sio
import mne
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_mat_eeg(file_path):
    """
    Load raw EEG data from a .mat file.
    Returns an mne.Raw plus the event array or object from `EEG.event`.
    """
    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    EEG = mat_data['EEG']

    # Convert data to channels x samples if needed
    eeg_signals = EEG.data
    if eeg_signals.shape[0] < eeg_signals.shape[1]:
        pass  # shape is already (n_channels, n_samples)
    else:
        # shape is (n_samples, n_channels), so transpose
        eeg_signals = eeg_signals.T

    srate = float(EEG.srate)
    ch_names = [f"ch_{i}" for i in range(eeg_signals.shape[0])]
    ch_types = ["eeg"] * eeg_signals.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

    raw = mne.io.RawArray(eeg_signals, info, verbose=False)
    events_struct = EEG.event  # This is presumably an array of event structs

    print(f"Loaded raw data from {os.path.basename(file_path)} "
          f"with shape={eeg_signals.shape}, sfreq={srate}")
    return raw, events_struct


def extract_real_events(events_struct):
    """
    Convert your MATLAB event structures (EEG.event) to an MNE-compatible
    'events' array of shape (n_events, 3). Typically:
      events[i] = [sample_index, 0, event_id]

    Also build a dictionary mapping {event_code: int_label}.

    Adjust logic to match your data. Example:
      events_struct[0].latency   => sample index
      events_struct[0].type      => condition code or label (string or int)
    """

    # If 'events_struct' is an array of mat_struct,
    # each might have fields: .latency, .type
    # We must unify them into MNE standard => events array
    mne_events = []
    event_id_map = {}
    current_code = 1  # we start labeling distinct events from 1 upward

    for e in events_struct:
        latency = int(e.latency)  # sample index
        etype = e.type  # could be string or int

        # If etype is a string like 'stim1', 'stim2', we map each unique to a code
        # If etype is already numeric, just cast to int
        if isinstance(etype, str):
            if etype not in event_id_map:
                event_id_map[etype] = current_code
                current_code += 1
            code = event_id_map[etype]
        else:
            # assume numeric
            code = int(etype)
            # optionally track unique codes in event_id_map
            if code not in event_id_map.values():
                event_id_map[f"code_{code}"] = code

        # MNE wants an array of [sample, 0, code]
        mne_events.append([latency, 0, code])

    mne_events = np.array(mne_events, dtype=int)

    # If we want a final event_id dict in MNE style, that is:
    # { 'stim1': 1, 'stim2': 2, ... }
    # we already built event_id_map above.
    return mne_events, event_id_map


def preprocess_raw(raw, bandpass=(1., 40.), notch=50.0):
    print("Preprocessing: bandpass filtering...")
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
    if notch is not None:
        print(f"Preprocessing: notch filtering at {notch} Hz...")
        raw.notch_filter(freqs=[notch], verbose=False)
    return raw


def epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0):
    """
    Use the real events + event_id_map to create epochs.
    We assume 'event_id_map' is something like {'stim1': 1, 'stim2': 2, ...}.

    If your events are labeled 1,2,3 for different conditions,
    you can pass that directly as `event_id_map = {'1':1,'2':2,'3':3}` or similar.
    """
    print("Epoching data from real events...")
    # Convert event_id_map to an integer-coded dict if it isn't already
    # e.g., if event_id_map = {'stim1': 1, 'stim2': 2}

    # If your event_id_map is something like {'code_1':1, 'code_3':3, ...} that is fine.
    # Just pass it directly to MNE.
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
    Extract basic features (e.g., mean & std). Also, figure out real labels from the epoch metadata.

    If your MNE event_id has keys like {'stim1': 1, 'stim2': 2}, then
    each epoch in `epochs.events` has an integer code in column 2, e.g. 1 or 2.
    We can store that in y.
    """
    print("Extracting features from epochs...")
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    # Real labels come from epochs.events[:, 2]
    y = epochs.events[:, 2]  # shape (n_epochs, )

    # Example: compute (mean, std) per channel, then flatten
    features = []
    for i in range(n_epochs):
        epoch_data = data[i, :, :]
        ch_means = epoch_data.mean(axis=1)
        ch_stds = epoch_data.std(axis=1)
        feat_vec = np.concatenate([ch_means, ch_stds])
        features.append(feat_vec)

    X = np.array(features)  # shape: (n_epochs, 2*n_channels)
    return X, y


def train_classifier(X, y):
    print("Training classifier with real labels...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classification test accuracy: {acc:.2f}")
    return clf


def simulate_realtime(clf, epochs):
    """
    Minimal demonstration of 'real-time' inference on real-labeled epochs.
    We will iterate through each epoch, extract the same features on the fly,
    and call .predict().
    """
    print("\nSimulating real-time inference...\n")
    data = epochs.get_data()
    event_codes = epochs.events[:, 2]  # real labels
    n_epochs = len(data)

    for i in range(n_epochs):
        epoch_data = data[i, :, :]
        ch_means = epoch_data.mean(axis=1)
        ch_stds = epoch_data.std(axis=1)
        feature_vec = np.concatenate([ch_means, ch_stds]).reshape(1, -1)

        pred_class = clf.predict(feature_vec)[0]
        true_class = event_codes[i]
        print(f"Epoch {i + 1}/{n_epochs} → Predicted: {pred_class}, True: {true_class}")

        time.sleep(0.3)  # Sleep to simulate real-time arrival


def main():
    file_path = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR/Raw data/YAC/YAC_NR1_EEG.mat'
    raw, events_struct = load_mat_eeg(file_path)

    raw = preprocess_raw(raw, bandpass=(1., 40.), notch=50.0)

    # Convert from your EEG.event struct to MNE events array
    mne_events, event_id_map = extract_real_events(events_struct)

    # Create epochs from real events
    epochs = epoch_data(raw, mne_events, event_id_map, tmin=0.0, tmax=2.0)

    # Extract features + real labels from the epochs
    X, y = extract_features(epochs)

    # Train a classifier
    clf = train_classifier(X, y)

    # Simulate real-time streaming of epochs
    simulate_realtime(clf, epochs)


if __name__ == "__main__":
    main()