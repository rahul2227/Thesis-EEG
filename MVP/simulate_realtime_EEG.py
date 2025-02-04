import sys
import numpy as np
import pyqtgraph as pg
# Updated import: also bring in QtWidgets
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

# Import your data reader function
from ASR.data_reader import load_all_eeg_data


class RealTimeEEGSimulator:
    def __init__(self, eeg_data, sampling_rate, chunk_size=50, channel_index=0):
        self.eeg_data = eeg_data  # shape: (n_channels, n_samples)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.channel_index = channel_index

        # Current position in the data
        self.current_idx = 0
        self.n_channels, self.n_samples = self.eeg_data.shape

        # ---- Use QApplication instead of QGuiApplication ----
        self.app = QtWidgets.QApplication(sys.argv)

        self.win = pg.GraphicsLayoutWidget(show=True, title="EEG Real-Time Simulation")
        self.win.resize(800, 500)

        self.plot = self.win.addPlot(title=f"Simulated EEG Channel {channel_index}")
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude', units='µV')

        self.curve = self.plot.plot(pen='y')
        self.x_data = []
        self.y_data = []

        # Timer
        ms_per_chunk = (self.chunk_size / self.sampling_rate) * 1000.0
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

        chunk = self.eeg_data[self.channel_index, start:end]

        if len(self.x_data) == 0:
            t_start = 0.0
        else:
            t_start = self.x_data[-1] + (1.0 / self.sampling_rate)

        t_values = [t_start + i * (1.0 / self.sampling_rate)
                    for i in range(self.chunk_size)]
        self.x_data.extend(t_values)
        self.y_data.extend(chunk)

        self.curve.setData(self.x_data, self.y_data)
        self.current_idx = end

    def start(self):
        self.timer.start()
        QtWidgets.QApplication.instance().exec_()


def main():
    base_dir = '/Users/rahul/PycharmProjects/Thesis-EEG/osfstorage-archive/task1 - NR'
    data_generator = load_all_eeg_data(base_dir)

    try:
        first_data_entry = next(data_generator)
    except StopIteration:
        print(f"No EEG files found under: {base_dir}")
        return

    eeg_signals = first_data_entry['eeg_signals']  # shape: (n_channels, n_samples)
    sampling_rate = first_data_entry['sampling_rate']  # e.g. 500 Hz

    simulator = RealTimeEEGSimulator(
        eeg_data=eeg_signals,
        sampling_rate=sampling_rate,
        chunk_size=50,
        channel_index=0
    )

    print(f"Starting real-time simulation for file: {first_data_entry['file_name']}")
    simulator.start()


if __name__ == "__main__":
    main()