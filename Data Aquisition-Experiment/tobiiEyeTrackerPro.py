import csv
import os
import numpy as np
import time
import collections

try:
    import tobii_research as tr
except ImportError:
    tr = None
    print("Unable to import tobii_research. Eye tracking functionality will not be available.")

from math import isnan, pi, sqrt


class TobiiEyeTracker:

    RAW_HEADER = [
        "system_ts", "device_ts",
        "left_x", "left_y",
        "right_x", "right_y",
        "left_pupil", "right_pupil",
        "valid_left", "valid_right"
    ]

    def __init__(self, save_dir: str | None = None):
        self.et = None
        self.save_dir = save_dir
        self.raw_fp = None
        self._raw_writer = None
        self.raw_data = []
        self.pupil_areas = collections.deque()
        self.eye_validity = collections.deque()
        self.gaze_points = collections.deque()
        self.fixations = collections.deque()
        self.fixation_start_time = None
        self.fixation_points = []
        self.FIXATION_DURATION_THRESHOLD = 100
        self.FIXATION_DISPERSION_THRESHOLD = 0.01
        self.PUPILE_CONVERSION = 10
        self.recording = False

    # TODO:  Write a flag to return when I tracker is found, if not found then return false
    def connect(self):
        while True:
            print("[Log] Searching for eye tracker...")
            if tr is None:
                raise RuntimeError("[Log] Tobii research is not available. Will not connect.")
            eyetrackers = tr.find_all_eyetrackers()
            if eyetrackers:
                self.et = eyetrackers[0]
                print("[Log] Found eye tracker: " + self.et.model)
                break
            time.sleep(1)

    def start_recording(self):
        if not self.et:
            raise RuntimeError("Call connect() first.")
        if not self.recording:
            self._open_raw_log()
            self.et.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)
            self.et.subscribe_to(tr.EYETRACKER_EYE_OPENNESS_DATA, self.eye_openness_callback, as_dictionary=True)
            self.recording = True
            print("[Log] Eye-tracking recording started...")

    # ------------------------------------------------------------------
    # RAW Loggers
    # ------------------------------------------------------------------
    def _open_raw_log(self):
        if self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        raw_path = os.path.join(self.save_dir,
                                f"raw_gaze_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self._raw_fp = open(raw_path, "w", newline="")
        self._raw_writer = csv.writer(self._raw_fp)
        self._raw_writer.writerow(self.RAW_HEADER)
        print(f"[Log] Raw gaze stream → {raw_path}")

    def _close_raw_log(self):
        if self._raw_fp:
            self._raw_fp.flush()
            self._raw_fp.close()
            self._raw_fp = None
            self._raw_writer = None

    def gaze_data_callback(self, gaze_data):
        timestamp = time.time()

        # time stamps for raw data
        ts_system = gaze_data["system_time_stamp"]
        ts_device = gaze_data["device_time_stamp"]

        left_pupil_diameter = gaze_data.get('left_pupil_diameter', None)
        right_pupil_diameter = gaze_data.get('right_pupil_diameter', None)
        left_gaze_point = gaze_data.get('left_gaze_point_on_display_area', [None, None])
        right_gaze_point = gaze_data.get('right_gaze_point_on_display_area', [None, None])
        left_gaze_validity = gaze_data.get('left_gaze_point_validity', 0)
        right_gaze_validity = gaze_data.get('right_gaze_point_validity', 0)

        # logging raw samples
        if self._raw_writer:
            self._raw_writer.writerow([
                ts_system, ts_device,
                left_pupil_diameter, right_pupil_diameter,
                left_gaze_point[0], left_gaze_point[1],
                right_gaze_point[0], right_gaze_point[1],
                left_gaze_validity, right_gaze_validity
            ])

            self.raw_data.append([
                ts_system, ts_device,
                left_pupil_diameter, right_pupil_diameter,
                left_gaze_point[0], left_gaze_point[1],
                right_gaze_point[0], right_gaze_point[1],
                left_gaze_validity, right_gaze_validity
            ])

        pupil_areas = []
        if left_pupil_diameter and not isnan(left_pupil_diameter):
            radius = left_pupil_diameter / 2.0
            area_mm2 = pi * (radius ** 2)  # Area in mm²
            area_um2 = area_mm2 * 1_000_00  # Convert to µm²
            pupil_areas.append(area_um2)
        if right_pupil_diameter and not isnan(right_pupil_diameter):
            radius = right_pupil_diameter / 2.0
            area_mm2 = pi * (radius ** 2)  # Area in mm²
            area_um2 = area_mm2 * 1_000_000  # Convert to µm²
            pupil_areas.append(area_um2)
        if pupil_areas:
            average_pupil_area = sum(pupil_areas) / len(pupil_areas)
            average_pupil_area = average_pupil_area / self.PUPILE_CONVERSION
            self.pupil_areas.append((timestamp, average_pupil_area))

        # Gaze Points Processing
        gaze_points = []
        if left_gaze_validity == 1 and left_gaze_point and not any(isnan(coord) for coord in left_gaze_point):
            gaze_points.append(left_gaze_point)
        if right_gaze_validity == 1 and right_gaze_point and not any(isnan(coord) for coord in right_gaze_point):
            gaze_points.append(right_gaze_point)
        if gaze_points:
            avg_x = sum(point[0] for point in gaze_points) / len(gaze_points)
            avg_y = sum(point[1] for point in gaze_points) / len(gaze_points)
            self.gaze_points.append((timestamp, (avg_x, avg_y)))
            self.detect_fixation(avg_x, avg_y, timestamp)

    def eye_openness_callback(self, eye_openness_data):
        timestamp = time.time()
        left_eye_validity = eye_openness_data.get("left_eye_validity", 0)
        left_eye_openness_value = eye_openness_data.get("left_eye_openness_value", 0)
        right_eye_validity = eye_openness_data.get("right_eye_validity", 0)
        right_eye_openness_value = eye_openness_data.get("right_eye_openness_value", 0)

        # Eye is open and valid if at least one eye is open
        is_eye_valid = False
        if left_eye_validity == 1 and left_eye_openness_value > 0:
            is_eye_valid = True
        if right_eye_validity == 1 and right_eye_openness_value > 0:
            is_eye_valid = True
        self.eye_validity.append((timestamp, is_eye_valid))

    def detect_fixation(self, avg_x, avg_y, timestamp):
        if self.fixation_start_time is None:
            self.fixation_start_time = timestamp
            self.fixation_points = [(avg_x, avg_y)]
        else:
            # Calculate dispersion (max distance between current point and fixation points)
            dispersion = max(
                sqrt((x - avg_x) ** 2 + (y - avg_y) ** 2) for x, y in self.fixation_points
            )
            if dispersion <= self.FIXATION_DISPERSION_THRESHOLD:
                # Continue fixation
                self.fixation_points.append((avg_x, avg_y))
                # Check an if fixation duration threshold is met
                fixation_duration_ms = (timestamp - self.fixation_start_time) * 1000  # Convert to ms
                if fixation_duration_ms >= self.FIXATION_DURATION_THRESHOLD:
                    # Register fixation with timestamp
                    self.fixations.append((timestamp, fixation_duration_ms))
                    # Reset fixation tracking
                    self.fixation_start_time = None
                    self.fixation_points = []
            else:
                # Reset fixation tracking
                self.fixation_start_time = timestamp
                self.fixation_points = [(avg_x, avg_y)]

    def get_mean_pupil_area(self, interval):
        current_time = time.time()
        areas_in_interval = [area for t, area in self.pupil_areas if current_time - t <= interval]
        if areas_in_interval:
            return sum(areas_in_interval) / len(areas_in_interval)
        else:
            return None

    def get_average_gaze_point(self, interval):
        current_time = time.time()
        points_in_interval = [point for t, point in self.gaze_points if current_time - t <= interval]
        if points_in_interval:
            avg_x = sum(point[0] for point in points_in_interval) / len(points_in_interval)
            avg_y = sum(point[1] for point in points_in_interval) / len(points_in_interval)
            return avg_x, avg_y
        else:
            return None, None

    def is_user_looking(self, interval):
        current_time = time.time()
        validity_in_interval = [valid for t, valid in self.eye_validity if current_time - t <= interval]
        if validity_in_interval:
            return any(validity_in_interval)
        else:
            return False  # Assume not looking if no data

    def get_mean_fixation_duration(self, interval):
        current_time = time.time()
        # Fixations are stored as (timestamp, duration_ms)
        fixations_in_interval = [duration for t, duration in self.fixations if current_time - t <= interval]
        if fixations_in_interval:
            mean_fixation_duration_ms = sum(fixations_in_interval) / len(fixations_in_interval)
            mean_fixation_duration_s = mean_fixation_duration_ms / 1000.0  # Convert to seconds
            return mean_fixation_duration_s
        else:
            return None

    def stop_recording(self):
        if self.recording:
            self.et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
            self.et.unsubscribe_from(tr.EYETRACKER_EYE_OPENNESS_DATA, self.eye_openness_callback)
            self.recording = False
            self._close_raw_log()
            print('[LOG] Eye-tracking recording stopped...')


    def save_data_npy(self, directory: str):
        """
        Save eye‑tracking data into two NumPy binary files.

        Parameters
        ----------
        directory : str
            Destination folder. Created if it does not exist.

        Creates
        -------
        raw_data_<timestamp>.npy
            2‑D array containing every raw gaze sample recorded during
            the session. Columns correspond to `RAW_HEADER`.
        processed_data_<timestamp>.npy
            A dict (stored with allow_pickle=True) holding:
                - pupil_areas          → (N, 2)  [timestamp, area]
                - gaze_points          → (N, 3)  [timestamp, x, y]
                - eye_validity         → (N, 2)  [timestamp, bool]
                - fixations            → (N, 2)  [timestamp, duration_ms]
                - aggregate means      → floats for quick inspection
        """
        import os, time

        os.makedirs(directory, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')

        # ---------- RAW ----------
        raw_path = os.path.join(directory, f"raw_data_{ts}.npy")
        np.save(raw_path, np.array(self.raw_data, dtype=object))

        # ---------- PROCESSED ----------
        processed = {}

        if self.pupil_areas:
            pupil_arr = np.array(self.pupil_areas, dtype=float)
            processed['pupil_areas'] = pupil_arr
            processed['mean_pupil_area'] = float(np.mean(pupil_arr[:, 1]))

        if self.gaze_points:
            gaze_arr = np.array([(t, xy[0], xy[1]) for t, xy in self.gaze_points], dtype=float)
            processed['gaze_points'] = gaze_arr
            processed['mean_gaze_x'] = float(np.mean(gaze_arr[:, 1]))
            processed['mean_gaze_y'] = float(np.mean(gaze_arr[:, 2]))

        if self.eye_validity:
            eye_arr = np.array(self.eye_validity, dtype=object)
            processed['eye_validity'] = eye_arr
            processed['valid_ratio'] = float(np.mean(eye_arr[:, 1].astype(float)))

        if self.fixations:
            fix_arr = np.array(self.fixations, dtype=float)
            processed['fixations'] = fix_arr
            processed['mean_fixation_duration_ms'] = float(np.mean(fix_arr[:, 1]))

        proc_path = os.path.join(directory, f"processed_data_{ts}.npy")
        np.save(proc_path, processed, allow_pickle=True)

        print(f"[Log] Saved RAW to {raw_path}")
        print(f"[Log] Saved PROCESSED to {proc_path}")


    def cleanup_old_data(self, interval):
        current_time = time.time()
        while self.pupil_areas and current_time - self.pupil_areas[0][0] > interval:
            self.pupil_areas.popleft()
        while self.eye_validity and current_time - self.eye_validity[0][0] > interval:
            self.eye_validity.popleft()
        while self.gaze_points and current_time - self.gaze_points[0][0] > interval:
            self.gaze_points.popleft()
        while self.fixations and current_time - self.fixations[0][0] > interval:
            self.fixations.popleft()


if __name__ == "__main__":
    tobii_tracker = TobiiEyeTracker()
    tobii_tracker.connect()
    # Waiting for connection stabilization
    time.sleep(2)
    try:
        tobii_tracker.start_recording()
        interval = 1  # seconds
        while True:
            time.sleep(interval)
            mean_pupil_area = tobii_tracker.get_mean_pupil_area(interval)  # in µm²
            mean_fixation_duration = tobii_tracker.get_mean_fixation_duration(interval)  # in seconds
            user_looking = tobii_tracker.is_user_looking(interval)
            avg_x, avg_y = tobii_tracker.get_average_gaze_point(interval)
            if user_looking:
                print(f"\n[Data over last {interval} second(s)]:")
                if mean_pupil_area is not None:
                    print(f"Mean Pupil Area: {mean_pupil_area:.2f} µm²")
                else:
                    print("No valid pupil area data.")
                if mean_fixation_duration is not None:
                    print(f"Mean Fixation Duration: {mean_fixation_duration:.6f} s")
                else:
                    print("No fixation data.")
                if avg_x is not None and avg_y is not None:
                    print(f"Average Gaze Point: x = {avg_x:.4f}, y = {avg_y:.4f}")
                else:
                    print("No valid gaze point data.")
            else:
                print("\n[Info] User is not looking at the eye tracker.")
            # Clean up old data to prevent memory overflow
            tobii_tracker.cleanup_old_data(interval)
    except KeyboardInterrupt:
        print("\nStopping data collection.")
    finally:
        tobii_tracker.stop_recording()
