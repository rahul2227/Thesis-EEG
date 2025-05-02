import csv
import os
import time
import collections
from math import isnan, pi, sqrt
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------
#  The main application will set this once, right after logging starts:
#  >>> EXP_START_NS = time.perf_counter_ns()
#  >>> eye.set_exp_start_ns(EXP_START_NS)
# ----------------------------------------------------------------------
EXP_START_NS: Optional[int] = None

try:
    import tobii_research as tr
except ImportError:
    tr = None
    print("Unable to import tobii_research – eye-tracker functions disabled.")


class TobiiEyeTracker:
    """
    Thin wrapper around the Tobii Research SDK that
    1) streams every gaze callback to a CSV in real time,
    2) keeps all raw samples in memory for .npy export,
    3) computes rolling metrics (pupil area, fixations, …).
    """

    RAW_HEADER = [
        "t_rel_s",                       # new: time since EXP_START_NS
        "system_ts", "device_ts",
        "left_x", "left_y",
        "right_x", "right_y",
        "left_pupil", "right_pupil",
        "valid_left", "valid_right"
    ]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def __init__(self, save_dir: str | None = None):
        self.et = None
        self.save_dir = save_dir
        self._raw_fp = None
        self._raw_writer: Optional[csv.writer] = None
        self.raw_data = []                       # keeps every callback row
        self._raw_chunk_paths = []
        self.CHUNK_SIZE = 500_000  # flush every 500 k samples (~40 MB)
        self._chunk_idx = 0
        # rolling buffers for 1-Hz summary
        self.pupil_areas = collections.deque()
        self.eye_validity = collections.deque()
        self.gaze_points = collections.deque()
        self.fixations = collections.deque()
        self.fixation_start_time = None
        self.fixation_points = []
        # thresholds
        self.FIXATION_DURATION_THRESHOLD = 100      # ms
        self.FIXATION_DISPERSION_THRESHOLD = 0.01   # norm. display coords
        self.PUPIL_CONVERSION = 10                  # µm² scaling
        self.recording = False
        self.exp_start_ns: Optional[int] = None     # injected by main app

    # public helper -----------------------------------------------------
    def set_exp_start_ns(self, start_ns: int) -> None:
        """Receive experiment start time (perf_counter_ns) from caller."""
        self.exp_start_ns = start_ns

    # connection --------------------------------------------------------
    def connect(self) -> None:
        """Blocks until a Tobii tracker is found."""
        while True:
            print("[Log] Searching for eye tracker …")
            if tr is None:
                raise RuntimeError("tobii_research not available.")
            trackers = tr.find_all_eyetrackers()
            if trackers:
                self.et = trackers[0]
                print("[Log] Connected to", self.et.serial_number)
                return
            time.sleep(1)

    # recording ---------------------------------------------------------
    def start_recording(self) -> None:
        if not self.et:
            raise RuntimeError("Call connect() first.")
        if not self.recording:
            self._open_raw_log()
            self.et.subscribe_to(tr.EYETRACKER_GAZE_DATA,
                                 self.gaze_data_callback,
                                 as_dictionary=True)
            self.et.subscribe_to(tr.EYETRACKER_EYE_OPENNESS_DATA,
                                 self.eye_openness_callback,
                                 as_dictionary=True)
            self.recording = True
            print("[Log] Eye-tracking recording started.")

    def stop_recording(self) -> None:
        if self.recording:
            self.et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA,
                                     self.gaze_data_callback)
            self.et.unsubscribe_from(tr.EYETRACKER_EYE_OPENNESS_DATA,
                                     self.eye_openness_callback)
            self.recording = False
            self._close_raw_log()
            print("[Log] Eye-tracking recording stopped.")

    # raw CSV -----------------------------------------------------------
    def _open_raw_log(self) -> None:
        if self.save_dir is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir,
                            f"raw_gaze_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self._raw_fp = open(path, "w", newline="")
        self._raw_writer = csv.writer(self._raw_fp)
        self._raw_writer.writerow(self.RAW_HEADER)
        print("[Log] Raw gaze stream →", path)

    def _close_raw_log(self) -> None:
        if self._raw_fp:
            self._raw_fp.flush()
            self._raw_fp.close()
            self._raw_fp = None
            self._raw_writer = None

    # ------------------------------------------------------------------
    #              gaze  +  eye openness callbacks
    # ------------------------------------------------------------------
    def gaze_data_callback(self, g):
        # --- clocks ----------------------------------------------------
        rel_s = ((time.perf_counter_ns() - self.exp_start_ns) / 1e9
                 if self.exp_start_ns else 0.0)

        ts_sys   = g["system_time_stamp"]
        ts_dev   = g["device_time_stamp"]

        # unpack --------------------------------------------------------
        lpd = g.get("left_pupil_diameter")    # mm
        rpd = g.get("right_pupil_diameter")
        lgp = g.get("left_gaze_point_on_display_area", [None, None])
        rgp = g.get("right_gaze_point_on_display_area", [None, None])
        lval = g.get("left_gaze_point_validity", 0)
        rval = g.get("right_gaze_point_validity", 0)

        # write raw -----------------------------------------------------
        row = [rel_s, ts_sys, ts_dev,
               lgp[0], lgp[1], rgp[0], rgp[1],
               lpd, rpd, lval, rval]

        if self._raw_writer:
            self._raw_writer.writerow(row)
        self.raw_data.append(row)

        if len(self.raw_data) >= self.CHUNK_SIZE and self.save_dir:
            chunk_path = os.path.join(
                self.save_dir, f"_raw_chunk_{self._chunk_idx}.npy")
            np.save(chunk_path, np.array(self.raw_data, dtype=float))
            self._raw_chunk_paths.append(chunk_path)
            self.raw_data.clear()
            self._chunk_idx += 1

        # rolling metrics ----------------------------------------------
        now = time.time()
        # pupil area
        areas = []
        for d in (lpd, rpd):
            if d and not isnan(d):
                radius = d / 2.0
                area_um2 = (pi * radius ** 2) * 1_000_000 / self.PUPIL_CONVERSION
                areas.append(area_um2)
        if areas:
            self.pupil_areas.append((now, sum(areas) / len(areas)))
        # gaze point
        pts = []
        if lval == 1 and lgp and not any(isnan(c) for c in lgp):
            pts.append(lgp)
        if rval == 1 and rgp and not any(isnan(c) for c in rgp):
            pts.append(rgp)
        if pts:
            avg_x = sum(p[0] for p in pts) / len(pts)
            avg_y = sum(p[1] for p in pts) / len(pts)
            self.gaze_points.append((now, (avg_x, avg_y)))
            self.detect_fixation(avg_x, avg_y, now)

    def eye_openness_callback(self, d):
        now = time.time()
        l_ok = d.get("left_eye_validity")  == 1 and d.get("left_eye_openness_value", 0)  > 0
        r_ok = d.get("right_eye_validity") == 1 and d.get("right_eye_openness_value", 0) > 0
        self.eye_validity.append((now, l_ok or r_ok))

    # fixation detection (unchanged) -----------------------------------
    def detect_fixation(self, avg_x, avg_y, t):
        if self.fixation_start_time is None:
            self.fixation_start_time = t
            self.fixation_points = [(avg_x, avg_y)]
            return
        disp = max(sqrt((x - avg_x) ** 2 + (y - avg_y) ** 2)
                   for x, y in self.fixation_points)
        if disp <= self.FIXATION_DISPERSION_THRESHOLD:
            self.fixation_points.append((avg_x, avg_y))
            dur_ms = (t - self.fixation_start_time) * 1000
            if dur_ms >= self.FIXATION_DURATION_THRESHOLD:
                self.fixations.append((t, dur_ms))
                self.fixation_start_time = None
                self.fixation_points = []
        else:
            self.fixation_start_time = t
            self.fixation_points = [(avg_x, avg_y)]

    # ------------------------------------------------------------------
    #  getters used by the 1-Hz summary writer thread (unchanged)
    # ------------------------------------------------------------------
    def get_mean_pupil_area(self, window_s):
        now = time.time()
        vals = [a for t, a in self.pupil_areas if now - t <= window_s]
        return sum(vals) / len(vals) if vals else None

    def get_average_gaze_point(self, window_s):
        now = time.time()
        pts = [p for t, p in self.gaze_points if now - t <= window_s]
        if not pts:
            return None, None
        avg_x = sum(p[0] for p in pts) / len(pts)
        avg_y = sum(p[1] for p in pts) / len(pts)
        return avg_x, avg_y

    def is_user_looking(self, window_s):
        now = time.time()
        val = [v for t, v in self.eye_validity if now - t <= window_s]
        return any(val) if val else False

    def get_mean_fixation_duration(self, window_s):
        now = time.time()
        durs = [d for t, d in self.fixations if now - t <= window_s]
        return (sum(durs) / len(durs)) / 1000 if durs else None

    # ------------------------------------------------------------------
    #  save both RAW + processed as .npy --------------------------------
    # ------------------------------------------------------------------
    def save_data_npy(self, directory: str):
        """
        Creates two binary numpy files:

        raw_data_<timestamp>.npy
            2-D array; columns == RAW_HEADER (incl. t_rel_s)
        processed_data_<timestamp>.npy
            dict with pupil_areas, gaze_points, etc.
        """
        os.makedirs(directory, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')

        # RAW -----------------------------------------------------------
        raw_arrays = [np.load(p) for p in self._raw_chunk_paths]
        raw_arrays.append(np.array(self.raw_data, dtype=float))
        raw_all = np.concatenate(raw_arrays, axis=0) if len(raw_arrays) > 1 else raw_arrays[0]

        raw_path = os.path.join(directory, f"raw_data_{ts}.npy")
        np.save(raw_path, raw_all)

        # remove temp files
        for p in self._raw_chunk_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        # PROCESSED -----------------------------------------------------
        processed = {}
        if self.pupil_areas:
            pa = np.array(self.pupil_areas, dtype=float)
            processed["pupil_areas"] = pa
            processed["mean_pupil_area"] = float(np.mean(pa[:, 1]))
        if self.gaze_points:
            gp = np.array([(t, x, y) for t, (x, y) in self.gaze_points], dtype=float)
            processed["gaze_points"] = gp
            processed["mean_gaze"] = gp[:, 1:3].mean(axis=0)
        if self.eye_validity:
            ev = np.array(self.eye_validity, dtype=object)
            processed["eye_validity"] = ev
            processed["valid_ratio"] = float(np.mean(ev[:, 1].astype(float)))
        if self.fixations:
            fx = np.array(self.fixations, dtype=float)
            processed["fixations"] = fx
            processed["mean_fix_dur_ms"] = float(np.mean(fx[:, 1]))

        proc_path = os.path.join(directory, f"processed_data_{ts}.npy")
        np.save(proc_path, processed, allow_pickle=True)

        print(f"[Log] Saved RAW  → {raw_path}")
        print(f"[Log] Saved PROC → {proc_path}")

    # housekeeping -----------------------------------------------------
    def cleanup_old_data(self, window_s):
        now = time.time()
        for buf in (self.pupil_areas, self.eye_validity,
                    self.gaze_points, self.fixations):
            while buf and now - buf[0][0] > window_s:
                buf.popleft()


# ----------------------------------------------------------------------
# stand-alone quick test (leave unchanged if you don't need it)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    eye = TobiiEyeTracker(save_dir=".")
    eye.connect()
    EXP_START_NS = time.perf_counter_ns()
    eye.set_exp_start_ns(EXP_START_NS)
    eye.start_recording()
    try:
        time.sleep(10)
    finally:
        eye.stop_recording()
        eye.save_data_npy(".")