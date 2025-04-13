import tobii_research as tr
import threading

class EyeTrackerHelper:
    def __init__(self):
        self.eyetracker = None
        self.gaze_data = None
        self.lock = threading.Lock()
        self.is_tracking = False
        self.initialize_eye_tracker()

    def initialize_eye_tracker(self):
        eyetrackers = tr.find_all_eyetrackers()
        if eyetrackers:
            self.eyetracker = eyetrackers[0]
            print(f"Connected to eye tracker: {self.eyetracker.serial_number}")
            self.is_tracking = True
            self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)
        else:
            print("No eye tracker found.")
            self.is_tracking = False

    def gaze_data_callback(self, gaze_data):
        with self.lock:
            self.gaze_data = gaze_data

    def get_latest_gaze_data(self):
        with self.lock:
            if self.gaze_data is not None:
                # Extract gaze data
                left_gaze_point = self.gaze_data['left_gaze_point_on_display_area']
                right_gaze_point = self.gaze_data['right_gaze_point_on_display_area']
                left_pupil_diameter = self.gaze_data['left_pupil_diameter']
                right_pupil_diameter = self.gaze_data['right_pupil_diameter']
                timestamp = self.gaze_data['system_time_stamp']

                # Calculate average gaze point
                avg_gaze_point = self.calculate_average_point(left_gaze_point, right_gaze_point)

                # Calculate average pupil diameter
                avg_pupil_diameter = self.calculate_average_value(left_pupil_diameter, right_pupil_diameter)

                return {
                    'timestamp': timestamp,
                    'left_gaze_point': left_gaze_point,
                    'right_gaze_point': right_gaze_point,
                    'avg_gaze_point': avg_gaze_point,
                    'left_pupil_diameter': left_pupil_diameter,
                    'right_pupil_diameter': right_pupil_diameter,
                    'avg_pupil_diameter': avg_pupil_diameter
                }
            else:
                return None

    @staticmethod
    def calculate_average_point(point1, point2):
        if None not in point1 and None not in point2:
            return (
                (point1[0] + point2[0]) / 2,
                (point1[1] + point2[1]) / 2
            )
        elif None not in point1:
            return point1
        elif None not in point2:
            return point2
        else:
            return (None, None)

    @staticmethod
    def calculate_average_value(value1, value2):
        if value1 is not None and value2 is not None:
            return (value1 + value2) / 2
        elif value1 is not None:
            return value1
        elif value2 is not None:
            return value2
        else:
            return None

    def stop_tracking(self):
        if self.is_tracking and self.eyetracker:
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
            self.is_tracking = False
            print("Unsubscribed from gaze data.")

    def __del__(self):
        self.stop_tracking()
