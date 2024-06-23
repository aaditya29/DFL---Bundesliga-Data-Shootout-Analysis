from ultralytics import YOLO


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # method to detect the frames from the videos with self as reference and frames as list or array of image frames
    def detect_frames(self, frames):
        """

        """
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
