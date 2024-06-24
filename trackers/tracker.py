from ultralytics import YOLO
import supervision as sv
import pickle
import os


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # each detected object is assigned a unique tracker ID, enabling the continuous following of the object's motion path across different frames
        self.tracker = sv.ByteTrack()

    # method to detect the frames from the videos with self as reference and frames as list or array of image frames
    def detect_frames(self, frames):

        batch_size = 20  # stating batch_size as 20. In image processing, working with batches of data can be more efficient than processing each item individually

        detections = []  # initialising an empty list to store all the detection results

        """
        Following loop iterates over the frames in batches. It uses range() with a step size of batch_size, so `i` will take on values 0, 20, 40, etc., until it reaches or exceeds the length of frames
        
        self.model.predict() calls the prediction method of YOLO
        frames[i:i+batch_size] selects a batch of frames from the input.
        conf=0.1 sets a confidence threshold of 0.1 for the detections.
        """
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size], conf=0.1)
            # adding the etections from the current batch to the overall detections list.
            detections += detections_batch

        return detections  # returning the detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Here we are checking if:
        read_from_stub is True AND
        stub_path is not None AND
        The file at stub_path exist
                THEN
        we will open the stub file in binary read mode
        Loads the tracking data from the file using pickle
        And returns the loaded tracking data and exits the function
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        # putting the tracked object in a format so we can utilise easily
        # here we are using dictionary with the lists
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # overwriting goalkeeper with the player due to error in detection
        for frame_num, detection in enumerate(detections):
            # mapping class and names{0:person, 1: goal,.. etc}
            cls_names = detection.names

            # k is key and v is value
            cls_names_inv = {v: k for k, v in cls_names.items()}
            print(cls_names)

            # Converting to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Converting goalkeeper to player object
            """
            This loop iterates over the class IDs in the detection. If a class is identified as "goalkeeper", it changes its class ID to that of "player".
            """
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Tracking Objects

            """
            Following code is crucial in maintaining continuity of object identities across video frames, which is essential for tasks like counting unique objects or analyzing their trajectories over time.
            
            detection_with_tracks variable contains the updated tracking information, including both the current detections and their associated track IDs.
            
            self.tracker is an object responsible for tracking detected objects across multiple frames of video.
            
            detection_supervision is the input to the method.
            
            update_with_detections() is a method of the tracker object. Its purpose is to update the current tracks (ongoing object trajectories) with new detection information.
            """
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision)

            # for each frame we are going to have tracks of players, referees and balls
            # then we append a dict and and this is going to have key with track id and value is going to be bounding box
            # Each list item represents a frame, and we're adding an empty dictionary for the current frame.
            # This structure allows for frame-by-frame tracking of multiple objects.
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # going to loop over each detection with tracks
            """
            frame_detection[0] is the bounding box, converted to a list (it was likely a numpy array).
            frame_detection[3] is the class ID (e.g., player, referee).
            frame_detection[4] is the track ID, which is unique for each tracked object.
            """
            for frame_detection in detection_with_tracks:
                # 0 is the key of bounding box
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]  # 3 is the key of class id
                track_id = frame_detection[4]  # 4 is the key of track id

                """
                This part organizes the tracked objects by type and track ID.
                
                cls_names_inv is a dictionary mapping class names to IDs.
                For each player and referee, we're storing their bounding box in the current frame (frame_num), indexed by their track_id.
                """
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Processing ball detection

            """
            Here we use the original detection_supervision because there is only one ball and we don't need complex tracking like players or referees.
            
            The ball's bounding box is always stored with key 1, suggesting we're not tracking multiple balls or maintaining a consistent track ID for the ball across frames.
            """
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # If a stub_path was provided, save the tracking data to this file using pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
