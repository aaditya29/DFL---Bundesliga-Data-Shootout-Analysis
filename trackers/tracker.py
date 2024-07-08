from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import sys
import cv2
sys.path.append('../')


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # each detected object is assigned a unique tracker ID, enabling the continuous following of the object's motion path across different frames
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    # method to detect the frames from the videos with self as reference and frames as list or array of image frames
    def interpolate_ball_positions(self, ball_positions):
        # Converting ball position format to pandas dataframe format
        # We are getting track id 1 and if no track id then
        # it's going to be empty dictonary
        # and then we get bbox and if not bbox then empty list and then
        # the empty list will be interpolated by pandas dataframe
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        # converting to pandas dataframes with columns 'x1', 'y1', 'x2', and 'y2'.
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate/inserting Missing Values
        df_ball_positions = df_ball_positions.interpolate()
        # Edge case if the missing detection is first one we not going to interpolate it
        # bfill()performs a backward fill on the DataFrame, filling any remaining NaN values with the next valid value.
        df_ball_positions = df_ball_positions.bfill()

        # Converting the processed DataFrame back into the original nested dictionary structure.
        ball_positions = [{1: {"bbox": x}}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

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

    """
    Following method defines a method called draw_ellipse that takes several parameters:

    self: Indicates method in a class
    frame: video frame
    bbox: bounding box, ontaining coordinates
    color: color to draw the ellipse
    track_id: An optional parameter, defaulting to None
    """

    def draw_ellipse(self, frame, bbox, color, track_id=None):

        # y2 is set to the integer value of the fourth element in bbox
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        """
        Here we calling OpenCV's ellipse function to draw an ellipse on the frame.
    
        The center is at (x_center, y2)
        The axes are (width, 0.35*width), making it wider than it is tall
        The ellipse is not rotated (angle=0.0)
        It's drawn from -45 degrees to 235 degrees, creating a partial ellipse
        It uses the provided color
        The line thickness is 2 pixels
        The line type is cv2.LINE_4 i.e. (4-connected line)
        """
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Drawing rectangle
        # Defining the dimensions of rectangle to be drawn
        rectangle_width = 40
        rectangle_height = 20
        """
        x1_rect = x_center - rectangle_width // 2 calculates the x-coordinate of the left edge of the rectangle. It starts from the center (x_center) and subtracts half the width of the rectangle. Here // operator performs integer division, ensuring the result is a whole number.
        
        
        x2_rect = x_center + rectangle_width // 2 calculates the x-coordinate of the right edge of the rectangle. It starts from the center and adds half the width of the rectangle.
        
        y1_rect = (y2 - rectangle_height // 2) + 15 calculates the y-coordinate of the top edge of the rectangle. It starts from y2 (which is the bottom of the bounding box),subtracts half the height of the rectangle to center it vertically, then adds 15 pixels to move it slightly downward.
        
        y2_rect = (y2 + rectangle_height // 2) + 15 calculates the y-coordinate of the bottom edge of the rectangle.It starts from y2, adds half the height of the rectangle, and also adds 15 pixels to match the downward shift of the top edge.
        """
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:  # checking if track_id was provided then

            """
            Here we draw a filled rectangle on the frame.

            Co-ordinates are top-left and bottom-right corners of the rectangle and cv2.FILLED fills the rectangle with the specified color.
            """
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            """
            This calculates the x-coordinate for text placement. It is adjusted slightly if track_id is greater than 99 to ensure proper alignment.
            """
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            """
            Here we add the track ID as text on the frame:
            Text is the string representation of track_id
            Positioned based on the calculated coordinates
            Uses specified font, scale 0.6
            Black color (0,0,0)
            Thickness of 2
            """
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame

    """
    Here we define a method named draw_triangle that takes parameters:
    self: Indicates this is a method in a class
    frame: The image to draw on
    bbox: Bounding box coordinates
    color: Color for filling the triangle
    """

    def draw_triangle(self, frame, bbox, color):
        # y is set to the integer value of the second element of bbox
        y = int(bbox[1])
        # x is obtained by calling get_center_of_bbox(bbox), which returns the center x-coordinate of the bounding box
        x, _ = get_center_of_bbox(bbox)

        """
        Here we are defining the trangle points.
        This creates a NumPy array defining three points of a triangle:
        The bottom point at (x, y)
        The top-left point at (x-10, y-20)
        The top-right point at (x+10, y-20) whichforms an upward-pointing triangle.
        """
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        """
        Here we use  OpenCV's drawContours function to draw and fill the triangle:
        frame: The image to draw on
        [triangle_points]: A list containing the triangle points
        0: Index of the contour to draw (0 since there's only one)
        color: The color to fill the triangle with
        cv2.FILLED: Indicates to fill the triangle
        """
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)

        # Drawing triangle outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        In this method we are taking four parameters:
        self: The instance of the class this method belongs to
        frame: The current video frame to draw on
        frame_num: The current frame number
        team_ball_control: An array containing ball control data


        """
        # Drawing a semi-transparent rectangle
        # Creating a copy of the current frame, which will be used to create a semi-transparent overlay
        overlay = frame.copy()
        """
        Here we draw a white rectangle on the overlay image. The rectangle's top-left corner is at (1350, 850) and bottom-right corner is at (1900, 970). The -1 thickness means the rectangle is filled.
        
        And then we blend the overlay with the original frame to create a semi-transparent effect.
        cv2.addWeighted() function is used to blend two images.
        overlay:is the first input array (image). In this case, it's the copy of the frame with the white rectangle drawn on it.
        alpha: is the weight of the first array elements. It's set to 0.4 in our code, which means the overlay will be 40% opaque.
        frame: is the second input array (image). Here, it's the original frame.
        1 - alpha: is the weight of the second array elements. Since alpha is 0.4, this will be 0.6, meaning the original frame will be 60% visible.
        0: is a scalar added to each sum. In this case, it's 0, so nothing extra is added.
        frame: is the output array. The result is stored back in the original frame.
        """
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4  # for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # slicing the team_ball_control array from the beginning up to and including the current frame.
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        """
        Here we count, how many frames each team had control of the ball. It does this by counting how many times 1 (for team 1) and 2 (for team 2) appear in the sliced array.
        
        Then we calculate the percentage of ball control for each team by dividing each team's frame count by the total number of frames. These lines add text to the frame showing the ball control percentages for each team. The text is black, uses the HERSHEY_SIMPLEX font, has a scale of 1, and a thickness of 3.
        """
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",
                    (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",
                    (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    # Drawing Near Bounding Boxes
    """
    self: refers to the instance of the class this method belongs to.

    video_frames: A list of video frames to be annotated.

    tracks: A dictionary containing tracking information for players, the ball, and referees. This dictionary is structured with keys "players", "ball", and "referees", each containing lists of dictionaries for each frame.

    team_ball_control: Data that indicates which team has control of the ball for each frame.
    """

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # initialising an empty list to store the annotated video frames
        output_video_frames = []  # video frames after drawing the output on

        # Here we starting a loop that iterates through each frame in video_frames. enumerate() is used to get both the index (frame_num) and the frame itself.
        for frame_num, frame in enumerate(video_frames):
            # creating a copy of the current frame to avoid modifying the original frame.
            frame = frame.copy()

            # Retrieving dictionaries containing tracking information for players, the ball, and referees for the current frame.
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Drawing Players
            """
            Here we looping through each player in the dict.
        
            First we retrieve the player's team color, defaulting to red.
        
            Calling 'self.draw_ellipse' to draw an ellipse around the player's bounding box (player["bbox"]) in the specified color. 
        
            If the player has the ball (player.get('has_ball', False)), calls self.draw_traingle to draw a triangle on the player's bounding box.
            """
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["bbox"], color, track_id)

                # if player doesn't have ball then we can draw just a triangle with colour red
                if player.get('has_ball', False):
                    frame = self.draw_triangle(
                        frame, player["bbox"], (0, 0, 255))

            # Drawing Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Drawing team ball control box
            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
