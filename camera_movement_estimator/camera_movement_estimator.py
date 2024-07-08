from utils import measure_distance, measure_xy_distance
import cv2
import pickle
import numpy as np
import os
import sys
sys.path.append('../')


class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5
        """
        Here we define a dictionary self.lk_params that contains parameters for the Lucas-Kanade optical flow method.
        The Lucas-Kanade method is used to estimate the motion of objects between two consecutive frames.

        `winSize` specifies the size of the search window at each pyramid level. In this case, a window size of 15x15 pixels is used. This window is used to match features between frames.
        `maxLevel` specifies the maximum number of pyramid levels, including the initial image. A value of 2 means the algorithm will use three levels in total: the original image (level 0), and two reduced resolution images (levels 1 and 2).

        criteria specifies the termination criteria of the iterative search algorithm of the optical flow. It is a tuple where:
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT is a combination of two criteria types:
        cv2.TERM_CRITERIA_EPS: The algorithm stops if the specified accuracy (epsilon) is reached.
        cv2.TERM_CRITERIA_COUNT: The algorithm stops after a specified number of iterations.
        10 is the maximum number of iterations the algorithm will run.
        0.03 is the desired accuracy (epsilon). If the change in the search window position is less than this value, the algorithm will stop.
        """
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Here we converting the input frame from RGB color space to grayscale using OpenCV's cvtColor function. cv2.COLOR_BGR2GRAY specifies the conversion type.
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Creating an array of zeros with the same shape as first_frame_grayscale. This will serve as a mask to specify regions of interest for feature detection.
        mask_features = np.zeros_like(first_frame_grayscale)
        # Setting the first 20 columns and the columns from 900 to 1050 in the mask_features array to 1.
        # This indicates that only these regions will be used for feature detection.
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        """
        Here we are defining a dictionary self.features with parameters for the cv2.goodFeaturesToTrack function:

        `maxCorners: is used for maximum number of corners to return. If there are more corners than that, the strongest ones are returned.
        qualityLevel: Parameter characterizing the minimal accepted quality of image corners. A value of 0.3 means the algorithm will keep corners with a quality score of at least 30% of the best corner's score.
        minDistance: Minimum possible Euclidean distance between the returned corners.
        blockSize: Size of the averaging block used for computing a derivative covariation matrix over each pixel neighborhood.
        mask: The mask image specifying the regions where features will be detected.
        """
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    # This method adjusts the positions of tracked objects based on the movement of the camera.
    # We take three parameters
    # self: Refers to the instance of the class.
    # tracks: A dictionary containing tracking information for multiple objects. Each object has its own set of tracks across frames.
    # camera_movement_per_frame: A list where each element represents the camera movement(x and y coordinates) for each frame.
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        # Iterating over the tracks dictionary. object is the key representing a specific object, and object_tracks is the value, which is a list of tracks for that object.
        for object, object_tracks in tracks.items():
            # Iterating  over the list of object_tracks. frame_num is the index of the current frame, and track is the tracking information for that frame.
            for frame_num, track in enumerate(object_tracks):
                # Iterating over the track dictionary. track_id is the key representing a specific track ID, and track_info is the value, which contains information about that track
                for track_id, track_info in track.items():
                    # RetrievING the original position of the tracked object from track_info.
                    position = track_info['position']
                    # Retreiving the camera movement for the current frame from camera_movement_per_frame.
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Adjusting  the position of the tracked object by subtracting the camera movement from the original position. This compensates for the movement of the camera, resulting in the adjusted position.
                    position_adjusted = (
                        position[0]-camera_movement[0], position[1]-camera_movement[1])
                    # Updating the tracks dictionary with the adjusted position for the specific track_id within the current frame of the object.
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):

        # Reading the stub from the file
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # movement for x and movement for y coordinates and multiplying by frames
        camera_movement = [[0, 0]]*len(frames)

        # extrcting old frames and features
        # Converting the first frame in frames to grayscale.
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # Detecting  good features to track in old_gray using the parameters defined in self.features.
        #  The **self.features syntax unpacks the dictionary into keyword arguments.
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params)

            # Measuring distance between new and old features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            """
            Here we are iterating over the pairs of new and old features.

            new and old are points representing feature coordinates.
            .ravel() is a method in NumPy that flattens an array. It returns a contiguous flattened array, meaning it converts a multi-dimensional array into a one-dimensional array.

            new and old is likely a 2D array with the shape (1, 2) (or similar), containing the x and y coordinates of a feature point.
            new.ravel() flattens this 2D array into a 1D array with the shape (2,), which means new_features_point will be a 1D array containing the x and y coordinates
            """
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # calculating the distance between the new and old features using a custom function measure_distance.
                distance = measure_distance(
                    new_features_point, old_features_point)

                # If the current distance is greater than max_distance, update max_distance and calculate the x and y movement using a custom function measure_xy_distance.
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point)

            """
            Here we check the max_distance (the maximum distance between new and old features calculated in the loop) is greater than a threshold value, self.minimum_distance.
            If the condition is true we update the camera_movement list at the index frame_num (the current frame number).
            camera_movement_x and camera_movement_y are the x and y components of the movement vector, calculated as the largest movement of features between the two frames.

            After recording the camera movement, this line detects new features in the current frame (now stored in frame_gray).
            cv2.goodFeaturesToTrack is an OpenCV function used to detect good features (corners) to track in an image.
            The **self.features syntax unpacks the dictionary self.features into keyword arguments for the cv2.goodFeaturesToTrack function. This dictionary contains parameters like maxCorners, qualityLevel, minDistance, blockSize, and mask.
            """
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [
                    camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features)

            # Updating old_gray to be the current frame's grayscale image.
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6  # for ftransparency
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (
                10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames
