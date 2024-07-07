import cv2
import pickle
import numpy as np
import matplotlib as plt


class CameraMovementEstimator:
    def __init__(self, frame):

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

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):

        # Reading the stub from the file
        # movement for x and movement for y coordinates and multiplying by frames
        camera_movement = [[0, 0]]*len(frames)

        # extrcting old frames and features
        # Converting the first frame in frames to grayscale.
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # Detecting  good features to track in old_gray using the parameters defined in self.features.
        #  The **self.features syntax unpacks the dictionary into keyword arguments.
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
