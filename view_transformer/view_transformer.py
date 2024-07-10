import numpy as np
import cv2


class ViewTransformer():

    def __init__(self):
        court_width = 68
        court_length = 23.32

        # Defining vertices of trapezoid
        self.pixel_vertices = np.array([[110, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])
        # Defining vertices of trapezoid after we transform it into rectangle
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Converting pixel_vertices and target_vertices arrays to the float32 data type.
        # Done to ensure the data is in the correct format for processing.
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Creating a perspective transformer using OpenCV's getPerspectiveTransform function.
        # This function calculates the transformation matrix that maps the pixel_vertices (trapezoid in pixel space) to the target_vertices (rectangle in target space).
        self.persepctive_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices)

    def add_transformed_position_to_tracks(self, tracks):
        # Iterating over each object and its tracks in the dictionary
        for object, object_tracks in tracks.items():
            # Iteraring over each frame's track in the list of object tracks
            for frame_num, track in enumerate(object_tracks):
                # Iterating over each track ID and its info in the dictionary
                for track_id, track_info in track.items():
                    # Extracting ethe 'position_adjusted' value from track_info
                    position = track_info['position_adjusted']
                    # Converting the position to a NumPy array
                    position = np.array(position)
                    # Applying a transformation to the position using a method called transform_point.
                    position_transformed = self.transform_point(position)
                    # If the transformed position is not None, squeeze and convert to a list
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    # Updating the tracks dictionary with the transformed position
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
