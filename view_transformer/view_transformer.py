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

    def transform_point(self, point):
        # Converting point to the integer co-ordinates
        """
        point[0]: The x-coordinate of the point.
        point[1]: The y-coordinate of the point.
        int(): Converts the coordinates to integer values.
        """
        p = (int(point[0]), int(point[1]))
        """
        cv2.pointPolygonTest(): A function from the OpenCV library that checks whether a point is inside a polygon.
        self.pixel_vertices: The vertices of the polygon.
        p: The point to be tested.
        False: A flag indicating that the function should only test if the point is inside the polygon and not calculate the distance to the edge.
        """
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        # if the point is outside the polygon then execution ends
        if not is_inside:
            return None

        """
        point.reshape(-1, 1, 2): Reshapes the point to a specific format required by the cv2.perspectiveTransform() function.
        -1: Infers the dimension size from the other given dimensions.
        1: The number of rows.
        2: The number of columns (for the x and y coordinates).
        AND
        astype(np.float32) Converts the point data type to float32.
        """
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.persepctive_transformer)  # the transformed point
        return transform_point.reshape(-1, 2)

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
