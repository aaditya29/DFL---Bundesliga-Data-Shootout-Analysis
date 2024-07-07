import cv2
import pickle
import numpy as np
import matplotlib as plt


class CameraMovementEstimator:
    def __init__(self):
        pass

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):

        # Reading the stub from the file
        # movement for x and movement for y coordinates and multiplying by frames
        camera_movement = [[0, 0]]*len(frames)
