

class SpeedAndDistance_Estimator():
    def __init__(self):
        # Calculating speed of user every 5 frame
        self.frame_window = 5
        self.frame_rate = 24  # frame rate 24 frame per second
