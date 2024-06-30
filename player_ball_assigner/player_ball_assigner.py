import sys
from utils import get_center_of_bbox
sys.path.append('../')


class PlayerBallAssigner():

    def __init__(self):
        # setting max player ball distance as 70 pixels
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
