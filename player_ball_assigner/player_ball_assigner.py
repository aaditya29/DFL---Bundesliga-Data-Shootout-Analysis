import sys
from utils import get_center_of_bbox, measure_distance
sys.path.append('../')


class PlayerBallAssigner():

    def __init__(self):
        # setting max player ball distance as 70 pixels
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        # getting center of the ball box as ball position
        ball_position = get_center_of_bbox(ball_bbox)

        # getting closest player
        miniumum_distance = 99999
        assigned_player = -1

        # looping over each player
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position)
            # real distance is going to be minimum dist between the two
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player
