from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2


def main():

    # Reading video
    video_frames = read_video(
        '/Users/adityamishra/Documents/Football-Analysis/input-videos/08fd33_4.mp4')

    # Initialising Tracker
    tracker = Tracker('models/best.pt')

    # Getting object
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Interpolating/inserting ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    # Assigning player teams
    team_assigner = TeamAssigner()  # initialising team assigner
    # assigning team their colours and by giving them first frame and tracks of player in first frame
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    # Looping over each player in each frame and assigning them to colour team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Calling draw output function for object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Saving video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()
