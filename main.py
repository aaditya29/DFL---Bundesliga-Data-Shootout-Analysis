from utils import read_video, save_video
from trackers import Tracker
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

    # Saving cropped image of the player for analysis
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # cropping bbox from the frame
        cropped_image = frame[int(bbox[1]):int(
            bbox[3]), int(bbox[0]):int(bbox[2])]

        # saving the image
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        break

    # Calling draw output function for object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Saving video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()
