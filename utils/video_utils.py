"""
Here we will have utilites to read from a video and save the video
"""
import cv2

# function to return the list of frames of video


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []  # initalising empty frames
    while True:
        ret, frame = cap.read()
        if not ret:  # if falls video will end
            break
        frames.append(frame)

    return frames

# function to save sequence of image frames as a video file


"""
ouput_video_frames is expected to be a list or array of numpy arrays, where each numpy array represents an image frame.
output_video_path is a string representing the file path where the video will be saved, including the filename and extension 
"""


def save_video(output_video_frames, output_video_path):
    """
    cv2.VideoWriter_fourcc() is a function from OpenCV that creates a 4-byte code used to specify the video codec.
    XVID is an open-source video codec that provides good compression while maintaining quality
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # assinging video formam XVID

    """
    This creates a cv2.VideoWriter object which will be used to write the video file.

    output_video_path is the path where the video will be saved.
    
    24 is the frame rate of the output video
    after that we define frame and height of the video frame

    """
    out = cv2.VideoWriter(output_video_path, fourcc, 24,
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    # This loop iterates through each frame in the ouput_video_frames list and out.write(frame) writes each frame to the video file.
    for frame in output_video_frames:
        out.write(frame)

    # After all frames have been written, out.release() is called to release the VideoWriter object.
    out.release()
