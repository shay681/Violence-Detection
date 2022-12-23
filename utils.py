from os import listdir
from os.path import isfile, join
import cv2 as cv


def get_file_names_from_dir(dir: str):
    return [dir + f for f in listdir(dir) if isfile(join(dir, f))]


def read_video_from_file(file: str):
    return cv.VideoCapture(file)


def play_video(video):
    video_window = 'video_window'
    cv.namedWindow(video_window)
    while True:
        ret, frame = video.read()  # read a single frame
        if not ret:
            print("EOV or Could not read the frame")
            cv.destroyWindow(video_window)
            break
        reescaled_frame = frame
        #     for i in range(scaleLevel-1):
        #         reescaled_frame = cv.pyrDown(reescaled_frame)
        cv.imshow(video_window, reescaled_frame)
        waitKey = (cv.waitKey(1) & 0xFF)
        if waitKey == ord('q'):  # if Q pressed you could do something else with other keypress
            cv.destroyWindow(video_window)
            video.release()
            break


def break_video_into_frames(video, p=False):
    frames = []
    success = True
    while success:
        success, frame = video.read()
        frames.append(frame)
    if (p):
        print(f'Readed {len(frames)} frames.')
    return frames


def display_frame(frame):
    cv.imshow('Frame:', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


def play_frames(frames):
    video_window = 'video_window'
    cv.namedWindow(video_window)
    for f in frames:
        cv.imshow(video_window, f)
        waitKey = (cv.waitKey(1) & 0xFF)
        if waitKey == ord('q'):  # if Q pressed you could do something else with other keypress
            cv.destroyWindow(video_window)
            break


def loop_frames(frames):
    video_window = 'video_window'
    cv.namedWindow(video_window)
    last_frame_index = len(frames) - 1
    i = 0
    while i != last_frame_index:
        if (i + 1) == last_frame_index:
            i = 0
        cv.imshow(video_window, frames[i])
        i = i + 1
        waitKey = (cv.waitKey(1) & 0xFF)
        if waitKey == ord('q'):  # if Q pressed you could do something else with other keypress
            cv.destroyWindow(video_window)
            break
