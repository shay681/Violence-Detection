from utils import *

VIOLENT_PATH = 'data/fights/'
NON_VIOLENT_PATH = 'data/noFights/'

if __name__ == "__main__":
    violent_videos_files = get_file_names_from_dir(VIOLENT_PATH)
    non_violent_videos_files = get_file_names_from_dir(NON_VIOLENT_PATH)

    video = read_video_from_file(violent_videos_files[-1])
    video_frames = break_video_into_frames(video, True)
    loop_frames(video_frames)