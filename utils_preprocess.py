from os import listdir
from os.path import isfile, join
import cv2 as cv
import sys 

# sys.path.append("yolov7/")
sys.path.insert(1, 'yolov7/')
    
import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import numpy as np


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


def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        print('using gpu')
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


def run_inference(frame, model):
    # Resize and pad image
    frame_letter = letterbox(frame, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_letter = transforms.ToTensor()(frame_letter).half().to(device) # torch.Size([3, 768, 960])
    # Turn image into batch
    frame_unsqueeze = frame_letter.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    with torch.no_grad():
        output, _ = model(frame_unsqueeze) # torch.Size([1, 45900, 57])
    del frame, frame_letter
    del model
    return output, frame_unsqueeze


def visualize_output(output, image, save_path, model, show = False):
    plt.clf()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv.cvtColor(nimg, cv.COLOR_RGB2BGR)
    print(f'the output shape is: {output.shape}')
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.savefig(save_path)
    if show:
        plt.show()

    

#supress keypoints below the confidence threshold 
def supress_kpt(unfiltered_output, model):
    output = non_max_suppression_kpt(unfiltered_output, 
                0.25, # Confidence Threshold
                0.65, # IoU Threshold
                nc=model.yaml['nc'], # Number of Classes
                nkpt=model.yaml['nkpt'], # Number of Keypoints
                kpt_label=True)
    return output
