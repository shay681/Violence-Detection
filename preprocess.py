import cv2
import numpy as np
import os
import sys
import tqdm
import h5py
from utils_preprocess import *
from utils_preprocess import load_model
import gc

VIOLENT_PATH = 'data/train/Fight/'
NON_VIOLENT_PATH = 'data/train/NonFight'
IMAGES_OUT_FOLDER = 'poses_visualsition/'
H5_OUT_PATH = 'data/poses_pickle.h5'
POSE_ESTIMATION_MODEL_PATH = 'yolov7/yolov7-w6-pose.pt'

if __name__ == "__main__":
    with h5py.File(H5_OUT_PATH, 'w') as hf:
        violent_videos_files = get_file_names_from_dir(VIOLENT_PATH)
        non_violent_videos_files = get_file_names_from_dir(NON_VIOLENT_PATH)
        classes_paths = {'violent': VIOLENT_PATH, 'non_viloent': NON_VIOLENT_PATH}

        video = read_video_from_file(violent_videos_files[-1])
        video_frames = break_video_into_frames(video, True)
        # loop_frames(video_frames)

        # create dictionary with the output pose embeddings
        
        pose_estimation_model = load_model(POSE_ESTIMATION_MODEL_PATH)

        # Folder names are used as pose class names.
        for class_name, class_path in classes_paths.items():
            if not os.path.exists(os.path.join(IMAGES_OUT_FOLDER, class_name)):
                os.makedirs(os.path.join(IMAGES_OUT_FOLDER, class_name))
            
            class_videos_files = get_file_names_from_dir(class_path)
            images_out_folder_class = os.path.join(IMAGES_OUT_FOLDER, class_name)
            for video_file in tqdm.tqdm(class_videos_files, position=0):
                video = read_video_from_file(video_file)
                video_frames = break_video_into_frames(video, True)
                
                video_name = os.path.basename(video_file)
                video_name = os.path.splitext(video_name)[0]
                frame_num = 0
                for frame in video_frames:
                    if frame is not None:
                        # create a unique name for current frame
                        image_name = f'{video_name}_{frame_num}.png'
                        # print(image_name)
                        visualize_plot_path = os.path.join(images_out_folder_class, image_name)

                        # run pose estimation on frame
                        output, image =  run_inference(frame,pose_estimation_model) 
                        # output =  run_inference(frame,pose_estimation_model) 
                        # filter keypoints with low score
                        supressed_output = supress_kpt(output, pose_estimation_model)
                        with torch.no_grad():
                            keypoints = output_to_keypoint(supressed_output)
                        # print(type(keypoints))
                        
                        # plot the pose estimation on top of the frame
                        visualize_output(keypoints, image, visualize_plot_path, pose_estimation_model, show = False)
                        # del output, image
                        torch.cuda.empty_cache()
                        gc.collect()


                        # add output to dataframe
                        hf.create_dataset(f"{class_name}/{video_name}/{frame_num}",  data= keypoints)
                        
                        frame_num = frame_num + 1