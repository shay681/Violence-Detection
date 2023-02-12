import os
import h5py
import numpy as np
from utils import list_h5_files, count_videos
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def create_training_data(data_path, max_sentences, max_words, num_keypoints, val_split = 0):
    list_files = list_h5_files(data_path)
    n_videos = count_videos(data_path)
    print(f"There are {n_videos} videos.")
    data = np.zeros((n_videos, max_sentences, max_words, num_keypoints, 3))
    labels = []

    for file_name in list_files:
        video_ind = 0
        full_path = os.path.join(data_path, file_name)
        try:
            with h5py.File(full_path, "r") as f:   
                for class_label in f.keys():
                    label = 1 if class_label == 'violent' else 0
                    for video_name in f[class_label].keys(): # video is text
                        for frame_num in range(len(f[class_label + '/' + video_name].keys())): # frame is sentence
                            poses = np.array(f[class_label + '/' + video_name + f'/{frame_num}']) # pose is word
                            if poses.size !=0:
                                pose_lim = min(poses.shape[0],max_words)
                                data[video_ind, frame_num, :pose_lim , :, :] = poses[:pose_lim , :, :]
                        video_ind += 1
                        labels.append(label)
        except:
            pass

    data = data.reshape((n_videos, max_sentences, max_words,  num_keypoints * 3))
    labels = to_categorical(np.asarray(labels))
    print('Shape of reviews (data) tensor:', data.shape)
    print('Shape of sentiment (label) tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(val_split * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]

    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]
    x_testing = data[-nb_validation_samples:]
    y_testing = labels[-nb_validation_samples:]
    x_test, x_val, y_test, y_val = train_test_split(x_testing, y_testing, test_size=0.3, random_state=42)


    print('Number of positive and negative reviews in training and validation set')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    return x_train, y_train, x_test, y_test, x_val, y_val

def create_test_data(data_path, max_sentences, max_words, num_keypoints):
    list_files = list_h5_files(data_path)
    n_videos = count_videos(data_path)
    print(f"There are {n_videos} videos.")
    data = np.zeros((n_videos, max_sentences, max_words, num_keypoints, 3))
    labels = []

    for file_name in list_files:
        video_ind = 0
        full_path = os.path.join(data_path, file_name)
        try:
            with h5py.File(full_path, "r") as f:   
                for class_label in f.keys():
                    label = 1 if class_label == 'violent' else 0
                    for video_name in f[class_label].keys(): # video is text
                        for frame_num in range(len(f[class_label + '/' + video_name].keys())): # frame is sentence
                            poses = np.array(f[class_label + '/' + video_name + f'/{frame_num}']) # pose is word
                            if poses.size !=0:
                                pose_lim = min(poses.shape[0],max_words)
                                data[video_ind, frame_num, :pose_lim , :, :] = poses[:pose_lim , :, :]
                        video_ind += 1
                        labels.append(label)
        except:
            pass

    data = data.reshape((n_videos, max_sentences, max_words,  num_keypoints * 3))
    print("Number of videos:", len(labels))
    print("Violent videos:", sum(labels))
    print("Non-Violent videos:", len(labels) - sum(labels))
    labels = to_categorical(np.asarray(labels))
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data, labels