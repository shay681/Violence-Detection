import pandas as pd
import h5py
import numpy as np
import os
from model_compilators import get_model_parts
from keras.models import Model
from sklearn.model_selection import GridSearchCV
from keras.utils.np_utils import to_categorical
from utils import list_h5_files, count_videos
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

# Set Params
K_FOLDS = 5
NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 64 
DATA_PATH = './data/train/'
OUTPUT_PATH = './output/'
CSV_OUTPUT_FILE_NAME = f'Grid_HOKEY__BATCH_64_128_kfolds_{K_FOLDS}.csv'
MAX_SENTENCES = 151
MAX_WORDS = 20
NUM_OF_KEYPOINTS = 17

def load_data_and_labels(DATA_PATH):
    list_files = list_h5_files(DATA_PATH)
    n_videos = count_videos(DATA_PATH)
    print(f"\nThere are {n_videos} videos in data folder.\n")
    data = np.zeros((n_videos, MAX_SENTENCES, MAX_WORDS, NUM_OF_KEYPOINTS, 3))
    labels = np.zeros(n_videos)
    video_ind = 0
    for file_name in list_files:
        full_path = os.path.join(DATA_PATH, file_name)
        try:
            with h5py.File(full_path, "r") as f:   
                for class_label in f.keys():
                    label = 1 if class_label == 'violent' else 0
                    for video_name in f[class_label].keys(): # video is text
                        for frame_num in range(len(f[class_label + '/' + video_name].keys())): # frame is sentence
                            poses = np.array(f[class_label + '/' + video_name + f'/{frame_num}']) # pose is word
                            if poses.size !=0:
                                pose_lim = min(poses.shape[0],MAX_WORDS)
                                data[video_ind, frame_num, :pose_lim , :, :] = poses[:pose_lim , :, :]
                        labels[video_ind] = label
                        video_ind = video_ind + 1

        except:
            pass

    data = data.reshape((n_videos, MAX_SENTENCES, MAX_WORDS, NUM_OF_KEYPOINTS * 3))
    labels = to_categorical(np.asarray(labels))
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    return data, labels

def compile_model(optimizer='Adam'):
    review_input, preds = get_model_parts()
    model = Model(review_input, preds)
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)
    print('Model optimizer:', optimizer)
    return model


if __name__ == '__main__':
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)
    # Set Params
    optimizer = ['Adam']
    learn_rate = [0.0005]
    epochs = [15]
    batch_sizes = [64, 128]
    batch_size = BATCH_SIZE

    model = KerasClassifier(model=compile_model, loss="binary_crossentropy", epochs=epochs, batch_size=batch_size)

    # Load Data
    X, Y = load_data_and_labels(DATA_PATH)
    X = np.array(X)
    Y = np.array(Y)

    param_grid = dict(epochs=epochs, batch_size=batch_sizes, model__optimizer=optimizer, optimizer__learning_rate=learn_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=K_FOLDS)
    grid_result = grid.fit(X, Y)
    
    result_df = pd.DataFrame(grid_result.cv_results_)
    result_df.to_csv(OUTPUT_PATH + CSV_OUTPUT_FILE_NAME)
    print('done')
