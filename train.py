import pandas as pd
from model_compilators import compile_model
from utils import count_videos
from utils import list_h5_files, count_videos
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import h5py



# Params Settings:
maxlen = 100
max_sentences = 151
max_words = 20
embedding_dim = 51
num_keypoints = 17
validation_split = 0.2
learning_rate = 0.0005
# learning_rate = 0.0001
epochs = 30
batch_size = 64
# batch_size = 32
data_path = './data/hockey/train'
test_data_path = './data/hockey/test/'
ouput_path = './output/'
model_output_filename = f'hockey_cm'
training_history_path = ouput_path + 'history/' + model_output_filename + '.csv'
test_results_path = ouput_path + 'history/' + model_output_filename + '_results.csv'
test_predictions_path = ouput_path + 'history/' + model_output_filename + '_predictions.csv'

def load_training_data():
    list_files = list_h5_files(data_path)
    n_videos = count_videos(data_path)
    print(f"There are {n_videos} videos.")
    data = np.zeros((n_videos, max_sentences, max_words, num_keypoints, 3))
    labels = np.zeros(n_videos)
    video_ind = 0
    for file_name in list_files:
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
                        labels[video_ind] = label
                        video_ind = video_ind + 1
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

    X_train, y_train = np.array(data), np.array(labels)
    X_test, y_test = None, None
    return X_train, X_test, y_train, y_test

def load_test_data():
    list_files = list_h5_files(test_data_path)
    n_videos = count_videos(test_data_path)
    print(f"There are {n_videos} videos.")
    data = np.zeros((n_videos, max_sentences, max_words, num_keypoints, 3))
    labels = np.zeros(n_videos)
    video_ind = 0
    for file_name in list_files:
        full_path = os.path.join(test_data_path, file_name)
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
                        labels[video_ind] = label
                        video_ind = video_ind + 1
        except:
            pass

    data = data.reshape((n_videos, max_sentences, max_words,  num_keypoints * 3))
    labels = to_categorical(np.asarray(labels))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    return data, labels

def train():
    n_videos = count_videos(data_path)

    X_train, X_val, y_train, y_val = load_training_data()

    model = compile_model(n_videos, max_words, max_sentences, lr=learning_rate)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
     # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(model.history.history) 
    hist_df.to_csv(training_history_path)

    X_test, y_test = load_test_data()
    print(f'xtest shape: {X_test.shape}')
    print(f'xtest sum: {np.sum(X_test)}')

    results =  model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f'test_loss: {results[0]}, test accuracy: {results[1]}')
    results_df = pd.DataFrame(results) 
    results_df.to_csv(test_results_path)

    
    print("Evaluate")

    from sklearn.metrics import confusion_matrix

    #Predict
    y_prediction = model.predict(X_test)

    print(y_prediction.shape)
    preds_df = pd.DataFrame(y_prediction) 
    preds_df.to_csv(test_predictions_path)
    truth_df = pd.DataFrame(y_test) 
    truth_df.to_csv(test_predictions_path+'y_test.csv')

   

    model.save(ouput_path + model_output_filename)

if __name__ == '__main__':
    train()