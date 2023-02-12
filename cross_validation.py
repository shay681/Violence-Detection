import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from utils import list_h5_files, count_videos
from sklearn.model_selection import KFold
from tensorflow.python.client import device_lib
from keras.models import Model

import tensorflow as tf
from grid_search import load_data_and_labels
from model_compilators import get_model_parts

# Set Params
K_FOLDS = 5
LEARNING_RATE = 0.0005
EPOCHS = 30
BATCH_SIZE = 64 
GRU_UNITS = 64
DENSE_DIM = [16]
DATA_PATH = './data/train/'
OUTPUT_PATH = './output/'
MODEL_OUTPUT_FILE_NAME = 'HAN'
CSV_OUTPUT_FILE_NAME = f'HAN_{K_FOLDS}_FOLDS.csv'
MAX_SENTENCES = 151
MAX_WORDS = 20
NUM_OF_KEYPOINTS = 17

def compile_model(dense_dim: int):
    opt = Adam(learning_rate=LEARNING_RATE)
    review_input, preds = get_model_parts(dense_dim)
    model = Model(review_input, preds)
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=opt)
    print('Model Adam optimizer with LR=', LEARNING_RATE)
    return model

if __name__ == '__main__':
    print(device_lib.list_local_devices())

    # Set Params
    k_folds = K_FOLDS
    learning_rate = LEARNING_RATE
    epochs_list = EPOCHS
    batch_sizes = BATCH_SIZE
    gru_units_list = GRU_UNITS
    dense_dims = DENSE_DIM

    # Create DataFrame for saving data
    df_data = []

    # Load Data
    data_list, label_list = load_data_and_labels(DATA_PATH)
    data_list = np.array(data_list)
    label_list = np.array(label_list)
    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for dense_dim in dense_dims:
        print('-------------------------------------------')
        print('*******************************************')
        print(f'************Starting with dense_dim={dense_dim}*************')
        print('*******************************************')
        print('-------------------------------------------')
        fold_no = 1
        for train, test in kfold.split(data_list, label_list):
            model = compile_model(dense_dim)
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            history = model.fit(data_list[train], label_list[train], batch_size=batch_sizes, epochs=EPOCHS)

            # Generate generalization metrics
            scores = model.evaluate(data_list[test], label_list[test])
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            df_entry = {
                "fold": fold_no,
                "loss": scores[0],
                "accaury": scores[1],
                "batch_size": batch_sizes,
                "epochs": epochs_list,
                "learning_rate": learning_rate,
                "dense_dim": dense_dim,
                "score_object": scores,
                "history_df": history.history,
                }
            df_data.append(df_entry)
            # Increase fold number
            fold_no = fold_no + 1
        df = pd.DataFrame(df_data)
        df.to_csv(OUTPUT_PATH + f'CV_DENSE_DIM2' + CSV_OUTPUT_FILE_NAME)
        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
            print('------------------------------------------------------------------------')
            print('Average scores for all folds:')
            print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
            print(f'> Loss: {np.mean(loss_per_fold)}')
            print('------------------------------------------------------------------------')
