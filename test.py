from data import create_test_data
import pandas as pd
from keras.models import load_model


# Params Settings:
max_sentences = 151
max_words = 20
num_keypoints = 17
batch_size = 8
test_data_path = './data/test/'
ouput_path = './output/models/'
model_path = r'output\models\Adam_Vs_GRU\HAN_violance_detection_model_SGD_50_epochs'

def test(model):
    x_test, y_test = create_test_data(test_data_path, max_sentences, max_words, num_keypoints)
    results =  model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f'test_loss: {results[0]}, test accuracy: {results[1]}')


if __name__ == '__main__':
    model = load_model(model_path)
    test(model)