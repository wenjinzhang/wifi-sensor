import os
import scipy.io as scio
import random
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy import signal

path_testing = "dataset/testing"
path_training = "dataset/training"


def load_metadate(path):
    folders = os.listdir(path)
    num_class = list(folders)
    dataset = []
    classId = 0
    for folder in folders:
        if folder == ".DS_Store":
            continue
        class_path = os.path.join(path, folder)
        for (root, dirs, files) in os.walk(class_path):
            for name in files:
                file_path = os.path.join(root, name)
                if not file_path.endswith(".DS_Store"):
                    # print(file_path)
                    meta_data = scio.loadmat(file_path)
                    # time_data = np.moveaxis(meta_data['data_time'], -1, 0)
                    time_data = meta_data['data_time']
                    sample = (time_data, classId)
                    dataset.append(sample)
                    print(file_path, classId)
        print(folder, classId)
        classId += 1
    return dataset

training_data = load_metadate(path_training)
testing_data = load_metadate(path_testing)

# for x, y in training_data[:10]:
#     print(x.shape)

# mix data
random.shuffle(training_data)
random.shuffle(testing_data)
scaler = MinMaxScaler(feature_range=(0, 1))


def get_data_freq(data_time):
    flag = True
    spectrogram_all = []
    for i in range(data_time.shape[0]): 
        _, _, temp = signal.spectrogram(data_time[i, :], 100, nperseg=80, nfft=80, noverlap=40) 
        temp = np.expand_dims(np.abs(temp[:40, :]), axis=0)
        if flag == True:
            spectrogram_all = temp
            flag = False
        else: 
            spectrogram_all = np.concatenate([spectrogram_all, temp], axis=0)
        
    data_freq = spectrogram_all
    data_freq = np.reshape(data_freq, [data_freq.shape[0], -1], order='F')  
    # data_time = np.expand_dims(data_time, axis=2)
    # data_freq = np.expand_dims(data_freq, axis=2)
    return data_freq

def generate_dataset(data, scaler):
    data_time = []
    data_freq = []
    labels = []
    for features, label in data:
        if features.shape == (3, 3, 30, 325):
            features = np.absolute(features)
            features = np.reshape(features, (-1, 325))
            # features = scaler.fit_transform(features)
            data_time.append(features)
            data_freq.append(get_data_freq(features))
            labels.append(label)

    return np.array(data_time), np.array(data_freq), np.array(labels)

train_data_time, train_data_freq, train_label = generate_dataset(training_data, scaler)
test_data_time, test_data_freq, test_label  = generate_dataset(testing_data, scaler)


pickle_out = open("dataset.pickle","wb")
pickle.dump((train_data_time, train_data_freq, train_label, 
            test_data_time, test_data_freq, test_label ),
            pickle_out)
pickle_out.close()