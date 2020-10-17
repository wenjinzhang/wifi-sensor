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

    dict_class_id = {}

    dict_user = {}

    list_act_dataset = []

    dict_users_dataset = {} # key => dataset means classId => dataset(mat, userId)

    classId = 0
    for folder in folders:
        if folder == ".DS_Store":
            continue

        class_path = os.path.join(path, folder)

        dict_class_id[classId] = folder
        # print("----->>>>>path", class_path)
        user_id = 0

        dict_users_dataset[folder] = []
        for user in sorted(os.listdir(class_path)):
            if user == ".DS_Store":
                continue
            dict_user[user_id] = user
            user_path = os.path.join(class_path, user)
            
            for file_name in os.listdir(user_path):
               file_path = os.path.join(user_path, file_name) 
               if not file_path.endswith(".DS_Store"):
                    # print(file_path)
                    meta_data = scio.loadmat(file_path)

                    time_data = meta_data['data_time']
                    sample = (time_data, classId)
                    list_act_dataset.append(sample)

                    dict_users_dataset[folder].append((time_data, user_id))
                    # print(file_path, classId, user_id, folder)

            user_id += 1
        
        # print(folder, classId)
        classId += 1

    return dict_class_id, dict_user, list_act_dataset, dict_users_dataset 


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


def generate_dataset(dataset, scaler):
    # mix data
    random.shuffle(dataset)
    data_time = []
    data_freq = []
    labels = []
    for features, label in dataset:
        if features.shape == (3, 3, 30, 325):
            features = np.absolute(features)
            features = np.reshape(features, (-1, 325))
            # features = scaler.fit_transform(features)
            
            data_time.append(features)
            data_freq.append(get_data_freq(features))
            labels.append(label)

    data_time = np.array(data_time)
    data_time = np.expand_dims(data_time, axis=-1)
    data_freq = np.array(data_freq)
    data_freq = np.expand_dims(data_freq, axis=-1)

    return (data_time, data_freq, np.array(labels))


def save_to_pickle(data, file_name):
    pickle_out = open("./pickle_data/{}.pickle".format(file_name),"wb")
    pickle.dump(data, pickle_out)
    # pickle.dump((train_data_time, train_data_freq, train_label, 
    #         test_data_time, test_data_freq, test_label ),
    #         pickle_out)
    pickle_out.close()

if __name__=="__main__":
    scaler = MinMaxScaler(feature_range=(0, 1))
    dict_class_id, dict_user, list_act_dataset, dict_users_data = load_metadate(path_training)
    tuple_act_dataset = generate_dataset(list_act_dataset, scaler)
    save_to_pickle(tuple_act_dataset, "activity")

    for key in dict_users_data:
        dataset = generate_dataset(dict_users_data[key], scaler)
        save_to_pickle(dataset, key)

    save_to_pickle((dict_class_id, dict_user), "mapping")
