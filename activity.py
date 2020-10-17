import scipy.io as scio
from scipy import signal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

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
    return data_freq

def timestamp_to_str(timestamp):
    if timestamp == "":
        return ""
    time_local = time.localtime(timestamp)
    return time.strftime("%H:%M:%S",time_local)

class Activity:
    end_time = 0
    recognition_time = 0
    data_time = None
    start_time = ""
    end_time = ""
    inference_time = ""
    act = ""
    user =""

    # scaler = MinMaxScaler(feature_range=(0, 1))
    def __init__(self, start_time,):
        self.start_time = start_time
        self.isStart = True
        self.activity_dict()
    
    def set_csi(self, end_time, csi_data):
        self.end_time = end_time
        csi_data = np.absolute(csi_data)
        csi_data = np.reshape(csi_data, (-1, 325))
        # csi_data = self.scaler.fit_transform(csi_data)
        self.data_time = csi_data
        self.data_freq = get_data_freq(csi_data)
        self.data_time = np.expand_dims(self.data_time, axis = (0, -1))
        self.data_freq = np.expand_dims(self.data_freq , axis = (0, -1))

    def recognize(self, inference_time, act, user):
        self.inference_time = inference_time
        self.act = act
        self.user = user
        self.activity_dict()

    def activity_dict(self):
        if self.inference_time != "":
            self.inference_time = round(self.inference_time, 5)

        self.to_string = {
            "start_time" : timestamp_to_str(self.start_time),
            "end_time" : timestamp_to_str(self.end_time),
            "inference_time": self.inference_time,
            "act": self.act,
            "user": self.user,
        }
        

