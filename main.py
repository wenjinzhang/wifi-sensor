from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import os
import scipy.io as scio
import numpy as np
from scipy import signal

pretrained_model_ext = "./model/ext.h5"
pretrained_model_actrec = "./model/actrec.h5"
model_ext = load_model(pretrained_model_ext)
model_actrec = load_model(pretrained_model_actrec)


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

def get_data(path="./data"):
    folders = os.listdir(path)
    folders.sort(reverse=True)
    print(folders[0])
    meta_data = scio.loadmat(os.path.join(path, folders[0]))
    data_time = meta_data['data_time']
    if data_time.shape != (3,3,30, 325):
        return None, None
    data_time = np.absolute(data_time)
    data_time = np.reshape(data_time, (-1, 325))
    data_freq = get_data_freq(data_time)

    data_time = np.expand_dims(data_time, axis = (0, -1))
    data_freq = np.expand_dims(data_freq , axis = (0, -1))
    return np.array(data_time), np.array(data_freq)


app = Flask('__WIFI sensor__')

@app.route("/")
def index():
    return render_template('index.html')



@app.route("/show_act", methods=['GET'])
def show_act():
    data_time, data_freq = get_data()
    predict = model_actrec(model_ext([data_time, data_freq], training=False), training=False)
    acts = ["Raising hand", "Walking", "Squating"]
    classId = np.argmax(predict)
    print(acts[classId])
    data={}
    data['act'] = acts[classId]
    return jsonify(result=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)