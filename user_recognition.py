import os
import socket
import sys
import time
import re
import csi_decoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pickle
from activity import Activity
import requests

matplotlib.use('TkAgg')
from keras.models import load_model
from activity_segmentation import csi_segment_save_label
weberver = 'http://127.0.0.1:5000/activity'

# load models
def load_models(model_type):
    model_path = "./model/{}".format(model_type)
    model_ext = load_model(os.path.join(model_path, "feature.h5"))
    model_actrec = load_model(os.path.join(model_path, "classify.h5"))
    return (model_ext, model_actrec)

# model inference
pickle_in = open("./pickle_data/mapping.pickle", "rb")
(dict_class_id, dict_user) = pickle.load(pickle_in)

dict_models = {"activity": load_models("activity")}

for key in dict_class_id:
    model_type = dict_class_id[key]
    dict_models[model_type] = load_models(model_type)


def inference(activity):
    now = time.time()
    predict = dict_models["activity"][1](
            dict_models["activity"][0]([activity.data_time, activity.data_freq], training=False), 
            training=False
            )

    # index of activity
    classId = np.argmax(predict)
    # name of activity
    act = dict_class_id[classId]

    pre_user = dict_models[act][1](
                dict_models[act][0]([activity.data_time, activity.data_freq], training=False), 
                training=False
                )

    userId = np.argmax(pre_user)
    user = dict_user[userId]
    end_time = time.time()
    inference_time = end_time  - now
    activity.recognize(inference_time, act, user)
    return activity

'''
Parameters:
- HOST: IP address of the server computer
- PORT: port for TCP/IP connection (same port is used on the receiver)
- array_size: cache the recent TCP/IP packets for activity segmentation 
- segment_trigger: conduct segmentation for every segment_trigger samples
- var_thres: start data segmentation if CSI amplitude variance is over var_thres
- act_dur: durating of an activity to segment
'''

HOST = '192.168.1.100' 
PORT = 8080
array_size = 1000   
segment_trigger = 25 
sample_rate = 100
var_thres = 15 
act_dur = sample_rate * 3
delay = 3

'''
user_index: index of the user in the system
activity_index: index of the activity in the system
'''
user_index = 1
activity_index = 0
save_dir = './data/'

'''
Initializate figure  parameters to plot CSI in real-time
'''
plt.ion()
plt.figure(1)
his_var = [] # y-axis
plot_data_size = 30 


def realtime_csi(HOST, PORT, array_size, segment_trigger, ntx=3, nrx=3, subcarriers=30, var_thres_=20, act_dur_=200):
    '''
    Setup a TCP/IP connection between the receiver and the server computer
    Decode the received TCP/IP packets and extract CSI in real-time
    
    Parameters:
    - HOST: IP address of the server computer
    - PORT: port for TCP/IP connection (same port is used on the receiver)
    - array_size: cache the recent TCP/IP packets for activity segmentation
    - ntx: number of transmitting antennas 
    - nrx: number of receiving antennas
    - subcarriers: number of subcarriers
    '''
    
    '''
    Create a cache for caching most recent CSI values 
    '''
    csi_array = np.zeros((ntx, nrx, subcarriers, array_size), dtype=complex)
    
    
    '''
    Create a cache for caching the variance 
    '''
    # var_array = np.zeros((int(array_size/segment_trigger)))
    
    '''
    Create a socket object for TCP/IP communication
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen()
    conn, addr = sock.accept()
    
    '''
    Create an CSI decoder object to extract CSI from TCP/IP packets
    '''
    decoder = csi_decoder.CsiDecoder()
    
    count = 0
     
    '''
    Flag to remove the first detected segment: operating on the device
    '''
    flag = 0

    seg_count = 0
    seg_flag = False

    start_time = time.time()

    while True:
        try:
            '''
            Obtain the length of CSI in bytes
            '''
            length = conn.recv(2)
            length = int.from_bytes(length, "big")  
            
            '''
            Decode the received TCP/IP packet and extract CSI (stored as complex values)
            Update the cache
            '''
            csi = conn.recv(length)
            csi_entry = decoder.decode(length, csi)  
            csi_array = np.roll(csi_array, 1, axis=3)

            # print(csi_entry.csi)
            csi_array[0:ntx, 0:nrx, 0:subcarriers, 0] = csi_entry.csi[0:ntx, 0:nrx, 0:subcarriers]  

            if time.time()- start_time < delay:
                continue

            if count % segment_trigger == 0:  

                '''
                Calculate variance for every segment_trigger data points
                '''
                temp = np.abs(csi_array[0, 0, 0, 0:segment_trigger]) 
                temp_var  = np.var(np.abs(csi_array[0, 0, 0, 0:segment_trigger])) 
                # start = time.time()             
                '''
                Plot the CSI measurements in real time
                '''
                if len(his_var) > plot_data_size:
                    plt.clf()
                    del his_var[0]
                else:
                    his_var.append(temp_var)
                    plt.plot(his_var, c='r',ls='-', marker='o', mec='b',mfc='w')  
                    plt.pause(0.000000001)

                '''
                Segment the activity with the threshold 
                '''
                if temp_var > var_thres_ and not seg_flag:
                    print('Detect activity')
                    seg_flag = True
                    activity = Activity(time.time())
                    print(activity.to_string)
                    requests.post(weberver, data=activity.to_string)

                elif seg_flag == True:
                    seg_count = seg_count + 1 
                    if temp_var < var_thres_ and seg_count > int(act_dur_/segment_trigger):
                        if int(seg_count * segment_trigger) < array_size:
                            temp = csi_array[:, :, :, 0:int(seg_count*segment_trigger)]
                        else:
                            temp = csi_array[:, :, :, :]
                        print('temp', temp.shape)
                        # segment_index_ = csi_segment_save_label(temp, u_index=user_index, act_index=activity_index, directory=save_dir, segment_index=time.time())
                        # segment_index = segment_index_
                        if temp.shape == (3, 3, 30, 325):
                            activity.set_csi(time.time(), temp)
                            activity = inference(activity)
                            requests.post(weberver, data=activity.to_string)
                            print(activity.to_string)
                        seg_count = 0
                        seg_flag = False

            count = count + 1
        except OSError:
            continue
        
        except KeyboardInterrupt:
            conn.close()
            break
        
    conn.close()    
    sock.close()

if __name__ == '__main__':
    realtime_csi(HOST, PORT, array_size, segment_trigger, ntx=3, nrx=3, subcarriers=30, var_thres_=var_thres, act_dur_=act_dur)