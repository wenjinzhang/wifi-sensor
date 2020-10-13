from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import os
import scipy.io as scio
import numpy as np
from scipy import signal
import threading
from activity import Activity
import time
import socket
import csi_decoder
# import csi_receiver
import multiprocessing
import queue
import pickle
shared_resource_lock = threading.Lock()
# global data to store the history csi
queue = queue.Queue()
# a dict list {start_time, end_time, inference time, type}
activities = [
]

sample_rate = 100
act_dur = sample_rate * 3

kwargs = {
    "host" : '192.168.1.100',
    "port" : 8080,
    "array_size" : 1000,
    "act_dur" : act_dur,
}

def timestamp_to_str(timestamp):
    if timestamp == "":
        return ""
    time_local = time.localtime(timestamp)
    return time.strftime("%H:%M:%S",time_local)

def activity_dict(start_time="", end_time="", inference_time="", act="", user=""):
    if inference_time != "":
        inference_time = round(inference_time, 5)
        
    return {
        "start_time" : timestamp_to_str(start_time),
        "end_time" : timestamp_to_str(end_time),
        "inference_time": inference_time,
        "act": act,
        "user": user,
    }

def realtime_csi(host, port, array_size, act_dur=200, segment_trigger=25, ntx=3, nrx=3, subcarriers=30, var_thres_=25):
    global shared_resource_lock, activity, activities
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
    - var_thres: start data segmentation if CSI amplitude variance is over var_thres
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
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    conn, addr = sock.accept()
    
    '''
    Create an CSI decoder object to extract CSI from TCP/IP packets
    '''
    decoder = csi_decoder.CsiDecoder()
    
    count = 0

    seg_count = 0
    seg_flag = False    
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
                # if len(his_var) > plot_data_size:
                #     plt.clf()
                #     del his_var[0]
                # else:
                #     his_var.append(temp_var)
                #     plt.plot(his_var, c='r',ls='-', marker='o', mec='b',mfc='w')  
                #     plt.pause(0.000000001)

                '''
                Segment the activity with the threshold 
                '''
                if temp_var > var_thres_ and not seg_flag:
                    print('Detect activity')
                    activity = Activity(time.time())
                    shared_resource_lock.acquire()

                    # activities.append(activity_dict(activity.start_time))
                    activities.insert(0, activity_dict(activity.start_time))
                    shared_resource_lock.release()

                    seg_flag = True
                elif seg_flag == True:
                    seg_count = seg_count + 1 
                    if temp_var < var_thres_ and seg_count > int(act_dur/segment_trigger):
                        if int(seg_count * segment_trigger) < array_size:
                            temp = csi_array[:, :, :, 0:int(seg_count*segment_trigger)]
                        else:
                            temp = csi_array[:, :, :, :]
                        print('temp', temp.shape)
                        # segment_index_ = csi_segment_save_label(temp, u_index=user_index, act_index=activity_index, directory=save_dir, segment_index=segment_index_)
                        # segment_index = segment_index_ 
                        shared_resource_lock.acquire()
                        if temp.shape != (3, 3, 30, 325):
                            del activities[0]
                        else:
                            activity.set_csi(time.time(), temp)
                            queue.put(activity)
                        shared_resource_lock.release()
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


def inference():
    global shared_resource_lock, activity, activities

    while True:
        if not queue.empty():
            activity = queue.get()
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

            inference_time = time.time() - now
            
            shared_resource_lock.acquire()
            activities[0] = activity_dict(activity.start_time, activity.end_time, inference_time, act, user)
            shared_resource_lock.release()


app = Flask('__WIFI sensor__')
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/show_act", methods=['GET'])
def show_act():
    return jsonify(result=activities)

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    # csi_receiver = multiprocessing.Process(target=csi_receiver.realtime_csi, args=queue, kwargs=csi_receiver.kwargs)
    csi_receiver = threading.Thread(target=realtime_csi, kwargs=kwargs)
    classifer = threading.Thread(target=inference)
    csi_receiver.start()
    classifer.start()
    app.run(host='127.0.0.1', port=5000, debug=False)
