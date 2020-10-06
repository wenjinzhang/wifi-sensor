import os
import socket
import sys
import time
import re
import csi_decoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from activity_segmentation import csi_segment_save_label


HOST = '192.168.1.100' 
PORT = 8080
array_size = 1000   
sample_rate = 100
act_dur = sample_rate * 3
save_dir = './data/'

'''
Initializate figure  parameters to plot CSI in real-time
'''
def realtime_csi(HOST, PORT, array_size, act_dur=200, segment_trigger=25, ntx=3, nrx=3, subcarriers=30, var_thres_=20):
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
    init the plot
    '''
    plt.ion()
    plt.figure(1)
    his_var = [] # y-axis
    plot_data_size = 30 


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
    realtime_csi(HOST, PORT, array_size, act_dur=act_dur)