#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import savemat
from scipy.signal import butter, lfilter

# In[2]:


user_index = 3
activity_index = 1

# Parmeters
ntx=3 # Number of transmitting antennas
nrx=3 # Number of receiving antennas
subs=30 # Number of subcarriers
window = 10 # Non-overlapped window applied to the cached CSI
sampling_rate = 125 # Number of received WiFi packets per second


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def csi_segmentation_var(cache, tx, rx, N):
 
    # Split the cached CSI from the transmitting antenna tx 
    # and receiving antenna rx into non-overlapped chunks
    chunks = np.split(cache[0, 0, :, :], N, axis=1)
    
    # Variance for all N chunks
    chunk_mov_var = np.zeros(N)
    
    for i in range(N):
        chunk_mov_var[i] = np.mean(np.var(chunks[i], axis=1)) 
        
    plt.plot(chunk_mov_var)
    plt.show()


def csi_short_time_segmentation_realtime(cache, tx, rx, fs, threshold=1.5):
    '''
	Input: Real-time CSI sampled with a window with length k and a step size s
	Output: Segmented human activity
	Algorithm: 
    - Calculate variance/spectrogram for each short frame 
    - Plot the real-time variance/spectrogram
    - Detect the beginning of an activity if it is over a predefined threshold 
    - Cache the CSI within T seconds
    - Save the cached CSI as files
    
    cache.length = fs
    '''


def csi_segment_save_label(cache, u_index, act_index, directory, segment_index):
  
    # data_dic = {"data_time": segments[i][0], "data_freq": segments[i][1], "user_label": u_index, "activity_label": act_index, "peak_energy": peak_energy[i]}
    data_dic = {"data_time": cache, "user_label": u_index, "activity_label": act_index}
    savemat(directory + 'bev_seg_' + str(segment_index) + '.mat', data_dic)
    segment_index = segment_index + 1 
    return segment_index


    
def csi_segmentation_spectrogram_realtime(cache, tx, rx, fs, N_subcarrier=20, nperseg_=80, noverlap_=40, threshold=1.5, seg_length=100):
    '''
    Extract realtime segments for training CNN model
    Extract CSI spectrogram 

    Output: behavior segments 
    Input parameters:
        cache: most recent CSI measurement
        tx: transmitting antenna for CSI extraction
        rx: receiving antenna for CSI extraction 
        N_subcarrier: number of subcarrier used for data segmentation
        nperseg_: window length for spectrogram calculation
        threshold: threshold for behavior segmentation
    '''
     
    '''
    Convert complex CSI in cache into real values (for activity detection)
    '''
    pilot_signal = np.absolute(cache[tx, rx, 0:N_subcarrier, :])

    '''
    Reshape cache and extract mplitude measureents (for extracting CSI segments)
    '''
    cache_amplitude = np.absolute(np.reshape(cache, [cache.shape[0]*cache.shape[1]*cache.shape[2], cache.shape[3]]))
    for i in range(cache_amplitude.shape[0]):
        cache_amplitude[i,:] = butter_bandpass_filter(cache_amplitude[i,:], 0.5, 60, sampling_rate, order=2)

    print('cache_amplitude', cache_amplitude.shape)

    ''' 
    Calculate spectrogram given frame length (nperseg_)
    The length of the FFT is equaled to nperseg_ 
    Accumulate the spectrograms from N_subcarrier
    '''
    spectrogram_acc = []
    for i in range(N_subcarrier):
        _, _, spec = signal.spectrogram(pilot_signal[i, :], fs, nperseg=nperseg_, nfft=nperseg_, noverlap=noverlap_)
        if i==0:
            spectrogram_acc = spec[0:int(nperseg_/2),:]
        else:
            spectrogram_acc += spec[0:int(nperseg_/2),:]

    flag = True
    spectrogram_all = []
 
    for i in range(cache_amplitude.shape[0]): 
        _, _, temp = signal.spectrogram(cache_amplitude[i, :], fs, nperseg=nperseg_, nfft=nperseg_, noverlap=noverlap_) 
        temp = np.expand_dims(temp, axis=0) 
        if flag == True:
            spectrogram_all = temp
            flag = False
        else: 
            spectrogram_all = np.concatenate([spectrogram_all, temp], axis=0)
        
    print('spectrogram_all:', spectrogram_all.shape) 
    
    '''
    Calculate the spectrogram average 
    '''
    spectrogram_acc = np.mean(spectrogram_acc, axis=0)
 
    '''
    Detect peaks in the spectrogram average and segment activity

    peak_energy: save peak energy of each segment as a unique identifier of each behavior segment
    '''
    peaks, _ = signal.find_peaks(spectrogram_acc, distance=20, height = threshold)
    plt.plot(spectrogram_acc)
    plt.plot(peaks, spectrogram_acc[peaks], "x")
    plt.ylim((0, 20))
    plt.show()    
    print('spectrogram_acc[peaks]:', spectrogram_acc[peaks])
    peak_energy = spectrogram_acc[peaks]
      
    '''
    Map the peaks into time-domain indexes
    ''' 
    peaks_time = peaks * pilot_signal.shape[1]/spectrogram_acc.shape[0] 
    peaks_time = peaks_time.astype(int)  

    '''
    Map the peaks into frequency-domain indexes
    ''' 
    preaks_freq = peaks

#     plt.plot(pilot_signal[0, :])
#     plt.plot(peaks, pilot_signal[0, peaks], "x") 
#     plt.show()  
    
    '''
    Segment the CSI amplitude center at each peak
    Segment the CSI spectrogram 
    '''  
    seg_length_freq = int(float(spectrogram_all.shape[2])/float(cache_amplitude.shape[1])*seg_length)
    
    segments = [] 
    for i in range(len(peaks_time)):
        '''
        Time- and frequency-domain CSI segmentation
        '''
        if peaks_time[i]-seg_length<0:
            continue
        elif peaks_time[i]+seg_length>pilot_signal.shape[1]-1:
            continue
        else:
            temp = np.absolute(cache_amplitude[:, peaks_time[i]-seg_length:peaks_time[i]+seg_length])   
            
            temp_freq = np.absolute(spectrogram_all[:, 0:40, preaks_freq[i]-seg_length_freq:preaks_freq[i]+seg_length_freq])
            temp_freq = np.reshape(temp_freq, [temp_freq.shape[0], -1], order='F')  

            segments.append([temp, temp_freq])
            plt.imshow(temp)
            plt.show()

            plt.imshow(temp_freq)
            plt.show()

    return segments
 

def csi_segmentation_spectrogram_label(cache, tx, rx, fs, u_index, act_index, directory, segment_index, N_subcarrier=20, nperseg_=80, noverlap_=40, threshold=1.5, seg_length=100):
    '''
    Extract realtime segments for training CNN model
    

    Output: behavior segments 
    Input parameters:
        cache: most recent CSI measurement
        tx: transmitting antenna for CSI extraction
        rx: receiving antenna for CSI extraction
        u_index: user index in the system
        act_index: activity index in the system
        N_subcarrier: number of subcarrier used for data segmentation
        nperseg_: window length for spectrogram calculation
        threshold: threshold for behavior segmentation
    '''
     
    '''
    Convert complex CSI in cache into real values (for activity detection)
    '''
    pilot_signal = np.absolute(cache[tx, rx, 0:N_subcarrier, :])

    '''
    Reshape cache and extract mplitude measureents (for extracting CSI segments)
    '''
    cache_amplitude = np.absolute(np.reshape(cache, [cache.shape[0]*cache.shape[1]*cache.shape[2], cache.shape[3]]))
    for i in range(cache_amplitude.shape[0]):
        cache_amplitude[i,:] = butter_bandpass_filter(cache_amplitude[i,:], 0.5, 60, sampling_rate, order=2)

    print('cache_amplitude', cache_amplitude.shape)

    ''' 
    Calculate spectrogram given frame length (nperseg_)
    The length of the FFT is equaled to nperseg_ 
    Accumulate the spectrograms from N_subcarrier
    '''
    spectrogram_acc = []
    for i in range(N_subcarrier):
        _, _, spec = signal.spectrogram(pilot_signal[i, :], fs, nperseg=nperseg_, nfft=nperseg_, noverlap=noverlap_)
        if i==0:
            spectrogram_acc = spec[0:int(nperseg_/2),:]
        else:
            spectrogram_acc += spec[0:int(nperseg_/2),:]

    flag = True
    spectrogram_all = []
 
    for i in range(cache_amplitude.shape[0]): 
        _, _, temp = signal.spectrogram(cache_amplitude[i, :], fs, nperseg=nperseg_, nfft=nperseg_, noverlap=noverlap_) 
        temp = np.expand_dims(temp, axis=0) 
        if flag == True:
            spectrogram_all = temp
            flag = False
        else: 
            spectrogram_all = np.concatenate([spectrogram_all, temp], axis=0)
        
    print('spectrogram_all:', spectrogram_all.shape) 
    
    '''
    Calculate the spectrogram average 
    '''
    spectrogram_acc = np.mean(spectrogram_acc, axis=0)
 
    '''
    Detect peaks in the spectrogram average and segment activity

    peak_energy: save peak energy of each segment as a unique identifier of each behavior segment
    '''
    peaks, _ = signal.find_peaks(spectrogram_acc, distance=20, height = threshold)
    plt.plot(spectrogram_acc)
    plt.plot(peaks, spectrogram_acc[peaks], "x")
    plt.ylim((0, 20))
    plt.show()    
    print('spectrogram_acc[peaks]:', spectrogram_acc[peaks])
    peak_energy = spectrogram_acc[peaks]
      
    '''
    Map the peaks into time-domain indexes
    ''' 
    peaks_time = peaks * pilot_signal.shape[1]/spectrogram_acc.shape[0] 
    peaks_time = peaks_time.astype(int)  

    '''
    Map the peaks into frequency-domain indexes
    ''' 
    preaks_freq = peaks

#     plt.plot(pilot_signal[0, :])
#     plt.plot(peaks, pilot_signal[0, peaks], "x") 
#     plt.show()  
    
    '''
    Segment the CSI amplitude center at each peak
    Segment the CSI spectrogram 
    '''  
    seg_length_freq = int(float(spectrogram_all.shape[2])/float(cache_amplitude.shape[1])*seg_length)
    
    segments = [] 
    for i in range(len(peaks_time)):
        '''
        Time- and frequency-domain CSI segmentation
        '''
        if peaks_time[i]-seg_length<0:
            continue
        elif peaks_time[i]+seg_length>pilot_signal.shape[1]-1:
            continue
        else:
            temp = np.absolute(cache_amplitude[:, peaks_time[i]-seg_length:peaks_time[i]+seg_length])   
            
            temp_freq = np.absolute(spectrogram_all[:, 0:40, preaks_freq[i]-seg_length_freq:preaks_freq[i]+seg_length_freq])
            temp_freq = np.reshape(temp_freq, [temp_freq.shape[0], -1], order='F')  

            segments.append([temp, temp_freq])
            plt.imshow(temp)
            plt.show()

            plt.imshow(temp_freq)
            plt.show()

    
    for i in range(len(segments)):
        print('segments[i][0]', segments[i][0].shape)
        print('segments_freq[i][1]', segments[i][1].shape)  
        data_dic = {"data_time": segments[i][0], "data_freq": segments[i][1], "user_label": u_index, "activity_label": act_index, "peak_energy": peak_energy[i]}
        print('peak_energy[i]:', peak_energy[i])
        savemat(directory + 'bev_seg_' + str(segment_index) + '.mat', data_dic)
        segment_index = segment_index + 1 
     
    return segment_index
 