import os
import numpy as np
import tensorflow as tf 
from scipy.io import savemat
from scipy.io import loadmat
import pickle
from dnn_models import feature_extractor_base, user_recognizer, grad_base

'''
Parameters for training
  
batch_size: batch size
num_epoch: number of iteration for training
data_dir: folder to store the training data
train_test_split: ratio of the training data in the whole dataset
'''  
batch_size = 8
num_epoch = 30
data_dir = './data/'
train_test_split = 0.8
legitimate_user = 0

checkpoint_feature_extractor = "./model/feature_extractor/feature_extractor_cp.ckpt"
checkpoint_user_recognizer = "./model/user_recognizer/user_recognizer_cp.ckpt"
checkpoint_activity_recognitor = "./model/activity_recognitor/activity_recognitor_cp.ckpt"

pretrained_model_ext = "./model/ext.h5"
pretrained_model_actrec = "./model/actrec.h5"



# checkpoint_feature_extractor_verification = "./model/feature_extractor/feature_extractor_verf.ckpt"
# checkpoint_user_recognizer_verification = "./model/user_recognizer/user_recognizer_verf.ckpt"
# checkpoint_user_recognizer_verification = "./model/user_recognizer/user_recognizer_verf.ckpt"

def read_data(directory):
    '''
    Data loading
    Read the .mat files storing a dictionary with the following entries:
    - "data": segments
    - "user_label": user_label
    - "activity_label": activity_label
    
    Output: 
    - data: behavior segments with shape (n_samples, n_subcarrer, WiFi packets, channel(amplitude, relative amplitude))
    - label: storing user label/user index
    '''
    
    data_time = []
    data_freq = []
    label = []
    flag = True

    '''
    Read each .mat file in the directory
    Combine all data
    Combine all labels
    '''
    peak_energy = 0
    for file in os.listdir(directory):  
        if file[-4:] == '.mat':
            content = loadmat(directory + file) 
            
            ''' 
            The segmentation may involve multiple samples with the same data
            Compare peak_energy and skip the sample for training if the peak_energy is the same as the pervious sample
            ''' 
            if content['peak_energy'][0][0] == peak_energy:
                continue;
            else:
                peak_energy = content['peak_energy'][0]
            
            '''
            Extract data and labels
            Stack measurements of all subcarriers 
            '''
            temp_data_time = content['data_time']  
            temp_data_time = np.expand_dims(temp_data_time, axis=0)
            temp_data_time = np.absolute(temp_data_time)

            temp_data_freq = content['data_freq']  
            temp_data_freq = np.expand_dims(temp_data_freq, axis=0)
            temp_data_freq = np.absolute(temp_data_freq)
            
            temp_label = content['user_label'][0] 
    
            # print(temp_data[0, 0, 0], temp_label)
            if flag:
                data_time = temp_data_time
                data_freq = temp_data_freq
                label = temp_label
                flag = False
            else:
                data_time = np.concatenate([data_time, temp_data_time], axis=0)
                data_freq = np.concatenate([data_freq, temp_data_freq], axis=0)
                label = np.concatenate([label, temp_label], axis=0)   

    '''
    Append the 3rd dimension since CNN requires 3D input
    '''            
    data_time = np.expand_dims(data_time, axis=3)   
    data_freq = np.expand_dims(data_freq, axis=3)

    '''         
    Number of users in the profile
    '''   
    n_users = np.unique(label)
    n_users = n_users.shape[0]
    
    return data_time, data_freq, label, n_users




if __name__ == "__main__":
    # data_time, data_freq, label, n_users = read_data(data_dir)

    pickle_in = open("dataset.pickle","rb")
    (train_data_time, train_data_freq, train_label, test_data_time, test_data_freq, test_label) = pickle.load(pickle_in)
    train_data_time = np.expand_dims(train_data_time, axis = -1)
    train_data_freq = np.expand_dims(train_data_freq, axis = -1)
    ''' ===================================================================================================
    Training user identification model
    '''
    print('Training user identification model')
    print('data_time:', train_data_time.shape)
    print('data_freq:', train_data_freq.shape)
    print('label:', train_label.shape)

    print(train_label)
    
    '''
    Construct dataset for training
    '''
    n_samples = train_data_time.shape[0]
    train_label = tf.one_hot(train_label, depth=3) 
    test_label = tf.one_hot(test_label, depth=3) 
    dataset = tf.data.Dataset.from_tensor_slices((train_data_time, train_data_freq, train_label))
    
    dataset_train = dataset.take(round(n_samples * train_test_split))
    n_sample_test = n_samples - round(n_samples * train_test_split)
    dataset_test = dataset.skip(round(n_samples * train_test_split))
    
    '''
    CNN model construction
    '''
    h_time = train_data_time.shape[1]
    w_time = train_data_time.shape[2]
    c_time = train_data_time.shape[3] 

    h_freq = train_data_freq.shape[1]
    w_freq = train_data_freq.shape[2]
    c_freq = train_data_freq.shape[3]
    
    input_dim = {"h_time": h_time, "w_time": w_time, "c_time": c_time, "h_freq": h_freq, "w_freq": w_freq, "c_freq": c_freq, "n_users": 3}
    savemat('./model/input_dim.mat', input_dim)
    '''
    model_ext: extract behavioral features
    model_usr: classifier to identify user
    '''
    model_ext = feature_extractor_base((h_time, w_time, c_time), (h_freq, w_freq, c_freq))
    model_usr = user_recognizer(3)

    for layer in model_ext.layers:
        layer.trainable = True
    for layer in model_usr.layers:  
        layer.trainable = True

    print('Feature Extractor:')
    model_ext.summary()

    print('User recognizor:')
    model_usr.summary()
    
    '''
    Training feature extractor and user recognizer
    '''

    optimizer_feat = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_usr = tf.keras.optimizers.Adam(learning_rate=0.001)

    best_result = 0

    for epoch in range(num_epoch):

        '''
        Metrics of intermediate results
        '''
        epoch_loss_feat = tf.keras.metrics.Mean()  
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for sample_src_time, sample_src_freq, label_src in dataset_train.shuffle(n_samples).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE):

    #         print('sample_src_time', sample_src_time.shape)    
    #         print('sample_src_freq', sample_src_freq.shape)    

            loss, grad_feat, grad_usr = grad_base(model_ext, model_usr, [sample_src_time, sample_src_freq], label_src) 

            optimizer_feat.apply_gradients(zip(grad_feat, model_ext.trainable_variables)) 
            optimizer_usr.apply_gradients(zip(grad_usr, model_usr.trainable_variables))  

            '''
            Calculate the immediate loss and user identification accuracy
            '''
            # print(sample_src_time.shape)
            # print(sample_src_freq.shape)
            temp_predictions = model_usr(model_ext([sample_src_time, sample_src_freq], training=True), training=True)  

            epoch_accuracy(label_src, temp_predictions) 
            epoch_loss_feat(loss) 

        if epoch % 1 == 0:
            display = 'Epoch {:03d}:, Loss user: {:.3f}, activity_recognition_accuracy: {:.3%}'.format(epoch, 
                                                                     epoch_loss_feat.result(),
                                                                     epoch_accuracy.result())
            print(display)

            # if epoch_accuracy.result()>best_result:
            model_ext.save_weights(checkpoint_feature_extractor)
            model_usr.save_weights(checkpoint_user_recognizer)

    model_ext.save(pretrained_model_ext)
    model_usr.save(pretrained_model_actrec)
