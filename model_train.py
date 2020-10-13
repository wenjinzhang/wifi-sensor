import os
import numpy as np
import tensorflow as tf 
from scipy.io import savemat
from scipy.io import loadmat
import pickle
from dnn_models import feature_extractor_base, user_recognizer, grad_base


def train(model_type, no_of_class, batch_size=8, num_epoch=2):
    '''
    Parameters for training
    batch_size: batch size
    num_epoch: number of iteration for training
    '''
    model_path = "./model/{}".format(model_type)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load training data
    pickle_in = open("./pickle_data/{}.pickle".format(model_type), "rb")
    (train_data_time, train_data_freq, train_label) = pickle.load(pickle_in)
  
    ''' ===================================================================================================
    Training user identification model
    '''
    print('Training a model for ', model_type)

    print('data_time:', train_data_time.shape)
    print('data_freq:', train_data_freq.shape)
    print('label:', train_label.shape)
    
    '''
    Construct dataset for training
    '''
    n_samples = train_data_time.shape[0]
    train_label = tf.one_hot(train_label, depth=no_of_class) 
    dataset = tf.data.Dataset.from_tensor_slices((train_data_time, train_data_freq, train_label))

    dataset_train = dataset.take(round(n_samples * 0.9))
    
    '''
    CNN model construction

    model_ext: extract behavioral features
    model_usr: classifier to identify user
    '''
    model_ext = feature_extractor_base(train_data_time.shape[1:], train_data_freq.shape[1:])
    model_usr = user_recognizer(no_of_class)

    # for layer in model_ext.layers:
    #     print(layer.trainable)
    #     layer.trainable = True
    # for layer in model_usr.layers:  
    #     layer.trainable = True

    # print('Feature Extractor:')
    # model_ext.summary()

    # print('recognizor model:')
    # model_usr.summary()
    
    '''
    Training feature extractor and recognizor model
    '''
    optimizer_feat = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_usr = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(num_epoch):
        '''
        Metrics of intermediate results
        '''
        epoch_loss_feat = tf.keras.metrics.Mean()  
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for sample_src_time, sample_src_freq, label_src in dataset_train.shuffle(n_samples).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE):

            loss, grad_feat, grad_usr = grad_base(model_ext, model_usr, [sample_src_time, sample_src_freq], label_src) 

            optimizer_feat.apply_gradients(zip(grad_feat, model_ext.trainable_variables)) 
            optimizer_usr.apply_gradients(zip(grad_usr, model_usr.trainable_variables))  

            '''
            Calculate the immediate loss and user identification accuracy
            '''
            temp_predictions = model_usr(model_ext([sample_src_time, sample_src_freq], training=True), training=True)  

            epoch_accuracy(label_src, temp_predictions) 
            epoch_loss_feat(loss) 

        display = 'Epoch {:03d}:, Loss user: {:.3f}, activity_recognition_accuracy: {:.3%}'.format(epoch, 
                                                                     epoch_loss_feat.result(),
                                                                     epoch_accuracy.result())
        print(display)

    model_ext.save(os.path.join(model_path, "feature.h5"))
    model_usr.save(os.path.join(model_path, "classify.h5"))



if __name__ == "__main__":
    pickle_in = open("./pickle_data/mapping.pickle", "rb")
    (dict_class_id, dict_user) = pickle.load(pickle_in)
    
    # train activity recognition model
    train("activity", len(dict_class_id))

    # train multiple user recognition model for each kind of activities
    no_of_users = len(dict_user)
    
    for key in dict_class_id:
        train(dict_class_id[key], no_of_users)

