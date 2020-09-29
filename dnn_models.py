import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import tensorflow.keras as keras 
from functools import partial, partialmethod
# from STFNets import STFLayer

_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 0.001
_N_NEURON = 1024
_N_Class = 5
_Filter_SIZE = 128


 
def feature_extractor_base(input_shape1, input_shape2):
    
    inputs = keras.Input(shape = input_shape1, name='input_time')
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(inputs)

    x = layers.Conv2D(_Filter_SIZE, (8, 16), strides=(4, 8), activation='relu',  padding='valid', data_format='channels_last', kernel_initializer='random_uniform')(x)
    
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x) 
    output = tf.keras.layers.Dropout(rate=0.5)(x)   
 
    x = layers.Conv2D(_Filter_SIZE, (4, 4), strides=(2, 2),  padding='valid', activation='relu', kernel_initializer='random_uniform')(output) 
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)  
    output = tf.keras.layers.Dropout(rate=0.5)(x)
 
    x = layers.Conv2D(_Filter_SIZE*2, (4, 4), strides=(2, 2),  padding='valid', activation='relu', kernel_initializer='random_uniform', name='last_conv')(output) 
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)  
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(_N_NEURON, activation='relu', kernel_initializer='random_uniform')(x)
    output_rep = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x) 


    inputs_FFT = keras.Input(shape = input_shape2, name='input_FFT') 
    f = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(inputs_FFT)

    f = layers.Conv2D(_Filter_SIZE, (8, 8), strides=(4, 4), activation='relu',  padding='valid', data_format='channels_last', kernel_initializer='random_uniform')(f)
    f = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(f) 
    output_f = tf.keras.layers.Dropout(rate=0.5)(f)   

    f = layers.Conv2D(_Filter_SIZE, (4, 4), strides=(2, 2),  padding='valid', activation='relu', kernel_initializer='random_uniform')(output_f) 
    f = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(f)  
    output_f = tf.keras.layers.Dropout(rate=0.5)(f) 

    f = layers.Conv2D(_Filter_SIZE*2, (4, 4), strides=(2, 2),  padding='valid', activation='relu', kernel_initializer='random_uniform', name='last_conv_FFT')(output_f) 
    f = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(f)  
    f = tf.keras.layers.Dropout(rate=0.5)(f)
    f = layers.Flatten()(f)
    f = layers.Dense(_N_NEURON, activation='relu', kernel_initializer='random_uniform')(f)
    output_rep_f = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(f) 

    output_rep = layers.concatenate([output_rep, output_rep_f])

    model = keras.Model(inputs=[inputs, inputs_FFT], outputs=output_rep) 

    return model 

  
def user_recognizer(n_user):
    # x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='user_b1')

    inputs = keras.Input(shape = (_N_NEURON*2,))

    x = layers.Dense(_N_NEURON, activation='relu', kernel_initializer='random_uniform')(inputs)
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x) 
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = layers.Dense(_N_NEURON, activation='relu', kernel_initializer='random_uniform', name='abstract_layer')(x)
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x) 
    x = tf.keras.layers.Dropout(rate=0.5)(x) 

    output_ur = layers.Dense(n_user, activation='softmax')(x) 

    model = keras.Model(inputs=inputs, outputs=output_ur)  

    return model



'''
Adversarial Discriminator - Domain Discriminator
'''

@tf.custom_gradient
def GradientReversalOperator(x):
	def grad(dy):
		return -1 * dy
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(GradientReversalLayer, self).__init__()
		
	def call(self, inputs):
		return GradientReversalOperator(inputs)

def gradient_flip():
    inputs = keras.Input(shape = (_N_NEURON,))
    output = GradientReversalLayer()(inputs)
    flip = keras.models.Model(inputs=inputs, outputs=output)     
    return flip


def adversarial_discriminator(n_domain, n_class): 
 
    inputs = keras.Input(shape = (_N_NEURON*2))
    x = GradientReversalLayer()(inputs)
    x = keras.layers.Dense(_N_NEURON, activation='relu', kernel_initializer='random_uniform')(x) 
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x) 
    x = tf.keras.layers.Dropout(rate=0.5)(x)   
    
    x = keras.layers.Dense(_N_NEURON, activation='relu', kernel_initializer='random_uniform')(x) 
    x = layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x) 
    x = tf.keras.layers.Dropout(rate=0.5)(x)    
    
    classifier_d = keras.layers.Dense(n_domain, activation='softmax')(x)

    adv_model = keras.models.Model(inputs=inputs, outputs=classifier_d)     

    return adv_model 

 

'''
Loss Function
'''

def loss_balance(y_, n_class_dist):
    '''
    y_: logits of unlabeled data
    '''  
    y_d = tf.math.reduce_sum(y_, axis=0)/tf.math.reduce_sum(y_) 
    return tf.keras.losses.KLD(y_d, n_class_dist)


def loss_confidence(y_):
    L_c = -tf.reduce_mean(tf.math.log(y_) + tf.math.log(1-y_)) 
    return L_c

def loss_categorical_crossentropy(y, y_):
    return tf.keras.losses.categorical_crossentropy(y, y_, from_logits=False)
 
def loss_domain(model_dom, x, y, training):
    # print(y)
    y_ = model_dom(x, training=training) 
    return loss_categorical_crossentropy(y, y_) 
  
 

def entropy(model_ext, model_usr, model_dom, x, training):
    epsilon = 1e-5

    y_hat = model_usr(model_ext(x, training=training), training=training) 
    H = tf.math.multiply(tf.math.negative(y_hat), tf.math.log(y_hat+epsilon)) 
    H = tf.reduce_sum(H, axis=1)
    #W = 1.0 + tf.math.exp(-H) 
    H = tf.math.exp(-H) 
    return tf.reshape(H, shape=(H.shape[0],1))

def loss_label(model_ext, model_usr, x, y, training):  
 
    y_ = model_usr(model_ext(x, training=training), training=training)  
    return loss_categorical_crossentropy(y, y_), y_
 
def loss_unlabel(model_ext, model_usr, x, training):    
    y_ = model_usr(model_ext(x, training=training), training=training) 
    return loss_categorical_crossentropy(y_, y_), y_ 

 


def loss_all_adv(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta, training=True): 
    y_l_d_hat = model_dom(model_ext(x_l, training=training), training=training) 
    y_ul_d_hat = model_dom(model_ext(x_ul, training=training), training=training) 
    real_loss  = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_l_d, y_l_d_hat, from_logits=False))
    fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_ul_d, y_ul_d_hat, from_logits=False))
    loss_d = beta*(real_loss + fake_loss)

    loss_l, _ = loss_label(model_ext, model_usr, x_l, y_l, training=True) 
    loss_l = tf.math.reduce_mean(loss_l)
    loss_ul, _ = loss_unlabel(model_ext, model_usr, x_ul, training=True)  
    loss_ul = tf.math.reduce_mean(loss_ul)
   
    loss_all = loss_l + alpha*loss_ul + loss_d

    return loss_all, loss_l, loss_ul, loss_d

def loss_all_dom(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta, training=True): 
    y_l_d_hat = model_dom(model_ext(x_l, training=training), training=training) 
    y_ul_d_hat = model_dom(model_ext(x_ul, training=training), training=training) 
    real_loss  = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_l_d, y_l_d_hat, from_logits=False))
    fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_ul_d, y_ul_d_hat, from_logits=False))

    loss_d = beta*(real_loss + fake_loss)

    return loss_d, loss_d


def loss_all_usr(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    '''
    Label dataset: x_l, y_l, y_l_d,
    Unlabeled dataset: x_ul, y_ul_d
    ''' 
    
    loss_l, _ = loss_label(model_ext, model_usr, x_l, y_l, training=True) 
    loss_l = tf.math.reduce_mean(loss_l)

    loss_ul, _ = loss_unlabel(model_ext, model_usr, x_ul, training=True)  
    loss_ul = tf.math.reduce_mean(loss_ul)
   
    loss_all = loss_l + alpha*loss_ul  
 
     
    return loss_all, loss_l, loss_ul 
 
 

def loss_label_base(model_ext, model_usr, x, y, training):   
    y_ = model_usr(model_ext(x, training=training), training=training)  
    # print(y_) 
    # print(model_usr(model_ext(x, training=training), training=training))
    return tf.keras.losses.categorical_crossentropy(y, y_) 
 
 
def loss_base(model_ext, model_usr, x_l, y_l):
    loss_l = loss_label_base(model_ext, model_usr, x_l, y_l, training=True)     
    return loss_l

def grad_base(model_ext, model_usr, x_l, y_l):
    with tf.GradientTape(persistent=True) as tape:
        loss_value = loss_base(model_ext, model_usr, x_l, y_l)  
    return loss_value, tape.gradient(loss_value, model_ext.trainable_variables), tape.gradient(loss_value, model_usr.trainable_variables)  

def grad_feature_extractor(model_ext, model_usr, x_l, y_l):
    with tf.GradientTape() as tape:
        loss_value = loss_base(model_ext, model_usr, x_l, y_l) 
    return loss_value, tape.gradient(loss_value, model_ext.trainable_variables) 

def grad_user_recognizer(model_ext, model_usr, x_l, y_l):
    with tf.GradientTape() as tape:
        loss_value = loss_base(model_ext, model_usr, x_l, y_l) 
    return loss_value, 



def loss_all_usr_base(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    '''
    Label dataset: x_l, y_l, y_l_d,
    Unlabeled dataset: x_ul, y_ul_d
    ''' 
    loss_l, _ = loss_label(model_ext, model_usr, x_l, y_l, training=True) 
    loss_l = tf.math.reduce_mean(loss_l)
   
    loss_all = loss_l  
     
    return loss_all, [loss_l, 0]



def grad_add(var_list, var_list2, weight):
    for idx in range(len(var_list)):
        var_list[idx] = var_list[idx] + weight*var_list2[idx]
    return var_list 

def grad_negative(var_list):
    for idx in range(len(var_list)):
        var_list[idx] = tf.math.negative(var_list[idx])
    return var_list 

def grad_feat_base(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    with tf.GradientTape() as tape:
        loss_value, _ = loss_all_usr_base(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
    return loss_value, tape.gradient(loss_value, model_ext.trainable_variables)  

def grad_ur_base(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    with tf.GradientTape() as tape:
        loss_value, _ = loss_all_usr_base(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
    return loss_value, tape.gradient(loss_value, model_usr.trainable_variables) 



 
def grad_feat_usr(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    with tf.GradientTape() as tape:
        loss_all, lossl_l, loss_ul, loss_d = loss_all_adv(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
    return loss_all, tape.gradient(loss_all, model_ext.trainable_variables)  

def grad_ur(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    with tf.GradientTape() as tape:
        loss_all, lossl_l, loss_ul, loss_d = loss_all_adv(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
    return loss_all, tape.gradient(loss_all, model_usr.trainable_variables), lossl_l, loss_ul
  
def grad_dom(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    with tf.GradientTape() as tape:
        loss_all, lossl_l, loss_ul, loss_d = loss_all_adv(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta)
    return loss_all, tape.gradient(loss_all, model_dom.trainable_variables), loss_d

def grad_all(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
    with tf.GradientTape(persistent=True) as tape:
        loss_all, loss_l, loss_ul, loss_d = loss_all_adv(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
    return loss_all, loss_l, loss_ul, loss_d, tape.gradient(loss_all, model_ext.trainable_variables), tape.gradient(loss_all, model_usr.trainable_variables), tape.gradient(loss_d, model_dom.trainable_variables) 


# def grad_feat_dom(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
#     with tf.GradientTape() as tape: 
#         loss_value = loss_all_dom_rev(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
#     return loss_value, tape.gradient(loss_value, model_ext.trainable_variables)  

# def grad_feat_usr(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
#     with tf.GradientTape() as tape:
#         loss_value, _ = loss_all_usr(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
#     return loss_value, tape.gradient(loss_value, model_ext.trainable_variables)  

# def grad_ur(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
#     with tf.GradientTape() as tape:
#         loss_value, loss_list = loss_all_usr(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
#     return loss_value, tape.gradient(loss_value, model_usr.trainable_variables), loss_list[0], loss_list[1] 
  
# def grad_dom(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
#     with tf.GradientTape() as tape:
#         loss_value, loss_d = loss_all_dom(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta)
#     return loss_value, tape.gradient(loss_value, model_dom.trainable_variables), loss_d
 
# def grad_dom(model_ext, model_usr, model_dom, model_flip, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta):
#     grad_list = []
#     for dom in range(len(model_dom)):
#         with tf.GradientTape() as tape:
#             loss_value, loss_d = loss_all_dom_multi(model_ext, model_usr, model_dom, x_l, y_l, y_l_d, x_ul, y_ul_d, n_class_dist, alpha, beta, eta) 
#         grad_list.append(tape.gradient(loss_value, model_dom[dom].trainable_variables))
#     return loss_value, grad_list, loss_d