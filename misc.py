import keras.backend as K
import tensorflow as tf
import numpy as np


def get_gpu_session(ratio=None, interactive=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)
    return sess


def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)




def pack_up(train_data,train_label,extend_data,extend_label):
    len1=train_data.shape[0]
    len2=extend_data.shape[0]
    len= len1+len2
    new_data=np.zeros((len,32,32,32,1))
    new_label=np.zeros((len,1))
    for i in range(len):
        if i<len1:
            new_data[i,:,:,:,0]=train_data[i,:,:,:,0]
            new_label[i,0]=train_label[i,0]
        else:
            new_data[i,:,:,:,0]=extend_data[i-len1,:,:,:,0]
            new_label[i,0]= extend_label[i-len1,0]

    return  new_data,new_label

