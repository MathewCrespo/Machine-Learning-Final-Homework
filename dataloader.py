import numpy as np
import pandas as pd
import keras
NUM_CLASSES = 2
MASK_SIZE = 32

train_val_num = 465
test_num = 117

train_dir = './data/train_val/'
test_dir = './data/test/'

train_list = pd.read_table("./data/train_val.csv", sep=",")['name']
label_list = pd.read_table("./data/train_val.csv", sep=",")['diagnosis']
test_list = pd.read_table("./data/sample.csv",sep = ",")['name']

train_num = int(0.9*label_list.shape[0])
val_num = label_list.shape[0]-train_num


def get_train():
    train_num = int(0.9 * label_list.shape[0])
    train_data = np.ones((train_num, 32, 32, 32, 1))
    train_label = []
    for i in range(train_num):
        candidate = train_dir + train_list[i] + '.npz'
        data = np.load(candidate)
        train_data[i, :, :, :, 0] = \
            data['voxel'][int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2)] * \
            data['seg'][int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2)]
        train_label = np.append(train_label, label_list[i])
    train_data = train_data.reshape(train_data.shape[0], MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)
    train_data = train_data.astype('float32') / 255.
    train_label = keras.utils.to_categorical(train_label, NUM_CLASSES)
    return train_data,train_label

def get_validation():
    val_data = np.ones((val_num, 32, 32, 32, 1))
    val_label = []
    for i in range(train_num, label_list.shape[0]):
        candidate = train_dir + train_list[i] + '.npz'
        data = np.load(candidate)
        val_data[i-train_num, :, :, :, 0] = \
            data['voxel'][int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2)] * \
            data['seg'][int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2)]
        val_label = np.append(val_label, label_list[i])
    val_data = val_data.reshape(val_data.shape[0], MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)
    val_data = val_data.astype('float32') / 255.
    val_label = keras.utils.to_categorical(val_label, NUM_CLASSES)
    return  val_data,val_label

def get_test():
    test_data = np.ones((test_num, 32, 32, 32, 1))
    for i in range(0, test_num):
        candidate = test_dir + test_list[i] + '.npz'
        data = np.load(candidate)
        test_data[i, :, :, :, 0] = \
            data['voxel'][int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2)] * \
            data['seg'][int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2), \
            int((100 - MASK_SIZE) / 2):int((100 + MASK_SIZE) / 2)]
    test_data = test_data.reshape(test_data.shape[0], MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)
    test_data = test_data.astype('float32') / 255.
    return test_data

def data_augmentaion(data,label,updown_num,lr_num,frontback_num):
    new_data = np.zeros(np.shape(data))
    new_label = label
    for i in range(updown_num):
        for j in range(0, 32):
            for k in range(0,32):
                new_data[i, 32-k, :, j, 0] = data[i, k, :, 31-j, 0]
    for i in range (updown_num,updown_num+lr_num):
        for j in range(0, 32):
            for k in range(0,32):
                new_data[i, j, 31-k, :, 0] = data[i, 31 - j, k, :, 0]
    for i in range(updown_num+lr_num,updown_num+lr_num+frontback_num):
        for j in range(0, 32):
            for k in range(0,32):
                new_data[i, :, j, 31-k, 0] = data[i, :, 31 - j, k, 0]

    return new_data,new_label

