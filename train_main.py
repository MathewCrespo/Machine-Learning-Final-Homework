# final version
import numpy as np
import pandas as pd
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

import densenet3D
from misc import set_gpu_usage,pack_up
from dataloader import get_test,get_train,get_validation,data_augmentaion
set_gpu_usage()

# start to train
# data augmentation
train_data, train_label = get_train() # get train data

new_data,new_label = data_augmentaion(train_data,train_label,140,140,138)

train_data, train_label = pack_up(train_data,train_label,new_data,new_label)

mask = 32
input_size = (mask, mask, mask, 1)
model = densenet3D.createDenseNet(2, input_size, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16,
                                dropout_rate=0.1,
                                weight_decay=1E-4, verbose=True)

model.compile(optimizer=Adam(lr=1.e-3),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

early_stopping = EarlyStopping(monitor='binary_accuracy', min_delta=0, mode='max', patience=10, verbose=1)

BATCH_SIZE = 32
NUM_EPOCHS = 100
model.fit(train_data, train_label, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

model.save("densebest2.h5")

train_score = model.evaluate(train_data, train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (train_score[0], train_score[1] * 100))
# validation
val_data, val_label = get_validation()
test_score = model.evaluate(val_data, val_label)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (test_score[0], test_score[1] * 100))

## test
test_num = 117
test_dir = pd.read_table("./data/sample.csv",sep = ",")['name']

test_name = np.array(test_dir).reshape(test_num)
test_data = get_test()
predict = model.predict(test_data)[:, 1]
predicted = np.array(predict).reshape(test_num)
test_dict = {'Id': test_name, 'Predicted': predicted}

result = pd.DataFrame(test_dict, index=[0 for _ in range(test_num)])

result.to_csv("result.csv", index=False, sep=',')