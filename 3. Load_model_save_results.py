import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Dropout, Conv3D, Activation
import pickle

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

######################################################################
############################ Read dataset ############################
######################################################################
# Hyperparmeter
train_start_yr = 2002
train_end_yr = 2018
test_start_yr = 2019
test_end_yr = 2019
batch_size = 256
past_history = 24

path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC_convLSTM/Dataset/'

# convLSTM input dataset (pickle)
model_input_path = path +'model_input_dataset_5d/'

input_data_list = os.listdir(model_input_path)
input_data = [j for j in input_data_list if j.endswith('.pickle')]
input_data.sort()

for j in range(len(input_data)):
    with open(model_input_path + input_data[-1], 'rb') as t1:
        hist_data_5d, next_data_5d, date_data_5d, scaler = pickle.load(t1)

    test_hist_data_5d = hist_data_5d
    test_next_data_5d = next_data_5d
    test_date_data_5d = date_data_5d
    test_scaler = scaler

######################################################################
####################### Run the convLSTM model #######################
######################################################################

# Define the model
model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(None, 32, 32, 1),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"))

checkpoint_path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC_convLSTM/' \
                  'Model_checkpoint_'+str(train_start_yr)+'_'+str(train_end_yr)+'past_'+str(past_history)+\
                  '_batch_'+str(batch_size)+'/'
checkpoint_list = os.listdir(checkpoint_path)
checkpoint = [j for j in checkpoint_list if j.endswith('.hdf5')]
checkpoint.sort()
model.load_weights(checkpoint_path+checkpoint[-1])

test_next_data_5d_pred = model.predict(test_hist_data_5d)

saving_path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC_convLSTM/Results/' \
              +str(test_start_yr)+'_'+str(test_end_yr)+'past_'+str(past_history)+'_batch_'+str(batch_size)+'/'
createFolder(saving_path)
with open(saving_path+'model_results_Dataset.pickle', 'wb') as t1:
    pickle.dump([test_date_data_5d, test_next_data_5d, test_next_data_5d_pred, test_scaler], t1)

print('Done!!')



