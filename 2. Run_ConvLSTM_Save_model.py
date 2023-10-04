import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from keras import layers
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
# DCGAN-PB dataset (csv)
start_year = 2002
end_year = 2002

path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC_convLSTM/Dataset/'
values_path = path +'initial_dataset_2d/'
values_list = os.listdir(values_path)

Error_msg = values_path + '.DS_Store'

if os.path.exists(Error_msg):
    os.remove(Error_msg)
    values_list = os.listdir(values_path)

# values_csv = [j for j in values_list if (j.find(str(start_year)))]
values_csv = [j for j in values_list if j.endswith('.csv')]

initial_values = []
for i in range(len(values_csv)):
    df_values = pd.read_csv(values_path + values_csv[i])
    values = df_values.values[:, 5:]

    all_values = np.append(initial_values, values)

# for i in range(len(values_csv)):
#
#     data_st_yr = int(values_csv[i].split('_')[3])
#     data_end_yr = int(values_csv[i].split('_')[4])
#
#     if (start_year == data_st_yr) & (end_year == data_end_yr):
#         real_csv = np.append(real_csv, values_csv[i])

# df_values = pd.read_csv(values_path + real_csv[0])
# values = df_values.values[:, 5:]

# convLSTM input dataset (pickle)
model_input_path = path +'model_input_dataset_5d/'

input_data_list = os.listdir(model_input_path)
input_data = [j for j in input_data_list if j.endswith('.pickle')]

with open(model_input_path+input_data[1], 'rb') as t1:
    hist_data_5d, next_data_5d, date_data_5d, scaler = pickle.load(t1)

######################################################################
# ####################### Run the convLSTM model #######################
# ######################################################################
# # Hyperparmeter
# train_split = 6000
# batch_size = 256
# epochs = 100
#
# model = keras.Sequential(
#     [keras.Input(
#             shape=(None, 32, 32, 1)),
#         layers.ConvLSTM2D(
#             filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
#         ),
#         layers.BatchNormalization(),
#         layers.ConvLSTM2D(
#             filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
#         ),
#         layers.BatchNormalization(),
#         layers.ConvLSTM2D(
#             filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
#         ),
#         layers.BatchNormalization(),
#         layers.ConvLSTM2D(
#             filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
#         ),
#         layers.BatchNormalization(),
#         layers.Conv3D(
#             filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
#         ),
#     ]
# )
#
# model.compile(loss="binary_crossentropy", optimizer="adam")
# createFolder('/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC/ConvLSTM/Model_checkpoint')
# checkpoint_path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC/ConvLSTM/Model_checkpoint/'
# checkpoint = ModelCheckpoint(filepath=checkpoint_path+'{epoch:02d}-{val_loss:.4f}.hdf5', \
#                              monitor='val_loss', save_best_only=True, verbose=1)
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto')
# model_history = model.fit(hist_data_5d[:train_split], next_data_5d[:train_split], batch_size=batch_size, \
#                           epochs=epochs, verbose=2, validation_split=0.2, shuffle=False, \
#                           callbacks=[early_stopping, checkpoint])
# test_loss, test_acc = model.evaluate(hist_data_5d[:train_split], next_data_5d[:train_split], verbose=2)