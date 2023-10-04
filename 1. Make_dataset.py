import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def Make_Dataset_df(path, start_year, end_year, start_lon, end_lon, lon_bin, start_lat, end_lat, lat_bin):
    dif_year = (end_year - start_year) + 1
    total_year = np.linspace(start_year, end_year, dif_year, dtype=int)

    real_df = pd.DataFrame()

    path = path

    for y in range(len(total_year)):

        data_path = path + str(total_year[y]) + '/'
        file_list = os.listdir(data_path)
        file_list.sort()

        Error_msg = data_path + '.DS_Store'

        if os.path.exists(Error_msg):
            os.remove(Error_msg)

            data_path = path + str(total_year[y]) + '/'
            file_list = os.listdir(data_path)
            file_list.sort()

        else:
            print("Can not delete the file as it doesn't exists")

        for O_f in range(len(file_list)):

            DCGAN_file = data_path + file_list[O_f] + '/completed_pb_lr/1950.txt'

            if os.path.exists(DCGAN_file):

                # DCGAN-PB result
                with open(DCGAN_file, "r") as file:
                    DCGAN_pb_value = np.array([float(i) for line in file for i in line.split('/n') if i.strip()])

                # DCGAN_PB_result = np.reshape(DCGAN_pb_list, (32, 32))
                obs_year = int(file_list[O_f][16:20])
                obs_month = int(file_list[O_f][21:23])
                obs_day = int(file_list[O_f][24:26])
                obs_hour = int(file_list[O_f][27:29])

                # spec_time = datetime(obs_year, obs_month, obs_day, obs_hour, 0, 0)

                nan_value = np.where((DCGAN_pb_value == 9999.0) | (DCGAN_pb_value < 0.5) | (DCGAN_pb_value > 100))

                if len(nan_value[0]) == 0:
                    DCGAN_pb_value = DCGAN_pb_value
                else:
                    DCGAN_pb_value[nan_value] = np.nan

                date_data = {'Year': int(obs_year),
                             'Month': int(obs_month),
                             'Day': int(obs_day),
                             'Hour': int(obs_hour)}

                df_1_date = pd.DataFrame(date_data, index=[0])
                df_1_TEC = (pd.DataFrame(DCGAN_pb_value)).transpose()
                df_comb = pd.concat([df_1_date, df_1_TEC], axis=1)

                real_df = pd.concat([real_df, df_comb], axis=0)

            else:

                print('There is no data')

    return real_df


################### Setting ###################

start_year = 2010
end_year = 2010
start_lat = 25.5
end_lat = 41
start_lon = 120
end_lon = 135.5
lat_bin = 0.5
lon_bin = 0.5

path = '/Users/jeongseheon/Desktop/JSH/[2] Data/DCGAN_result/'
saving_path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC_convLSTM/Dataset/'
# createFolder(saving_path+'initial_dataset_2d')
title_name = 'DCGAN_PB_TEC_' + str(start_year) + '_' + str(end_year) + ''

if os.path.exists(saving_path + 'initial_dataset_2d/' + title_name + '_Dataset_'+str(start_year)+'_'
                  +str(end_year)+'.csv'):

    df_values = pd.read_csv(saving_path + 'initial_dataset_2d/' + title_name + '_Dataset_'+str(start_year)+'_'
                  +str(end_year)+'.csv')
    date = df_values.values[:, 1:5]
    values = df_values.values[:, 5:]

else:

    df_data = Make_Dataset_df(path, start_year, end_year, start_lon, end_lon, lon_bin, start_lat, end_lat, lat_bin)
    df_data_intp = df_data.interpolate()
    date = df_data_intp.values[:, 0:4]
    values = df_data_intp.values[:, 4:]

    # interpolation을 했음에도 불구하고 첫 값이 NAN인 경우 interpolation을 하지 못함 따라서 0으로 변경
    nan_value = np.where(np.isnan(values) == True)
    values[nan_value] = 0

    df_date = pd.DataFrame(date)
    df_values = pd.DataFrame(values)

    save_df = pd.concat([df_date, df_values], axis=1)

    save_df.to_csv(saving_path + 'initial_dataset_2d/' + title_name + '_Dataset_'+str(start_year)+'_'
                  +str(end_year)+'.csv')


def generate_dataset(data, date, n_samples, past_history, future_target):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    data_4d = scaled_data.reshape(n_samples, 32, 32, 1)

    date_data_n_frames = []
    for i in range(n_samples - past_history - future_target - future_target):  # 영화 샘플수만큼 반복

        for t in range(past_history):

            hist_data = data_4d[i + t, :, :, :]  # 32 * 32 * 1
            hist_data_n_frames_1 = hist_data.reshape(1, 32, 32, 1)
            date_data_n_frames_1 = date[i + (t + future_target), :]

            if t > 0:
                hist_data_n_frames = np.concatenate([hist_data_n_frames, hist_data_n_frames_1])
                date_data_n_frames = np.concatenate([date_data_n_frames, date_data_n_frames_1])
            else:
                hist_data_n_frames = hist_data_n_frames_1
                date_data_n_frames = date_data_n_frames_1

        for f in range(future_target):

            next_data = data_4d[i + (f + future_target), :, :, :]  # 32 * 32 * 1
            next_data_n_frames_1 = next_data.reshape(1, 32, 32, 1)

            if f > 0:
                next_data_n_frames = np.concatenate([next_data_n_frames, next_data_n_frames_1])
            else:
                next_data_n_frames = next_data_n_frames_1

        hist_data_5d_1 = hist_data_n_frames.reshape(1, -1, 32, 32, 1)
        next_data_5d_1 = next_data_n_frames.reshape(1, -1, 32, 32, 1)
        date_data_1 = date_data_n_frames.reshape(1, -1)

        if i > 0:
            hist_data_5d = np.concatenate([hist_data_5d, hist_data_5d_1])
            next_data_5d = np.concatenate([next_data_5d, next_data_5d_1])
            date_data_5d = np.concatenate([date_data_5d, date_data_1])
        else:
            hist_data_5d = hist_data_5d_1
            next_data_5d = next_data_5d_1
            date_data_5d = date_data_1

        print('Data shape:' + str(hist_data_5d.shape))
        print('Future shape:' + str(next_data_5d.shape))

    return hist_data_5d, next_data_5d, date_data_5d, scaler


past_history = 24
future_target = 24

hist_data_5d, next_data_5d, date_data_5d, scaler = generate_dataset(values, date,\
                                                                    values.shape[0], past_history, future_target)

createFolder(saving_path + 'model_input_dataset_5d')
with open(saving_path + 'model_input_dataset_5d/' + title_name + '_past_' + str(past_history) + '_Dataset_'
          +str(start_year)+'_'+str(end_year)+'.pickle', 'wb') as t1:
    pickle.dump([hist_data_5d, next_data_5d, date_data_5d, scaler], t1)
