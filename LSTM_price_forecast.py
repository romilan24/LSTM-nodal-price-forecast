import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from torch.utils.data import DataLoader, TensorDataset
#from torch.cuda import is_available
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from preprocess import rename_columns, swap_missing_data, interpolate_missing, split_data, date_and_hour, calculate_mape, add_holiday_variable, create_lagged_variables

# Import necessary libraries
import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.metrics import mean_absolute_error
from keras import Sequential
from keras import layers
from keras.models import Model
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Flatten, Conv1D
from keras.layers import MaxPooling1D, GRU, Input,Masking, Concatenate, dot
from keras.optimizers import Adam, SGD
from keras.losses import MeanAbsoluteError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler

#rename column headers
column_mapping = {
    'Datetime': 'datetime',
    'Current demand': 'caiso_load_actuals',
    'KCASANFR698_Temperature': 'SF_temp',
    'KCASANFR698_Dew_Point':'SF_dew',
    'KCASANFR698_Humidity':'SF_humidity',
    'KCASANFR698_Speed':'SF_windspeed',
    'KCASANFR698_Gust':'SF_windgust',
    'KCASANFR698_Pressure':'SF_pressure',
    'KCASANJO17_Temperature': 'SJ_temp',
    'KCASANJO17_Dew_Point':'SJ_dew',
    'KCASANJO17_Humidity':'SJ_humidity',
    'KCASANJO17_Speed':'SJ_windspeed',
    'KCASANJO17_Gust':'SJ_windgust',
    'KCASANJO17_Pressure':'SJ_pressure',
    'KCABAKER271_Temperature': 'BAKE_temp',
    'KCABAKER271_Humidity':'BAKE_humidity',
    'KCABAKER271_Speed':'BAKE_windspeed',
    'KCABAKER271_Pressure':'BAKE_pressure',
    'KCAELSEG23_Temperature': 'EL_temp',
    'KCAELSEG23_Dew_Point':'EL_dew',
    'KCAELSEG23_Humidity':'EL_humidity',
    'KCAELSEG23_Speed':'EL_windspeed',
    'KCAELSEG23_Gust':'EL_windgust',
    'KCAELSEG23_Pressure':'EL_pressure',
    'KCARIVER117_Temperature': 'RIV_temp',
    'KCARIVER117_Dew_Point':'RIV_dew',
    'KCARIVER117_Humidity':'RIV_humidity',
    'KCARIVER117_Speed':'RIV_windspeed',
    'KCARIVER117_Gust':'RIV_windgust',
    'KCARIVER117_Pressure':'RIV_pressure'
}

# Columns for swap missing NaN data between SF and SJ
sf_columns = [
    'KCASANFR698_Temperature', 'KCASANFR698_Dew_Point', 'KCASANFR698_Humidity',
    'KCASANFR698_Speed', 'KCASANFR698_Gust', 'KCASANFR698_Pressure'
]

sj_columns = [
    'KCASANJO17_Temperature', 'KCASANJO17_Dew_Point', 'KCASANJO17_Humidity',
    'KCASANJO17_Speed', 'KCASANJO17_Gust', 'KCASANJO17_Pressure'
]


# Specify the date ranges
train_start_date = pd.to_datetime('2021-01-02')
train_end_date = pd.to_datetime('2023-10-03')
predict_start_date = train_end_date + pd.Timedelta(days=1)
predict_end_date = predict_start_date + pd.Timedelta(hours=24)

# Assuming you want date objects
train_start_date_date = train_start_date.date()
train_end_date_date = train_end_date.date()

# Load and preprocess data
path = 'C:/Users/groutgauss/Machine_Learning_Projects/CAISO Price Forecast/Machine Learning/'
merged_df = pd.read_csv(path + 'data.csv')
merged_df = swap_missing_data(merged_df, sf_columns, sj_columns) #Swap SF and SJ weather data for NaN values
merged_df = interpolate_missing(merged_df) # Interpolate missing values
data = rename_columns(merged_df, column_mapping) #renames the column headers
#date_and_hour(data)
data = add_holiday_variable(data, 'datetime', train_start_date, train_end_date) # Add holiday variable
#data = create_lagged_variables(data, 'TH_SP15_GEN-APND', lag_range=7) # Create lagged variables

#filter train/test data for start/end dates
df = data[(data['datetime'] >= train_start_date) & (data['datetime'] <= train_end_date + pd.Timedelta(days=1))]

# Split data into features and target
X = df.drop(['datetime','TH_SP15_GEN-APND'], axis=1).values
y = df['TH_SP15_GEN-APND'].values

#Transform prep
def apply_PCA(X_input, cum_variance, if_apply):
    
    if if_apply:
    
        pca = PCA(n_components = cum_variance)
        # make pipeline to first standardize then apply PCA on data
        scaler_pca = make_pipeline(MinMaxScaler(), pca)
        X_pca = scaler_pca.fit(X_input).transform(X_input)

        return X_pca
    
    else:
        
        return np.array(X_input)

params_pca = {'cum_variance' : 0.8, 'if_apply' : True }
X_pca = apply_PCA(X, **params_pca)
X_pca.shape


def windowing(X_input,y_input, history_size):
    
    data = []
    labels = []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size : i, :])
        labels.append(y_input[i])
        
    return np.array(data), np.array(labels).reshape(-1,1)


train_cutoff = int(0.8*X_pca.shape[0])
val_cutoff   = int(0.9*X_pca.shape[0])

scaler_y = MinMaxScaler()
scaler_y.fit(y[:train_cutoff].reshape(-1,1))
y_norm = scaler_y.transform(y.reshape(-1,1))

hist_size= 24
data_norm = np.concatenate((X_pca,y_norm), axis = 1)

X_train, y_train = windowing(data_norm[:train_cutoff,:],data_norm[:train_cutoff,-1], hist_size)
X_val, y_val     = windowing(data_norm[train_cutoff :val_cutoff,:],data_norm[train_cutoff:val_cutoff,-1], hist_size)
X_test, y_test   = windowing(data_norm[val_cutoff :,:],data_norm[val_cutoff:,-1], hist_size)

#Training
def base_model_lstm():

    model = Sequential()
    model.add(LSTM(units = 32, return_sequences = True, activation="relu", input_shape = X_train.shape[-2:]))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    return model


epoch = 125
batch_size = 64
steps_per_epoch = len(X_train) // batch_size
cyclic_lr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-04,
                                                maximal_learning_rate=1e-02,
                                                scale_fn=lambda x: 1/(2**(x-1)),
                                                step_size=6 * steps_per_epoch)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
optimizer = Adam(learning_rate=cyclic_lr, amsgrad=True)

lstm_model = base_model_lstm()
lstm_model.compile(optimizer = optimizer, loss = 'mean_absolute_error')
lstm_model.summary()

history = lstm_model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs =epoch, 
                   batch_size = batch_size, callbacks=[callback])


##Prediction##
y_pred = lstm_model.predict(X_test)
y_pred_actual = scaler_y.inverse_transform(y_pred.reshape(-1,1))
y_test_inv = scaler_y.inverse_transform(y_test)


def plot_results(y_pred_actual, y_test_inv, history, model_name):
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))

    # Prediction vs actual price chart
    ax[0].plot(y_pred_actual[:1000])
    ax[0].plot(y_test_inv[:1000])
    ax[0].legend(['prediction', 'actual'], loc='upper left')
    ax[0].set_title(f'Prediction vs actual price for 1000 observation in test set ({model_name})')
    ax[0].set_xlabel('Observation')
    ax[0].set_ylabel('Price')

    # MAE chart
    ax[1].plot(history.history['loss'], label='Training Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].legend()
    ax[1].set_title(f'Training and validation MAE ({model_name})')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MAE')

    fig.tight_layout()
    plt.show()

print('')
print('')
print('---------------------------------------------------')
print(f'LSTM MAE for test set : {round(mean_absolute_error(y_pred,y_test),3)}')
print('---------------------------------------------------')
y_pred_actual = scaler_y.inverse_transform(y_pred)
print('')
plot_results(y_pred_actual, y_test_inv, history,'LSTM')