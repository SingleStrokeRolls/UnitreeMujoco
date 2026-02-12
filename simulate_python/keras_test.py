import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
WINDOW_SIZE = 10

test_df = pd.read_csv('./csv/1237.csv')
test_df = test_df.iloc[:210,:]
test_df['acc_mag'] = np.sqrt(test_df['acc_x']**2 + test_df['acc_y']**2 + test_df['acc_z']**2)
test_df['gyro_mag'] = np.sqrt(test_df['gyro_x']**2 + test_df['gyro_y']**2 + test_df['gyro_z']**2)
predictions = []
input_x = []
model = keras.models.load_model('0210.keras')
# for i in range(200):
#     data = test_df.iloc[i:i+WINDOW_SIZE, -3:]
#     input_x.append(data)
# input_x = np.array(input_x)    
# probs = model.predict(input_x)
# pred  = probs.argmax(axis=1)
for i in range(10):
    data = test_df.iloc[i:i+WINDOW_SIZE, -3:]
    data = np.array(data)
    data = np.expand_dims(data, axis=0) 
    start = time.time()
    probs = model.predict(data)
    end = time.time()
    pred  = probs.argmax(axis=1)
    print(pred) 
    print(f"Prediction Duration: {end - start}")