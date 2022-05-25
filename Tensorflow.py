from Final import climate_write

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import History
from sklearn.preprocessing import LabelEncoder


# So the first thing I need to do is train_test_split these again. I want to see
# if I can use some of the extra columns in a tensorflow model that I dropped from
# the GBC and RFC classifiers. 

# print(climate.columns)
# print(climate_write.head())



# le = LabelEncoder()
# encoded = le.fit_transform()
features = climate_write[['year', 'month', 'day', 'max_temp',
                    'min_temp']].to_numpy()

labels = climate_write.total_precipitation.to_numpy()

features_shape = features.shape


minmax = MinMaxScaler()

train_data, test_data, train_labels, test_labels = \
    train_test_split(features, labels, test_size = 0.2)
    
   
# train_labels = minmax.fit_transform(train_labels)
# test_labels = minmax.transform(test_labels)    git p
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_labels = train_labels.astype('float32')
test_labels = test_labels.astype('float32')
# train_data = tf.convert_to_tensor(train_data)
print(train_labels)
opt = Adam(learning_rate = 0.001)

# print(Counter(climate_write.total_precipitation))
inp = keras.Input(shape = (features_shape))
model = Sequential()
model.add(Dense(8, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.compile(optimizer = opt, loss = 'mse', metrics = ['accuracy'])

model.fit(train_data, train_labels, verbose = 1, epochs = 12, batch_size = 100)

print(model.predict(test_data))