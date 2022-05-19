from Final import climate
from Final import tf_precipitation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt


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
# print(climate.head())


# So it looks like my lambda apply didnt save, so lets run that again really quick.
climate[['fog', 'thunder', 'sleet', 'hail', 'rime', 
         'high_winds', 'drizzle', 'rain']] = \
             climate[['fog', 'thunder', 'sleet', 'hail', 
                      'rime','high_winds', 'drizzle', 'rain']]\
                          .apply([lambda x: 1 if x == True else 0])
# print(climate.head())
# Okay that is fixed. Need to make the features and labels now. 



# le = LabelEncoder()
# encoded = le.fit_transform()
features = climate[['year', 'month', 'day', 'max_temp',
                    'min_temp', 'fog', 'thunder', 'rime',
                    'high_winds']].to_numpy()

labels = tf_precipitation.to_numpy()

features_shape = features.shape

train_data, test_data, train_labels, test_labels = \
    train_test_split(features, labels, test_size = 0.2)
    
minmax = MinMaxScaler()
minmax.fit_transform(train_data, train_labels)
inp = keras.Input(shape = (features_shape))
model = Sequential()
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'sigmoid'))

