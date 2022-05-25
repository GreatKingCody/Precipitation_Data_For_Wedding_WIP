from Final import climate
from Final import precip_amount
from Final import train_data, train_labels, test_data, test_labels
import pandas as pd
import numpy as np


import time                                          
import datetime as dt

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


'''
#We need to convert our labels to someting RandomForestClassifier can understand.
# It does not understand continuious, so while I could convert it to an int, I dont
# think that is statistically the best way to do that. Sklearn as a preprocessing
# model for label encoding, which I used below to relabel the labels into something
# it understands and can read in on.
'''


# rf_labels = le.fit_transform(rf_labels)


RFC = RandomForestClassifier(verbose = 0, warm_start = False, max_features = None, 
                             max_depth = None, criterion = 'gini',
                             min_samples_split = 35, n_estimators = 80)

# Beautiful, so I got my best parameters from the grid search, you can see the 
# best parameters in the RFC call. Lets fit it now. 


# parameters = {'n_estimators': [50, 100, 150, 200], 
#               'max_depth': [None, 25, 50, 100], 
#               'max_features': [None, 4, 6, 8, 10],
#               'min_samples_split': [2, 4, 10, 25, 75, 125, 150, 200],
#               'criterion': ['gini']
#               }

# I JUST found out about halving grid search which gives as good or  better 
# results quicker, so we are DEFINITELY going to use this from now on.

# grid = HalvingGridSearchCV(RFC, parameters)
# results = grid.fit(train_data, train_labels)
# print(results.best_params_)
# print(results.best_estimator_)
# print(results.best_score_)


# print(climate.head())
RFC.fit(train_data, train_labels)
score = RFC.score(test_data, test_labels)
# print(score)


# Lets give it a shot to see it it will predict a past date correctly.
avg_max_temp_july = climate.loc[climate.month == 7]\
    .loc[climate.year == 1996].max_temp.mean()
avg_min_temp_july = climate.loc[climate.month == 7]\
    .loc[climate.year == 1996].min_temp.mean()
pred_x = np.array([1996, 7, 29, avg_max_temp_july, avg_min_temp_july]).reshape(1, -1)


# print(climate.loc[climate.month == 7].loc[climate.year == 1996]\
    # .loc[climate.day == 29])
# print(RFC.predict(pred_x))
# print(RFC.predict_proba(pred_x))


# Now lets try a future date, but lets build a function for it 
# (and also a fun printout).
def rf_predict(df, year, month, day):
    avg_max_temp = df.loc[df.month == month].loc[df.day == day].max_temp.mean()
    avg_min_temp = df.loc[df.month == month].loc[df.day == day].min_temp.mean()   
    pred_x = np.array([year, month, day, avg_max_temp, avg_min_temp]).reshape(1, -1)
    return pred_x
pred_x = rf_predict(climate, 2022, 5, 28)


date = [int(pred_x[0][1]), int(pred_x[0][2]), int(pred_x[0][0])]


prediction = RFC.predict(pred_x)
probability = RFC.predict_proba(pred_x)

# print(prediction)
# print(probability)
proba_of_rain = round(probability[0][1], 2) * 100
if prediction == 1:
    print(f'On {date[0]}-{date[1]}-{date[2]}:\nI predict it will rain!')
    print(f'There is a {proba_of_rain} percent chance of rain!')
else:
    print(f'On {date[0]}-{date[1]}-{date[2]}:\nI do not think it will rain!')
    print(f'There is a {proba_of_rain} percent chance of rain!')


'''
This model predict there was a 58 %\ chance it did rain and a 41%\ chance it did not
rain on July 29th, 1996 (as requested by my friend for the day he was born). 
Historical records show that it DID in fact rain on that day, therefore we can
say the the model predicted correctly that it rained. 


For the future prediction, the model is predicting that it has a 76 %\ chance
of not raining and a 24 %\ chance of raining tomorrow (at the time of writing),
and weather data for tomorrow shows that there is a 24%\ chance of rain. For the
day after tomorrow, it is showing there is a 80.2%\ chance of not raining and a 
19.8%\ chance of raining. The Weather Channel shows there is an 18%\ chance of
raining. That is about 2%\ off of what the Weather Channel states it will be.


So it looks like the RFC has a 75%\ likelihood that it is correct, based on the raw
score of the model. I want to try and also fit a Random Forest Regressor, 
so I am going to do that below with the
same steps as above. I will have to change the labels because the regressor
runs on a quantative basis, so it needs all numbers instead of a bool.
'''




climate['raw_precip'] = precip_amount
# print(climate.head())
# print(climate.raw_precip.unique())
climate.raw_precip.fillna(climate.raw_precip.mean())

features = climate[['year', 'month', 'day', 'max_temp', 'min_temp']].to_numpy()
labels = climate['raw_precip'].to_numpy().reshape(-1, 1)

minmax = MinMaxScaler()
labels = minmax.fit_transform(labels)

train_data, test_data,\
train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)


RFR = RandomForestRegressor(verbose = 1, warm_start = True, 
                            criterion = 'absolute_error', random_state = 11,
                            bootstrap = True)


# RFR.fit(train_data, train_labels)
# score = RFR.score(test_data, test_labels)
# print(score)


# parameters = {'max_depth': [None],
#               'min_samples_split': [None],
#               'max_features': [None, 1],
#               'n_estimators': [100]}

# grid = HalvingGridSearchCV(RFR, parameters)
# results = grid.fit(train_data, train_labels)
# score = grid.score(test_data, test_labels)
# print(score)
# print(results.best_params_)
# print(results.best_score_)

