from Final import climate
from Final import train_data, test_data, train_labels, test_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


#We need to convert our labels to someting RandomForestClassifier can understand.
# It does not understand continuious, so while I could convert it to an int, I dont
# think that is statistically the best way to do that. Sklearn as a preprocessing
# model for label encoding, which I used below to relabel the labels into something
# it understands and can read in on.
# rf_labels = le.fit_transform(rf_labels)


RFC = RandomForestClassifier(verbose = 1, warm_start = False, max_features = None, 
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



# RFC.fit(train_data, train_labels)
# score = RFC.score(test_data, test_labels)
# print(score)
#So it looks like the RFC has a 75% likelihood that it is correct. I want to try
# and also fit a Random Forest Regressor, so I am going to do that below with the
# same steps as above


RFR = RandomForestRegressor(verbose = 1, warm_start = True, 
                            criterion = 'absolute_error', random_state = 11)

parameters = {'max_depth': [None, 1, 10, 50, 100, 150, 200, 300, 500, 1000],
              'min_samples_split': [2, 10, 50, 100, 150, 200, 300, 500, 1000],
              'max_features': [None, 1, 'sqrt', 'log2'],
              'n_estimators': [1, 10, 50, 100, 150, 200, 300, 500, 1000]}

grid = HalvingGridSearchCV(RFR, parameters)
results = grid.fit(train_data, train_labels)
score = grid.score(test_data, test_labels)
print(score)
print(results.best_params_)
print(results.best_score_)