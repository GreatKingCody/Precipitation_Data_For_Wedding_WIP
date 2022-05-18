from Final import climate
from Final import train_data, test_data, train_labels, test_labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier



# Alright, so we are going to start a grid search first, and then we will print our
# test score. We are going to do a GBC because it is a very accurate ML model.


# First thing is to create the parameters for the grid search.
# parameters = {'n_estimators': [2, 10, 25, 50, 75, 100, 150, 200, 210],
#               'learning_rate': [.0001, .001, .01, .1, .15, 0.19, .2, .21, .25, .3],
#               'min_samples_split': [2, 5, 10, 25, 50, 75, 95, 100, 105, 125, 150],
#               'max_depth': [3, 4, 5, 10, 15, 25, 50, 75, 100, 150, 200],
#               'max_features': [1, 2, 3, 4, 5],
#               }

# After numerous gridsearch testing, below I have fit the GBC with the best parameters.
# The run took about 8 hours to fit, so I left it all day!


gbc = GradientBoostingClassifier(loss = 'exponential', random_state = 11, 
                                 verbose = 0, validation_fraction = 0.2, tol = 1e-8, 
                                 n_iter_no_change = 1000, n_estimators = 200,
                                 learning_rate = .19, min_samples_split = 95,
                                 max_depth = 3, max_features = 4)


# grid = GridSearchCV(gbc, parameters)
# results = grid.fit(train_data, train_labels)
# print(results.best_params_)
# print(results.best_estimator_)
# print(results.best_score_)
# The lines above print a few different statistics about the ML model.
# The most important one is the .best_params_ in my opinion, it shows what had the
# best scores. 


# Time to officially fit!
gbc.fit(train_data, train_labels)
gbc_test_score = gbc.score(test_data, test_labels)


# Alright, to predict I will need the min and max temps for the month, I could do it
# by day, but unfortunatey that is a ton of work. 30 * each month, so I am going to 
# pass on that one for now. 
april_max_temp = climate.loc[climate.month == 4].max_temp.mean()
may_max_temp = climate.loc[climate.month == 5].max_temp.mean()
june_max_temp = climate.loc[climate.month == 6].max_temp.mean()
sep_max_temp = climate.loc[climate.month == 9].max_temp.mean()
oct_max_temp = climate.loc[climate.month == 10].max_temp.mean()
april_min_temp = climate.loc[climate.month == 4].min_temp.mean()
may_min_temp = climate.loc[climate.month == 5].min_temp.mean()
june_min_temp = climate.loc[climate.month == 6].min_temp.mean()
sep_min_temp = climate.loc[climate.month == 9].min_temp.mean()
oct_min_temp = climate.loc[climate.month == 10].min_temp.mean()



# Lets just print the score to check one more time
print(gbc_test_score)


# year, month, day, max_temp, min_temp is the order our numpy array prediction
#should be in 
pred_x = np.array([2023, 4, 15, april_max_temp, april_min_temp]).reshape(1, -1)


#Okay, lets see what the probability of rain is in april!
april_prediction = gbc.predict(pred_x)
april_prediction_prob = gbc.predict_proba(pred_x)
print(april_prediction)
print(april_prediction_prob)
# We can say in april we 75 percent sure that it will not rain! 
# There is a 25% chance it will rain. 