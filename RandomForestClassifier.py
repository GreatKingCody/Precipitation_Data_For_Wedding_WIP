from Final.py import climate
from Final.py import train_data, test_data, train_labels, test_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt

from sklearn.ensemble import RandomForestClassifier


#We need to convert our labels to someting RandomForestClassifier can understand.
# It does not understand continuious, so while I could convert it to an int, I dont
# think that is statistically the best way to do that. Sklearn as a preprocessing
# model for label encoding, which I used below to relabel the labels into something
# it understands and can read in on.
# rf_labels = le.fit_transform(rf_labels)





parameters = {'n_estimators': [1, 25, 75, 125, 200], 
              'max_depth': [None, 25, 50, 100], 
              'max_features': [None, 4, 6, 8, 10]}

# grid = GridSearchCV(RFC, parameters)
# results = grid.fit(train_data, train_labels)
# print(results.best_params_)
# I need to let this run, but it will take upwards of 8+ hours. Will let it run
# overnight.



# RFC = RandomForestClassifier(n_estimators = 188, random_state = 11, 
#                                  verbose = 1, warm_start = True)

# RFC.fit(train_data, train_labels)
# score = RFC.score(test_data, test_labels)


# print(score)
# print(max(score))