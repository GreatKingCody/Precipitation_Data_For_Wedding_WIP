import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt

# from scipy.stats import chi2_contingency
# import matplotlib.ticker as mtick


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# import tensorflow as tf
# from tensorflow	import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import Accuracy
# from tensorflow.keras.callbacks import History
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.metrics import r2_score





df = pd.read_csv('Hendersonville_Data.csv', low_memory = False)

# print(df.head())

df.DATE = pd.to_datetime(df.DATE)
# print(df.DATE.head())
# I'm going to need to seperate this by the following columns: year, month, day
# but I will address that later because I need a little bit more research into 
# how to seperate without converting each of these to strings and filtering them.
# I think I can use pd.query() plus a 'dt' function, and that will prevent a ton of
# extra work on my part.

#Let me fix fix the dtypes real quick
# df = df.convert_dtypes()
# print(df.columns)
# I cant auto convert the values because it causes issues with the ML model later on.


# All upper case if rough, I feel like I am being yelled at... (looking at you SQL).
# Let's fix that.
df.columns = df.columns.str.lower()
# print(df.columns)
# Much better. Easier to call too. Let me change some column names to give them
# a better description of what they actually do. That means its time for the readme.


# Okay so selenium is awful, I basically made a bot that clicked through the pages
# to get to the readme, but it took WAY too long to execute, so I just downloaded
# the text file like I should have in the first place.



#I need to seperate the date into seperate columns before seperating the df.
df['year'], df['month'], df['day'] = df['date'].dt.year, \
                                     df.date.dt.month, \
                                     df.date.dt.day
#Okay that was actually a lot easier than I thought it would be. Thank you dt.


#Lets clean it up a little now that we have the readme
climate = df[['year', 'month', 'day', 'tmax', 'tmin', 'prcp', 'snow', 'snwd', 'dapr', 
              'dasf', 'mdpr','mdsf']]
pd.set_option('mode.chained_assignment', None)
#I started to create two different dataframes, but realized most of the data in
# the dataframe was actually pretty useless. Also the set_options keeps the
# Pandas from putting a bunch of shit in the output about taking a slice from a 
# different dataframe.


# print(temp.columns)
# Basic renaming so I dont have to go back to the readme.
climate.rename(columns = {'tmax': 'max_temp', 
                          'tmin': 'min_temp', 
                          'prcp': 'precipitation', 
                          'snwd': 'snow_depth', 
                          'dapr': 'days_in_multiday_precipitation', 
                          'dasf': 'days_in_multiday_snowfall', 
                          'mdpr': 'multiday_precipitation', 
                          'mdsf': 'multiday_snowfall'}, inplace = True)

# Prcp is in tenths of mm's (like temperature), we need to convert to float
# and then divide by 10 to get the actual number of mms that it rained / temp. 
climate[['precipitation', 'multiday_precipitation']]=\
climate[['precipitation', 'multiday_precipitation']]\
.astype('float64') / 10


# Also need to convert these to F
climate.max_temp = (climate.max_temp / 10 * 1.8) + 32
climate.min_temp = (climate.min_temp / 10 * 1.8) + 32


# The warning is annoying, so I am going to turn it off. 


# Quick function for setting certain columns to zero
def set_zero(df, column):
    df[column] = df[column].fillna(0)

set_zero(climate, ['snow', 'snow_depth', 'days_in_multiday_precipitation', 
                   'days_in_multiday_snowfall', 'multiday_precipitation', 
                   'multiday_snowfall'])


# Quick function to convert mms(how we have it) into inches(freedom units)
def to_inches(df, column):
    df[column] = df[column] * 0.03937008
to_inches(climate, ['precipitation', 'snow', 'snow_depth', 'multiday_precipitation', 'multiday_snowfall'])
# print(climate.multiday_precipitation.unique())


# I had to cross reference some information to be sure that what I had in this
# table was actually accurate. I looked up historical weather from another source
# and it confirms that the data I have now is correct.
# print(climate.loc[44730])


# After looking at the output of this, I realize this was only the measure of how
# deep the snow was as time progressed, this doesn't help with climate pred.
climate.drop('snow_depth', axis = 1, inplace = True)

# I think it might be a good idea to readd the date column in case it is useful
# later, I don't know if it will be, but we will see. 
climate['date'] = df.date


# I had a feeling that the snow was not accounted for in the precipitation column,
# so I went ahead and looked. I also found out that I need to add most of the 
# precipitation type columns together to get a better idea of the total, so lets
# do that now
# print(climate.loc[44650:44700])

climate['total_precipitation'] = climate.precipitation + climate.snow +\
                                 climate.multiday_precipitation + \
                                 climate.multiday_snowfall
climate.is_copy = False
#Dropping the columns I just added together
climate.drop(['precipitation', 'multiday_precipitation', 'snow',
              'multiday_snowfall', 'days_in_multiday_precipitation', 
              'days_in_multiday_snowfall'], axis = 1, inplace = True)


# I want to preface this by saying there are 222 missing max_temp values,
# and 122 missing min temp values. I am going to replace them with the mean value
# of the column beacuse the count of total missing values are not concidered 
# statistically significant. 
# print(climate.isnull().sum())
climate.max_temp = climate.max_temp.fillna(np.mean(climate.max_temp))
climate.min_temp = climate.min_temp.fillna(np.mean(climate.min_temp))


#Okay, so I think we can can now start a machine learning model with the cleaned data
# I think I am going to start with something basic, but I will use a few different 
# models and we will work our way to tensorflow. I also will probably do some 
# statistical testing later on, but that will require me splitting the dataset 
# by both month and year, which will be a lot of work. Also, the
# RandomForestClassifier is not the right choice for this dataset, but I want to
# see how well it preforms anyway. It isnt the right dataset because I need to 
# not just predict the amount of precipitation, but also the weather conditions.
# That means I will need to probably build both a MultiLabelClassifier as well as a
# Tensorflow model to properly predict the data.

RFC = RandomForestClassifier()
minmax = MinMaxScaler()
std = StandardScaler()
normal = Normalizer()
le = LabelEncoder()


#I just realized I forgot to add the weather attribute columns. Let me do that
# before I build the model.

# To add the weather attrubutes, I am going to select them from the og dataframe.
attributes = df[['wt01', 'wt03', 'wt04', 'wt05', 'wt06', 'wt11','wt14','wt16']]
attributes.fillna(0, inplace = True)

# Important part of the ML model is to make sure they are in bool form.
attributes = attributes.astype('bool')

# True/False values look ugly, so lets fix that.
attributes.apply([lambda x: 1 if x == True else 0])

# I dont like the og column names either.
attributes.rename(columns = {'wt01': 'fog', 'wt03': 'thunder', 'wt04': 'sleet',
                            'wt05': 'hail', 'wt06': 'rime', 'wt11': 'high_winds',
                            'wt14': 'drizzle', 'wt16': 'rain'}, inplace = True)

# Add climate and attributes.
climate = pd.concat([climate, attributes], axis = 1)


# I dont really know why I still have nan values, but they are easy to replace.
climate.total_precipitation.fillna(0, inplace = True)


#Feature and label creation. FINALLY.
rf_features = climate[['year', 'month', 'day', 'max_temp', 'min_temp']].to_numpy()
# rf_labels = climate.total_precipitation.to_numpy()

#We need to convert our labels to someting RandomForestClassifier can understand.
# It does not understand continuious, so while I could convert it to an int, I dont
# think that is statistically the best way to do that. Sklearn as a preprocessing
# model for label encoding, which I used below to relabel the labels into something
# it understands and can read in on.
# rf_labels = le.fit_transform(rf_labels)

climate.total_precipitation = \
    climate.total_precipitation.apply([lambda x: 1 if x > 0 else 0])
rf_labels = climate.total_precipitation
train_data, test_data, train_labels, test_labels =\
train_test_split(rf_features, rf_labels, test_size = 0.2)



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
# There is a 25% chance it will




# I want to seperate my data by month, so we are going to do that to run some statistical testing. I will probably
# also seperate by year stating at 2000 or so.

def by_month(df, month):
    return df.loc[(df.month == month)]

climate_january = by_month(climate, 1)
climate_february = by_month(climate, 2)
climate_march = by_month(climate, 3)
climate_april = by_month(climate, 4)
climate_may = by_month(climate, 5)
climate_june = by_month(climate, 6)
climate_july = by_month(climate, 7)
climate_august = by_month(climate, 8)
climate_september = by_month(climate, 9)
climate_october = by_month(climate, 10)
climate_november = by_month(climate, 11)
climate_december = by_month(climate, 12)

# Seperating by year.
def by_year(df, year):
    return df.loc[(df.year == year)]

climate_2000 = by_year(climate, 2000)
climate_2001 = by_year(climate, 2001)
climate_2002 = by_year(climate, 2002)
climate_2003 = by_year(climate, 2003)
climate_2004 = by_year(climate, 2004)
climate_2005 = by_year(climate, 2005)
climate_2006 = by_year(climate, 2006)
climate_2007 = by_year(climate, 2007)
climate_2008 = by_year(climate, 2008)
climate_2009 = by_year(climate, 2009)
climate_2010 = by_year(climate, 2010)
climate_2011 = by_year(climate, 2011)
climate_2012 = by_year(climate, 2012)
climate_2013 = by_year(climate, 2013)
climate_2014 = by_year(climate, 2014)
climate_2015 = by_year(climate, 2015)
climate_2016 = by_year(climate, 2016)
climate_2017 = by_year(climate, 2017)
climate_2018 = by_year(climate, 2018)
climate_2019 = by_year(climate, 2019)
climate_2020 = by_year(climate, 2020)
climate_2021 = by_year(climate, 2021)
climate_2022 = by_year(climate, 2022)

# print(climate_2022.head())