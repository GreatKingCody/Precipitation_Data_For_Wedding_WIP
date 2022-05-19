import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt

# from scipy.stats import chi2_contingency
# import matplotlib.ticker as mtick



# from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


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
tf_precipitation = climate.total_precipitation
# I came back and added this so that I can use it for my tensorflow model

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



#I just realized I forgot to add the weather attribute columns. Let me do that
# before I build the model.

# To add the weather attrubutes, I am going to select them from the og dataframe.
attributes = df[['wt01', 'wt03', 'wt04', 'wt05', 'wt06', 'wt11','wt14','wt16']]
attributes.fillna(0, inplace = True)

# Important part of the ML model is to make sure they are in bool form.
attributes = attributes.astype('bool')

# True/False values look ugly, so lets fix that.
attributes = attributes.apply([lambda x: 1 if x == True else 0])

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
climate.total_precipitation = \
    climate.total_precipitation.apply([lambda x: 1 if x > 0 else 0])
rf_labels = climate.total_precipitation
train_data, test_data, train_labels, test_labels =\
train_test_split(rf_features, rf_labels, test_size = 0.2)


# I am going to stop on this file here, and seperate it out into
# other files to keep it a little more organized. 

