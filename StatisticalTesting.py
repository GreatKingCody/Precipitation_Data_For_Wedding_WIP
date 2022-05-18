from Final import climate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt



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