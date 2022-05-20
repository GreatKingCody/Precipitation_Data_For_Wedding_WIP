from Final import climate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time                                          
import datetime as dt

from scipy.stats import chi2_contingency


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

climate_1899 = by_year(climate, 1899)
climate_1900 = by_year(climate, 1900)
climate_1901 = by_year(climate, 1901)
climate_1902 = by_year(climate, 1902)
climate_1903 = by_year(climate, 1903)
climate_1904 = by_year(climate, 1904)
climate_1905 = by_year(climate, 1905)
climate_1906 = by_year(climate, 1906)
climate_1907 = by_year(climate, 1907)
climate_1908 = by_year(climate, 1908)
climate_1909 = by_year(climate, 1909)
climate_1910 = by_year(climate, 1910)
climate_1911 = by_year(climate, 1911)
climate_1912 = by_year(climate, 1912)
climate_1913 = by_year(climate, 1913)
climate_1914 = by_year(climate, 1914)
climate_1915 = by_year(climate, 1915)
climate_1916 = by_year(climate, 1916)
climate_1917 = by_year(climate, 1917)
climate_1918 = by_year(climate, 1918)
climate_1919 = by_year(climate, 1919)

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



def mean_max_temp(dataframe):
    return dataframe.max_temp.mean()

max_temp_mean_1899 = mean_max_temp(climate_1899)
max_temp_mean_1900 = mean_max_temp(climate_1900)
max_temp_mean_1901 = mean_max_temp(climate_1901)
max_temp_mean_1902 = mean_max_temp(climate_1902)
max_temp_mean_1903 = mean_max_temp(climate_1903)
max_temp_mean_1904 = mean_max_temp(climate_1904)
max_temp_mean_1905 = mean_max_temp(climate_1905)
max_temp_mean_1906 = mean_max_temp(climate_1906)
max_temp_mean_1907 = mean_max_temp(climate_1907)
max_temp_mean_1908 = mean_max_temp(climate_1908)
max_temp_mean_1909 = mean_max_temp(climate_1909)
max_temp_mean_1910 = mean_max_temp(climate_1910)
max_temp_mean_1911 = mean_max_temp(climate_1911)
max_temp_mean_1912 = mean_max_temp(climate_1912)
max_temp_mean_1913 = mean_max_temp(climate_1913)
max_temp_mean_1914 = mean_max_temp(climate_1914)
max_temp_mean_1915 = mean_max_temp(climate_1915)
max_temp_mean_1916 = mean_max_temp(climate_1916)
max_temp_mean_1917 = mean_max_temp(climate_1917)
max_temp_mean_1918 = mean_max_temp(climate_1918)
max_temp_mean_1919 = mean_max_temp(climate_1919)

max_temp_mean_2001 = mean_max_temp(climate_2001)
max_temp_mean_2002 = mean_max_temp(climate_2002)
max_temp_mean_2003 = mean_max_temp(climate_2003)
max_temp_mean_2004 = mean_max_temp(climate_2004)
max_temp_mean_2005 = mean_max_temp(climate_2005)
max_temp_mean_2006 = mean_max_temp(climate_2006)
max_temp_mean_2007 = mean_max_temp(climate_2007)
max_temp_mean_2008 = mean_max_temp(climate_2008)
max_temp_mean_2009 = mean_max_temp(climate_2009)
max_temp_mean_2010 = mean_max_temp(climate_2010)
max_temp_mean_2011 = mean_max_temp(climate_2011)
max_temp_mean_2012 = mean_max_temp(climate_2012)
max_temp_mean_2013 = mean_max_temp(climate_2013)
max_temp_mean_2014 = mean_max_temp(climate_2014)
max_temp_mean_2015 = mean_max_temp(climate_2015)
max_temp_mean_2016 = mean_max_temp(climate_2016)
max_temp_mean_2017 = mean_max_temp(climate_2017)
max_temp_mean_2018 = mean_max_temp(climate_2018)
max_temp_mean_2019 = mean_max_temp(climate_2019)
max_temp_mean_2020 = mean_max_temp(climate_2020)
max_temp_mean_2021 = mean_max_temp(climate_2020)


