# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 07:02:56 2020

@author: SanketM
"""


import os
print(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2.csv')

print(series.head())


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()

#series.set_index('timestamp')
#pd.plotting.lag_plot(series)

values = pd.DataFrame(series.value)
df = pd.concat([values.shift(1),values],axis =1)
df.columns = ['t-1','t']
result = df.corr()
print(result)


#This function removes the trend and seasonsality
#the interval argument could be set to 7 to remove weekly seasonality, 360 for yearly and so on.
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)


s1values = difference(series.value)
print(s1values)
plt.plot_date(series['timestamp'][:len(series)-1],s1values)
#plt.gcf().autofmt_xdate()

from statsmodels.tsa.stattools import adfuller
adfullerresult = adfuller(series.value)
print(adfullerresult)