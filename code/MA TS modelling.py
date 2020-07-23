# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 07:02:38 2020

@author: SanketM
"""


import os
print(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M5.csv')

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

# Tail-rolling average transform
#X = series.value
rolling = series.rolling(window=30)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
# plot original and transformed dataset
plt.plot_date(series['timestamp'],rolling_mean.value,color="red",fmt="-")
#rolling_mean.plot(color='red')
plt.show()


## Now Set threshold.
