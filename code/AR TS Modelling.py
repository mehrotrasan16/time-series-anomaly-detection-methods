# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:17:04 2020

@author: SanketM
"""



# from pandas import read_csv
# from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
print(series.head())
#series = read_csv('', header=0, index_col=0)
series.plot()
plt.show()

series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()

# series.set_index('timestamp')
# pd.plotting.lag_plot(series)