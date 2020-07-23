# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 04:00:34 2020

@author: SanketM
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(".\\data\\yahoo\\dataset\\ydata-labeled-time-series-anomalies-v1_0\\A2Benchmark\\synthetic_1.csv")
print(df)

print(df.describe())

print(df.info())

timest1 = df[:1]
print(timest1)

df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')

print(df)

df_indexed = df.set_index('timestamp')

print(df_indexed.info())

from statsmodels.tsa.seasonal import seasonal_decompose

# Multiplicative Decomposition 
#result_mul = seasonal_decompose(df_indexed['value'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('', fontsize=12)
plt.show()

threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print("File: data\\yahoo\\dataset\\ydata-labeled-time-series-anomalies-v1_0\\A2Benchmark\\synthetic_1.csv")
print("threshold: 500")
print("Outliers:")
print(outliers)
