# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:20:23 2020

@author: SanketM
"""


import os
print(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt 
from sklearn.metrics import mean_squared_error
import numpy as np

series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')

print(series.head())


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()
#plt.clf()
# from pandas.plotting import lag_plot
# series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
# lag_plot(series)

# from pandas.plotting import autocorrelation_plot
# series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
# autocorrelation_plot(series)

from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series['value'], lags=100)
#plt.clf()

series.set_index('timestamp')
df_indexed = series.copy().set_index('timestamp')

# create lagged dataset
values = pd.DataFrame(series.value)
df = pd.concat([values.shift(1),values],axis =1)
print("df\n",df.head())
df.columns = ['t-1','t']
result = df.corr()
print("Correlation Matrix Result for lag = 1: ", "\n" ,result)

# split into train and test sets
X = df.values
print("X","\n",X)
train, test = X[1:len(X)-7], X[len(X)-7:]
print("\ntrain\n",train,"\ntest\n",test)
train_X, train_y = train[:,0], train[:,1]
print("\ntrainX\n",train_X,"\ntrain_y\n",train_y)
test_X, test_y = test[:,0], test[:,1]
print("\ntestX\n",test_X,"\ntest_y\n",test_y)

#Persistence Model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
fig1 = plt.figure()
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()

############################# AutoReg Model ######################
from statsmodels.tsa.ar_model import AutoReg

#Load the series
#series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
series = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2.csv')
# split dataset
X = series.value
train, test = X[1:len(X)-20], X[len(X)-20:]
# train autoregression
model = AutoReg(train, lags=31)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
print(predictions)
predictions = predictions.replace(np.nan,0)
for i in range(len(predictions)):
 	print('predicted=%f, expected=%f' % (predictions.iloc[i], test.iloc[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
#plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


##Plot a list of Autocorrelation vs Values at lags
autocorr = []
for lag in range(0,20):
    print("Lag : " + str(lag) + "\n")
    rmse = sqrt(mean_squared_error(test[:len(predictions)],predictions))
    print(f"Test RMSE: {rmse}")
    df = pd.concat([values.shift(lag),values],axis =1)
    df.columns = ['t-'+str(lag),'t']
    result = df.corr()
    autocorr.append(result.t[0])
    print(result)
    print("\n\n")
    
plt.plot([i for i in (autocorr)])

##Get all anomalies
serAnomaly = series[series.is_anomaly == 1]
insAnomaly = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2_outliers.csv')

# allAnomaly = insAnomaly.set_index('timestamp').join(serAnomaly.set_index('timestamp'),on='timestamp',lsuffix="inserted",rsuffix="original")
allAnomaly = insAnomaly.join(serAnomaly.set_index('timestamp'),on='timestamp',lsuffix="inserted",rsuffix="original",how='outer')

#cross check if the above join is legit
#insAnomaly.timestamp.isin(serAnomaly.timestamp)
#serAnomaly.timestamp.isin(insAnomaly.timestamp)
#checked, it is legit

##we have list of anomalies
##now we set a threshold, using k-sd method

