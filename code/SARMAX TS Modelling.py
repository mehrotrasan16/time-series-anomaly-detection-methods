# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 07:03:05 2020

@author: SanketM
"""


import os
print(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
series = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2.csv')


print(series.head())


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')  #format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()

series.set_index('timestamp')
#pd.plotting.lag_plot(series)

values = pd.DataFrame(series.value)
df = pd.concat([values.shift(1),values],axis =1)
df.columns = ['t-1','t']
result = df.corr()
print(result)

#plot the autocorr func to see if there is a statistically significant difference for ARIMA analysis
from statsmodels.graphics.tsaplots import plot_acf
series = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2.csv')
plot_acf(series['value'], lags=100)
## Anywhere within the first forty lags is statistically significant for an ARIMA analysis




##for the ARIMA model, we use statsmodels library ARIMA() module, then call fit() on it and then predict()
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# # fit model
# model = ARIMA(series.value, order=(15,1,0)) # lag value to 30 for autoregression, uses a difference order of 1 to make the time series stationary, and uses a moving average model of 0.
# model_fit = model.fit() #disp= 0 to hide debug information
# print(model_fit.summary())

# # plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())


X = series.value
print("\n\nX\n",X)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
print("\ntrain\n",train,"\ntest\n",test)
history = [x for x in train]
predictions = list()
for t in range(len(test)):
 	model = SARIMAX(history, order=(5,1,0))
 	model_fit = model.fit(disp=0)
 	output = model_fit.forecast()
 	yhat = output[0]
 	predictions.append(yhat)
 	obs = test.iloc[t]
 	history.append(obs)
 	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
fig1 = plt.figure()
plt.plot(test.tolist())
plt.plot(predictions, color='red')
plt.show()
