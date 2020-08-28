#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Autoregression-modelling-of-Time-series-for-Outlier-detection" data-toc-modified-id="Autoregression-modelling-of-Time-series-for-Outlier-detection-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Autoregression modelling of Time series for Outlier detection</a></span></li><li><span><a href="#Persistence-Model" data-toc-modified-id="Persistence-Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Persistence Model</a></span></li><li><span><a href="#AutoRegression-model" data-toc-modified-id="AutoRegression-model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>AutoRegression model</a></span><ul class="toc-item"><li><span><a href="#Auto-Correlation-at-lag-=-k" data-toc-modified-id="Auto-Correlation-at-lag-=-k-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Auto Correlation at lag = k</a></span></li></ul></li><li><span><a href="#Get-all-anomalies" data-toc-modified-id="Get-all-anomalies-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Get all anomalies</a></span></li><li><span><a href="#Using-AR-to-model-whole-TS" data-toc-modified-id="Using-AR-to-model-whole-TS-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Using AR to model whole TS</a></span></li><li><span><a href="#Determine-Seasonality" data-toc-modified-id="Determine-Seasonality-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Determine Seasonality</a></span></li><li><span><a href="#Stationarize-Time-Series-acc-to-Seasonality" data-toc-modified-id="Stationarize-Time-Series-acc-to-Seasonality-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Stationarize Time Series acc to Seasonality</a></span></li><li><span><a href="#Retrain-AutoReg-on-Stationary-TS" data-toc-modified-id="Retrain-AutoReg-on-Stationary-TS-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Retrain AutoReg on Stationary TS</a></span></li><li><span><a href="#Test-Stationarity-using-the-Augmented-Dicky-Fuller-Test" data-toc-modified-id="Test-Stationarity-using-the-Augmented-Dicky-Fuller-Test-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Test Stationarity using the Augmented Dicky Fuller Test</a></span></li><li><span><a href="#Define-threshold" data-toc-modified-id="Define-threshold-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Define threshold</a></span></li><li><span><a href="#See-if-AR-model-+-threshold-=-Anomalies" data-toc-modified-id="See-if-AR-model-+-threshold-=-Anomalies-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>See if AR model + threshold = Anomalies</a></span></li><li><span><a href="#Anomaly-Detection-with-the-differenced-AR-model" data-toc-modified-id="Anomaly-Detection-with-the-differenced-AR-model-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Anomaly Detection with the differenced AR model</a></span></li><li><span><a href="#Calculate-F1-score" data-toc-modified-id="Calculate-F1-score-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Calculate F1 score</a></span></li><li><span><a href="#Doing-this-in-a-loop" data-toc-modified-id="Doing-this-in-a-loop-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Doing this in a loop</a></span></li></ul></div>

# # Autoregression modelling of Time series for Outlier detection 
# By Sanket Mehrotra
# 
# Source Ref: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

# A regression model provides an output based on a linear combination of input values
# $$
# \hat{y} = b_0 + b_1*X_1
# $$

# an AR model assumes autocorrelation

# The stronger the correlation between the output variable and a specific lagged variable, the **more weight** that autoregression model can put on that variable when modeling.

# In[1]:


import os
os.getcwd()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt 
from sklearn.metrics import mean_squared_error
import numpy as np


# In[3]:


series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')

print(series.head())


# In[4]:


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()


# In[5]:


from statsmodels.graphics.tsaplots import plot_acf
#series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
plot_acf(series['value'], lags=100)
#plt.clf()


# In[6]:


df_indexed = series.copy().set_index('timestamp')


# In[7]:


# create lagged dataset
values = pd.DataFrame(series.value)
lagged_df = pd.concat([values.shift(1),values],axis =1)
print("df\n",lagged_df.head())
lagged_df.columns = ['t-1','t']
result = lagged_df.corr()
print("Correlation Matrix Result for lag = 1: ", "\n" ,result)


# In[8]:


# split into train and test sets
X = lagged_df.values
print("X","\n",X)
train, test = X[1:len(X)-7], X[len(X)-7:]
print("\ntrain\n",train,"\ntest\n",test)
train_X, train_y = train[:,0], train[:,1]
print("\ntrainX\n",train_X,"\ntrain_y\n",train_y)
test_X, test_y = test[:,0], test[:,1]
print("\ntestX\n",test_X,"\ntest_y\n",test_y)


# # Persistence Model

# In[9]:


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


# # AutoRegression model

# In[10]:


from statsmodels.tsa.ar_model import AutoReg

#Load the series
#series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
series = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2.csv')


# In[11]:


# split dataset
X = series.value
train, test = X[1:len(X)-20], X[len(X)-20:]


# In[12]:


# train autoregression
model = AutoReg(train, lags=31)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)


# In[13]:


# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
predictions = predictions.replace(np.nan,0)
for i in range(len(predictions)):
 	print('predicted=%f, expected=%f' % (predictions.iloc[i], test.iloc[i]))


# In[14]:


rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[15]:


#plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# ## Auto Correlation at lag = k 

# Calculate the Pearson Correlation coefficient for different lags to see how that varies.

# In[16]:


values = pd.DataFrame(series.value)
values


# In[17]:


predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
predictions = predictions.replace(np.nan,0)


# In[18]:


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


# # Get all anomalies

# In[19]:


##Get all anomalies
serAnomaly = series[series.is_anomaly == 1]
insAnomaly = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2_outliers.csv')

# allAnomaly = insAnomaly.set_index('timestamp').join(serAnomaly.set_index('timestamp'),on='timestamp',lsuffix="inserted",rsuffix="original")
allAnomaly = insAnomaly.join(serAnomaly.set_index('timestamp'),on='timestamp',lsuffix="inserted",rsuffix="original",how='outer')


# In[20]:


allAnomaly


# # Using AR to model whole TS

# I'm planning on doing this by fitting the model on a TS, then running forecast/predict with the index arguments from 0 -> train + test

# Of all the mutated datasets M1 - M5, The original series, M1 and M2 and M3 may be suitable for testing, I don't get M4 and M5 may overwhelm the results.

# In[21]:


series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv')
series.head()


# In[22]:


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'],series['value'])
plt.gcf().autofmt_xdate()


# In[23]:


# s1 = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M1.csv')
# s1.head()
# s1['timestamp'] = pd.to_datetime(s1['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
# plt.plot_date(s1['timestamp'],s1['value'])
# plt.gcf().autofmt_xdate()

# s2 = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M2.csv')
# s2.head()
# s2['timestamp'] = pd.to_datetime(s2['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
# plt.plot_date(s2['timestamp'],s2['value'])
# plt.gcf().autofmt_xdate()

# s3 = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M3.csv')
# s3.head()
# s3['timestamp'] = pd.to_datetime(s3['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
# plt.plot_date(s3['timestamp'],s3['value'])
# plt.gcf().autofmt_xdate()

# s4 = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M4.csv')
# s4.head()
# s4['timestamp'] = pd.to_datetime(s4['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
# plt.plot_date(s4['timestamp'],s4['value'])
# plt.gcf().autofmt_xdate()

# s5 = pd.read_csv('D://Temp/time-series/data/yahoo-mutated/A2/synthetic_1_value_M5.csv')
# s5.head()
# s5['timestamp'] = pd.to_datetime(s5['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
# plt.plot_date(s5['timestamp'],s5['value'])
# plt.gcf().autofmt_xdate()


# In[24]:


# split dataset
X = series.value
size = int(len(X)*0.66) #
train, test = X[1:size], X[1:len(X)]


# The above line is training the AR model on half the series and testing it on the whole series.

# In[25]:


from statsmodels.tsa.ar_model import AutoReg
# train autoregression
model = AutoReg(train, lags=30)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)


# The below code makes the fitted AR model predict/model the TS from the first element to the last element, given it was trained on half the series. If it was trained on the whole series, then it seemed to overfit and predict everything perfectly.

# In[26]:


# make predictions
predictions = model_fit.predict(start=1, end=len(X)-1, dynamic=False)
predictions = predictions.replace(np.nan,0)
for i in range(len(predictions)):
 	print('predicted=%f, expected=%f' % (predictions.iloc[i], test.iloc[i]))


# In[27]:


len(test),len(predictions)


# In[28]:


rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[29]:


#plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# In[30]:


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'][:len(series)-1],test)
plt.plot_date(series['timestamp'][:len(series)-1],predictions,color='red')
plt.gcf().autofmt_xdate()


# In[31]:


df_indexed.info()


# In[34]:


#predictdf.set_index('timestamp').info()


# In[35]:


predictdf = pd.concat([series['timestamp'][:len(series)-1],predictions], axis=1)

predictdf.rename(columns={0:'value'},inplace=True)

predictdf['value'] = predictdf['value'].shift(-2)

predictdf = predictdf.dropna()
predictdf

predictdf = predictdf.set_index('timestamp')


# # Determine Seasonality
# 
# **WHY do this??**
# 
# If we know the seasonality, we can perform the right kind of differencing operation on the Time series to stationarize it. 
# Maybe it will make the TS simpler to model and a simpler model will make the outliers stand out more in the model.

# In[36]:


# from statsmodels.tsa.seasonal import seasonal_decompose

# df_indexed = series.copy().set_index('timestamp')

# # Additive Decomposition
# result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')

# # Plot
# plt.rcParams.update({'figure.figsize': (10,10)})
# #result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
# result_add.plot().suptitle('Additive Decompose', fontsize=22)
# plt.show()


# # Stationarize Time Series acc to Seasonality

# In[37]:


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset.iloc[i] - dataset.iloc[i - interval]
		diff.append(value)
	return pd.Series(diff)

interval = 1
stTS = difference(series['value'],interval)
plt.plot_date(series['timestamp'][interval:],stTS)
plt.gcf().autofmt_xdate()


# In[104]:


stTSseries = pd.concat([series['timestamp'][:len(series)],predictions2], axis=1)
stTSseries = stTSseries.dropna()
stTSseries.rename(columns={0:'value'},inplace=True)
stTSseries


# Even thought the seasonality may be 7, 30, 365 days, differencing with lag = 1 is the most effective.

# We can see that the outliers really stand out right now.

# # Retrain AutoReg on Stationary TS

# In[107]:


len(stTS),len(series)


# In[108]:


# split dataset
X = stTS
size = int(len(X)*0.66)
train, test = X[1:size], X[1:len(X)]


# In[109]:


# train autoregression
model2 = AutoReg(train, lags=30)
model_fit2 = model2.fit()
print('Coefficients: %s' % model_fit2.params)


# The below code makes the fitted AR model predict/model the TS from the first element to the last element, given it was trained on half the series. If it was trained on the whole series, then it seemed to overfit and predict everything perfectly.

# In[124]:


# make predictions
predictions2 = model_fit2.predict(start=30, end=len(X) - 1, dynamic=False)
#predictions2 = predictions2.replace(np.nan,0)

#since training with lag = 30 , use the training values for first 30 predictions. 
# for i in range(30):
#     predictions2.iloc[i] = train.iloc[i]
    
for i in range(len(predictions2)):
 	print('predicted=%f, expected=%f' % (predictions2.iloc[i], test.iloc[i]))


# In[121]:


len(predictions2),len(test),len(series)


# In[114]:


rmse = sqrt(mean_squared_error(test, predictions2))
print('Test RMSE: %.3f' % rmse)


# In[115]:


#plot results
plt.plot(test)
plt.plot(predictions2, color='red')
plt.show()


# In[116]:


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'][:len(series)-2],test)
plt.plot_date(series['timestamp'][:len(series)-2],predictions2,color='red')
plt.gcf().autofmt_xdate()


# In[46]:


predictdf2 = pd.concat([series['timestamp'][:len(series)-1],predictions2], axis=1)


# In[47]:


predictdf2.rename(columns={0:'value'},inplace=True)


# In[48]:


predictdf2['value'] = predictdf2['value'].shift(-2)


# In[49]:


predictdf2 = predictdf2.dropna()
predictdf2


# # Test Stationarity using the Augmented Dicky Fuller Test

# In[50]:


from statsmodels.tsa.stattools import adfuller
X = stTS
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# The more negative this number, the more confidently we can reject the possiblity that the series is stationary. 

# # Define threshold

# Manually at the moment.

# In[51]:


stTS


# In[52]:


stTS.mean(), stTS.median(), stTS.mode()


# Very obvious outliers, let's try setting the threshold manually for now to 500

# In[53]:


threshold = 1200
st_threshold = 500


# can also try the z score, or the inter quartile range method or the n std deviations method

# # See if AR model + threshold = Anomalies

# In[54]:


test


# In[55]:


predictions, type(predictions)


# In[56]:


predictdf


# In[57]:


series


# In[58]:


series[series.is_anomaly == 1]


# In[59]:


threshold


# In[60]:


df_indexed


# In[62]:


#predictdf.set_index('timestamp')


# In[63]:


matrix = df_indexed.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')


# In[64]:


matrix


# In[65]:


outliers = matrix[matrix.value_data > matrix.value_predict + threshold]


# In[66]:


outliers


# In[67]:


# outliers = outliers.append(predictdf[predictdf['value'] < -1*(threshold)])


# In[68]:


outliers


# In[69]:


outliers = matrix[matrix.value_data > matrix.value_predict + threshold]
not_outliers = matrix[matrix.value_data <= matrix.value_predict + threshold]
#not_outliers.append(predictdf[predictdf['value'] > -1*(threshold)])
not_outliers


# In[70]:


p = len(series[series.is_anomaly == 1])
p

n = len(series[series.is_anomaly == 0])
n


# In[71]:


series.set_index('timestamp')


# In[72]:


# predictdf.set_index('timestamp')


# In[73]:


# series.set_index('timestamp').join(predictdf.set_index('timestamp'),on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')


# In[75]:


#outliers.set_index('timestamp')


# In[76]:


checkingmatrix = series.set_index('timestamp').join(outliers,on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
checkingmatrix


# In[77]:


truepositives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 1]
truepositives


# In[78]:


falsepostives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 0]
falsepostives


# In[79]:


checkingmatrix2 =series.set_index('timestamp').join(not_outliers,on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
checkingmatrix2


# In[80]:


truenegatives=checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 0]
truenegatives


# In[81]:


truenegatives=checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 0]
falsenegatives = checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 1]
falsenegatives


# # Anomaly Detection with the differenced AR model

# In[82]:


predictions2


# In[83]:


predictions2.mean(),predictions2.std()


# In[84]:


test


# In[92]:


series, type(series)


# In[93]:


predictdf2,type(predictdf2)


# In[87]:


len(predictdf2[predictdf2['value'] > st_threshold]),len(predictdf2[predictdf2['value'] < st_threshold])


# In[88]:


len(predictdf2[predictdf2['value'] < (-1)*st_threshold]),len(predictdf2[predictdf2['value'] > (-1)*st_threshold])


# In[89]:


st_threshold = 300


# In[106]:


stTSseries['value']


# In[95]:


predictdf2['value']


# In[ ]:





# In[105]:


outliers2 = predictdf2[stTSseries['value'] > predictdf2['value'] + st_threshold]
#outliers2 = outliers2.append(predictdf2[predictdf2['value'] < (-1) * st_threshold])


# In[ ]:


not_outliers2 =  predictdf2[predictdf2['value'] < st_threshold]
not_outliers2 =  not_outliers2.append(predictdf2[predictdf2['value'] > (-1) * st_threshold])


# In[ ]:


outliers2


# In[ ]:


outliers2.drop_duplicates(subset='timestamp')


# In[ ]:


not_outliers2


# In[ ]:


not_outliers2.drop_duplicates(subset='timestamp')


# In[ ]:


outliers2 = outliers2.set_index('timestamp')
not_outliers2 = not_outliers2.set_index('timestamp')


# In[ ]:


posmatrix = df_indexed.join(outliers2, on='timestamp',how='inner',lsuffix = '_test',rsuffix='_predict')
negmatrix = df_indexed.join(not_outliers2, on='timestamp',how='inner',lsuffix = '_test',rsuffix='_predict')


# In[ ]:


posmatrix,negmatrix


# In[ ]:


negmatrix


# In[ ]:


negmatrix.drop_duplicates()


# # Calculate F1 score

# In[ ]:


p = len(series[series.is_anomaly == 1])
p


# In[ ]:


n = len(series[series.is_anomaly == 0])
n


# In[ ]:


precision = len(truepositives)/(len(truepositives) + len(falsepostives))


# In[ ]:


recall = len(truepositives)/(len(truepositives) + len(falsenegatives))


# $$ F_1 = 2 * \frac{precision * recall}{precision +
# recall} $$

# In[ ]:


f1 = 2*(precision * recall)/(precision + recall)


# In[ ]:


f1


# # Doing this in a loop

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg


# In[ ]:


start_time = datetime.now()
all_csv = glob.glob(f'../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')


# In[ ]:


start_time = datetime.now() 
f1_plot = []
rmse_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
#     if index > 30:
#          break
    
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df_indexed = df.set_index('timestamp')
    #print(df_indexed)
    
    # split dataset
    X = df.value
    size = int(len(X)*0.66)
    train, test = X[1:size], X[1:len(X)]
    
    # train autoregression
    model = AutoReg(train, lags=30)
    model_fit = model.fit()

    # make predictions
    predictions = model_fit.predict(start=1, end=len(X)-1, dynamic=False)
    predictions = predictions.replace(np.nan,0)

    #calculate rmse
    rmse = sqrt(mean_squared_error(test, predictions))
    print(f'file: {file}')
    print('Test RMSE: %.3f' % rmse)
    rmse_plot.append(rmse)
    
    #plot results
    #plt.rcParams.update({'figure.figsize': (10,10)})
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.savefig("./ARoutput/" + fname +"ARLag30")
    plt.show()
    
    
    predictdf = pd.concat([df['timestamp'][:len(df)-1],predictions], axis=1)
    predictdf.rename(columns={0:'value'},inplace=True)
    predictdf['value'] = predictdf['value'].shift(-2)
    predictdf = predictdf.dropna()
    predictdf = predictdf.set_index('timestamp')
    
    matrix = df_indexed.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')
    
    threshold = 1200
    
    outliers = matrix[matrix.value_data > matrix.value_predict + threshold]
    not_outliers = matrix[matrix.value_data <= matrix.value_predict + threshold]
    
    print("threshold: ", threshold)
    print("Outliers:")
    print(outliers)
    
#     p = len(series[series.is_anomaly == 1])
#     n = len(series[series.is_anomaly == 0])

    checkingmatrix = df.set_index('timestamp').join(outliers,on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
    truepositives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 1]
    falsepostives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 0]
    
    checkingmatrix2 =df.set_index('timestamp').join(not_outliers,on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
    truenegatives=checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 0]
    falsenegatives = checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 1]
    
    if(len(truepositives) + len(falsepostives) > 0):
        precision = len(truepositives)/(len(truepositives) + len(falsepostives))
        print("Precision: ", precision)
        precision_plot.append(precision)
    else:
        print("truepositives + falsepositives = 0")
        continue
    
    
    recall = len(truepositives)/(len(truepositives) + len(falsenegatives))
    print("Recall:", recall)
    recall_plot.append(recall)
    if(precision + recall > 0):
        f1 = 2*(precision * recall)/(precision + recall)
        print("F1:",f1)
        f1_plot.append(f1)
    else:
        print("precision + recall = 0")
        continue
    

    
end_time=datetime.now()
print(f"AR Modelling and Anomaly Analysis of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)


# In[ ]:


print(sum(f1_plot)/len(f1_plot))


# In[ ]:




