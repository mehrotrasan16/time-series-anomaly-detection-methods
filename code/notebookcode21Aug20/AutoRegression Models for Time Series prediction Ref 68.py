#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Autoregression-modelling-of-Time-series-for-Outlier-detection" data-toc-modified-id="Autoregression-modelling-of-Time-series-for-Outlier-detection-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Autoregression modelling of Time series for Outlier detection</a></span></li><li><span><a href="#Persistence-Model" data-toc-modified-id="Persistence-Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Persistence Model</a></span></li><li><span><a href="#AutoRegression-model" data-toc-modified-id="AutoRegression-model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>AutoRegression model</a></span><ul class="toc-item"><li><span><a href="#Auto-Correlation-at-lag-=-k" data-toc-modified-id="Auto-Correlation-at-lag-=-k-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Auto Correlation at lag = k</a></span></li></ul></li><li><span><a href="#Get-all-anomalies" data-toc-modified-id="Get-all-anomalies-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Get all anomalies</a></span></li><li><span><a href="#Using-AR-to-model-whole-TS" data-toc-modified-id="Using-AR-to-model-whole-TS-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Using AR to model whole TS</a></span></li><li><span><a href="#Define-threshold" data-toc-modified-id="Define-threshold-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Define threshold</a></span></li><li><span><a href="#See-if-AR-model-+-threshold-=-Anomalies" data-toc-modified-id="See-if-AR-model-+-threshold-=-Anomalies-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>See if AR model + threshold = Anomalies</a></span></li><li><span><a href="#Stationarize-Time-Series-acc-to-Seasonality" data-toc-modified-id="Stationarize-Time-Series-acc-to-Seasonality-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Stationarize Time Series acc to Seasonality</a></span></li><li><span><a href="#Retrain-AutoReg-on-Stationary-TS" data-toc-modified-id="Retrain-AutoReg-on-Stationary-TS-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Retrain AutoReg on Stationary TS</a></span></li><li><span><a href="#Test-Stationarity-using-the-Augmented-Dicky-Fuller-Test" data-toc-modified-id="Test-Stationarity-using-the-Augmented-Dicky-Fuller-Test-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Test Stationarity using the Augmented Dicky Fuller Test</a></span></li><li><span><a href="#Anomaly-Detection-with-the-differenced-AR-model" data-toc-modified-id="Anomaly-Detection-with-the-differenced-AR-model-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Anomaly Detection with the differenced AR model</a></span></li><li><span><a href="#Calculate-F1-score" data-toc-modified-id="Calculate-F1-score-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Calculate F1 score</a></span></li><li><span><a href="#Doing-this-in-a-loop" data-toc-modified-id="Doing-this-in-a-loop-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Doing this in a loop</a></span></li><li><span><a href="#Loop-with-Standardized-data" data-toc-modified-id="Loop-with-Standardized-data-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Loop with Standardized data</a></span></li><li><span><a href="#Fixing-performance" data-toc-modified-id="Fixing-performance-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Fixing performance</a></span></li><li><span><a href="#Fixing-Performance-with-standardization" data-toc-modified-id="Fixing-Performance-with-standardization-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>Fixing Performance with standardization</a></span></li></ul></div>

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


# In[101]:


# split dataset
X = series.value
size = int(len(X)*0.66) #
train, test = X[1:size], X[1:len(X)]


# The above line is training the AR model on half the series and testing it on the whole series.

# In[102]:


from statsmodels.tsa.ar_model import AutoReg
# train autoregression
model = AutoReg(train, lags=30)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)


# The below code makes the fitted AR model predict/model the TS from the first element to the last element, given it was trained on half the series. If it was trained on the whole series, then it seemed to overfit and predict everything perfectly.

# In[128]:


# make predictions
predictions = model_fit.predict(start=1, end=len(X)-1, dynamic=False)
predictions = predictions.replace(np.nan,0)

for i in range(30):
    predictions.iloc[i] = train.iloc[i]
    
for i in range(len(predictions)):
 	print('predicted=%f, expected=%f' % (predictions.iloc[i], test.iloc[i]))


# In[104]:


len(test),len(predictions)


# In[105]:


rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[106]:


#plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# In[107]:


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'][:len(series)-1],test)
plt.plot_date(series['timestamp'][:len(series)-1],predictions,color='red')
plt.gcf().autofmt_xdate()


# In[108]:


df_indexed.info()


# In[109]:


#predictdf.set_index('timestamp').info()


# In[120]:


predictdf = pd.concat([series['timestamp'][:len(series)-1],predictions], axis=1)

predictdf.rename(columns={0:'value'},inplace=True)

#predictdf['value'] = predictdf['value'].shift(-2)

predictdf = predictdf.dropna()
predictdf = predictdf.set_index('timestamp')
predictdf


# # Define threshold

# let's try setting the threshold manually for now to 1200

# In[121]:


threshold = 1200


# can also try the z score, or the inter quartile range method or the n std deviations method

# # See if AR model + threshold = Anomalies

# In[122]:


test


# In[123]:


predictions, type(predictions)


# In[124]:


predictdf


# In[125]:


series


# In[116]:


series[series.is_anomaly == 1]


# In[117]:


threshold


# In[118]:


df_indexed


# In[43]:


#predictdf.set_index('timestamp')


# In[44]:


matrix = df_indexed.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')


# In[45]:


matrix


# In[46]:


outliers = matrix[matrix.value_data > matrix.value_predict + threshold]


# In[47]:


outliers


# In[48]:


# outliers = outliers.append(predictdf[predictdf['value'] < -1*(threshold)])


# In[49]:


outliers


# In[50]:


outliers = matrix[matrix.value_data > matrix.value_predict + threshold]
not_outliers = matrix[matrix.value_data <= matrix.value_predict + threshold]
#not_outliers.append(predictdf[predictdf['value'] > -1*(threshold)])
not_outliers


# In[51]:


p = len(series[series.is_anomaly == 1])
p

n = len(series[series.is_anomaly == 0])
n


# In[52]:


series.set_index('timestamp')


# In[53]:


# predictdf.set_index('timestamp')


# In[54]:


# series.set_index('timestamp').join(predictdf.set_index('timestamp'),on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')


# In[55]:


#outliers.set_index('timestamp')


# In[56]:


checkingmatrix = series.set_index('timestamp').join(outliers,on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
checkingmatrix


# In[57]:


truepositives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 1]
truepositives


# In[58]:


falsepostives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 0]
falsepostives


# In[59]:


checkingmatrix2 =series.set_index('timestamp').join(not_outliers,on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
checkingmatrix2


# In[60]:


truenegatives=checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 0]
truenegatives


# In[61]:


truenegatives=checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 0]
falsenegatives = checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 1]
falsenegatives


# # Stationarize Time Series acc to Seasonality

# In[62]:


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset.iloc[i] - dataset.iloc[i - interval]
		diff.append(value)
	return pd.Series(diff)


# In[63]:


interval = 1
stTS = difference(series['value'],interval)
plt.plot_date(series['timestamp'][interval:],stTS)
plt.gcf().autofmt_xdate()


# Even thought the seasonality may be 7, 30, 365 days, differencing with lag = 1 is the most effective.

# We can see that the outliers really stand out right now.

# In[64]:


stTS


# In[65]:


stTS.mean(), stTS.median(), stTS.mode()


# # Retrain AutoReg on Stationary TS

# In[95]:


len(stTS),len(series)


# In[96]:


# split dataset
X = stTS
size = int(len(X)*0.66)
train, test = X[1:size], X[1:len(X)]


# In[97]:


# train autoregression
model2 = AutoReg(train, lags=30)
model_fit2 = model2.fit()
print('Coefficients: %s' % model_fit2.params)


# The below code makes the fitted AR model predict/model the TS from the first element to the last element, given it was trained on half the series. If it was trained on the whole series, then it seemed to overfit and predict everything perfectly.

# In[98]:


# make predictions
predictions2 = model_fit2.predict(start=1, end=len(X)-1, dynamic=False)
predictions2 = predictions2.replace(np.nan,0)
for i in range(len(predictions2)):
 	print('predicted=%f, expected=%f' % (predictions2.iloc[i], test.iloc[i]))


# In[70]:


len(predictions2),len(test),len(series)


# In[71]:


type(predictions2)


# In[72]:


rmse = sqrt(mean_squared_error(test, predictions2))
print('Test RMSE: %.3f' % rmse)


# In[73]:


#plot results
plt.plot(test)
plt.plot(predictions2, color='red')
plt.show()


# In[74]:


series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
plt.plot_date(series['timestamp'][:len(series)-2],test)
plt.plot_date(series['timestamp'][:len(series)-2],predictions2,color='red')
plt.gcf().autofmt_xdate()


# In[75]:


predictdf2 = pd.concat([series['timestamp'][:len(series)-1],predictions2], axis=1)


# In[76]:


predictdf2.rename(columns={0:'value'},inplace=True)


# In[77]:


predictdf2['value'] = predictdf2['value'].shift(-2)


# In[78]:


predictdf2 = predictdf2.dropna()
predictdf2


# # Test Stationarity using the Augmented Dicky Fuller Test

# In[79]:


from statsmodels.tsa.stattools import adfuller
X = stTS
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# The more negative this number, the more confidently we can reject the possiblity that the series is stationary. 

# # Anomaly Detection with the differenced AR model

# In[80]:


predictions2


# In[81]:


predictions2.mean(),predictions2.std()


# In[82]:


test


# In[83]:


series, type(series)


# In[84]:


stTSDataframe = pd.concat([series['timestamp'],stTS],axis = 1 )
stTSDataframe.rename(columns={0:'value'},inplace=True)
stTSDataframe, type(stTSDataframe)


# In[85]:


predictdf2,type(predictdf2)


# In[88]:


st_threshold = 300


# In[94]:


predictdf2['value'], stTSDataframe['value']


# In[93]:


outliers2 = predictdf2['value'][stTSDataframe['value'] > predictdf2['value'] + st_threshold]
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

# In[1]:


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
from sklearn.preprocessing import StandardScaler
from math import sqrt


# In[2]:


start_time = datetime.now()
all_csv = glob.glob(f'../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')


# In[20]:


start_time = datetime.now() 
f1_plot = []
rmse_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 3:
         break
    
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
    
    #combine with df_indexed with predictions to pick out the outliers.
    matrix = df_indexed.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')
    
    #manual threshold set
    threshold = 1200
    
    #compare the real df value with the AR model and then 
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
print(f"Average F1 score over {index+1} runs is = {sum(f1_plot)/len(f1_plot)}")


# In[ ]:


print(sum(f1_plot)/len(f1_plot))


# # Loop with Standardized data

# In[10]:


start_time = datetime.now() 
f1_plot = []
rmse_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 3:
         break
    
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
    df_indexed = df.set_index('timestamp')
    #print(df_indexed)
    
    # prepare data for standardization
    values = df_indexed.copy()
    values = values.drop(columns=['is_anomaly'],axis=1)
    #values = values.reshape((len(values), 1))
    
    # train the standardization
    scaler2 = StandardScaler()
    scaler2 = scaler2.fit(values)
    print('Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
    # standardize the dataset
    standardaized = scaler2.transform(values)
    
    df['std_value'] = standardaized
    
    # split dataset
    X = df.std_value
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
    
    #combine with df_indexed with predictions to pick out the outliers.
    matrix = df_indexed.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')
    
    #manual threshold set
    threshold = 1200
    
    #compare the real df value with the AR model and then 
    outliers = matrix[matrix.std_value > matrix.value_predict + threshold]
    not_outliers = matrix[matrix.std_value <= matrix.value_predict + threshold]
    
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
print(f"Average F1 score over {index+1} runs is = {sum(f1_plot)/len(f1_plot)}")


# # Fixing performance

# The performance of the above test suggested I was not doing something right.
# So I checked out the ARIMA code from machine learning mastery.com to verify my implementation. They provided and alternate implementation that predicted each point in the test set and was updated subsequently by training a new model on all the predictions so far. 

# In[7]:


X = df_indexed.value
print("\n\nX\n",X)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
print("\ntrain\n",train,"\ntest\n",test)
history = [x for x in train]
predictions = list()
for t in range(len(test)):
 	model = AutoReg(history, lags=30)
 	model_fit = model.fit()
 	output = model_fit.predict(start=len(history), end=len(history)+1, dynamic=False)
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


# In[32]:


start_time = datetime.now() 
f1_plot = []
rmse_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 3:
         break
    
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
    df_indexed = df.set_index('timestamp')
    #print(df_indexed)
    
    # split dataset
    X = df.value
    size = int(len(X)*0.5)
    train, test = X[0:size], X[size:len(X)]
    
    history = [x for x in train]
    predictions = list()
        
    for t in range(len(test)):
        # train autoregression
        model = AutoReg(history, lags=30)
        model_fit = model.fit()
        # make predictions
        output = model_fit.predict(start=len(history), end=len(history)+1, dynamic=False)
        yhat = output[0]
        predictions.append(yhat)
        #obs = X.iloc[t]
        history.append(yhat)
        #print('predicted=%f, expected=%f' % (yhat, obs))


    #calculate rmse
    error = sqrt(mean_squared_error(test, predictions))
    print(f'file: {file}')
    print('Test RMSE: %.3f' % error)
    rmse_plot.append(error)
    
    #plot results
    #plt.rcParams.update({'figure.figsize': (10,10)})
    plt.plot_date(df['timestamp'][size:],df['value'][size:])
    plt.plot_date(df['timestamp'][size:],predictions,color="red",fmt="-")
    plt.gcf().autofmt_xdate()
    plt.savefig("./ARoutput/" + fname +"ARLag30")
    plt.show()
    
    
    predictdf = pd.concat([df['timestamp'][:len(df)-1],pd.Series(predictions)], axis=1)
    predictdf.rename(columns={0:'value'},inplace=True)
    predictdf['value'] = predictdf['value'].shift(-2)
    predictdf = predictdf.dropna()
    predictdf = predictdf.set_index('timestamp')
    
    #combine with df_indexed with predictions to pick out the outliers.
    matrix = df_indexed.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')
    
    #manual threshold set
    threshold = 1200
    
    #compare the real df value with the AR model and then 
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
print(f"Average F1 score over {index} runs is = {sum(f1_plot)/len(f1_plot)}")


# In[25]:


pd.Series(predictions)


# In[ ]:





# # Fixing Performance with standardization

# In[3]:


start_time = datetime.now() 
f1_plot = []
rmse_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 30:
         break
    
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
    df_indexed = df.set_index('timestamp')
    #print(df_indexed)
        
    # prepare data for standardization
    values = df_indexed.copy()
    values = values.drop(columns=['is_anomaly'],axis=1)
    #values = values.reshape((len(values), 1))
    
    # train the standardization
    scaler2 = StandardScaler()
    scaler2 = scaler2.fit(values)
    print('Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
    # standardize the dataset
    standardaized = scaler2.transform(values)
    
    df['std_value'] = standardaized
    df_indexed['std_value'] = standardaized
    
     # split dataset
    X = df.std_value
    size = int(len(X)*0.5)
    train, test = X[0:size], X[size:len(X)]
    
    history = [x for x in train]
    predictions = list()
        
    for t in range(len(test)):
        # train autoregression
        model = AutoReg(history, lags=30)
        model_fit = model.fit()
        # make predictions
        output = model_fit.predict(start=len(history), end=len(history)+1, dynamic=False)
        yhat = output[0]
        predictions.append(yhat)
        obs = X.iloc[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))


    #calculate rmse
    error = sqrt(mean_squared_error(test, predictions))
    print(f'file: {file}')
    print('Test RMSE: %.3f' % error)
    rmse_plot.append(error)
    
    #plot results
    #plt.rcParams.update({'figure.figsize': (10,10)})
    plt.plot_date(df['timestamp'][size:],df['std_value'][size:])
    plt.plot_date(df['timestamp'][size:],predictions,color="red",fmt="-")
    plt.gcf().autofmt_xdate()
    plt.savefig("./ARoutput/" + fname +"ARLag30")
    plt.show()
    
    
    predictdf = pd.concat([df['timestamp'][:len(df)-1],pd.Series(predictions)], axis=1)
    predictdf.rename(columns={0:'value'},inplace=True)
    predictdf['value'] = predictdf['value'].shift(-2)
    predictdf = predictdf.dropna()
    predictdf = predictdf.set_index('timestamp')
    
    #combine with df_indexed with predictions to pick out the outliers.
    matrix = df.join(predictdf, on='timestamp',how='inner',lsuffix='_data',rsuffix='_predict')
    
    #manual threshold set
    threshold = 1.2
    
    #compare the real df value with the AR model and then 
    outliers = matrix[matrix.std_value > matrix.value_predict + threshold]
    #not_outliers = matrix[matrix.value_data <= matrix.value_predict + threshold]
    not_outliers = matrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'] #checkingmatrix[checkingmatrix.value_data <= checkingmatrix.value_prediction + manual_threshold]
    not_outliers = not_outliers.drop_duplicates()
    
    
    print("threshold: ", threshold)
    print("Outliers:")
    print(outliers)
    
#     p = len(series[series.is_anomaly == 1])
#     n = len(series[series.is_anomaly == 0])

    checkingmatrix = df.set_index('timestamp').join(outliers.set_index('timestamp'),on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
    truepositives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 1]
    falsepostives = checkingmatrix[checkingmatrix['is_anomaly_test'] == 0]
    
    checkingmatrix2 =df.set_index('timestamp').join(not_outliers.set_index('timestamp'),on='timestamp',how='inner',lsuffix='_test',rsuffix='_predict')
    truenegatives=checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 0]
    falsenegatives = checkingmatrix2[checkingmatrix2['is_anomaly_test'] == 1]
    
    if(len(truepositives) + len(falsepostives) > 0):
        precision = len(truepositives)/(len(truepositives) + len(falsepostives))
        print("Precision: ", precision)
        precision_plot.append(precision)
    else:
        print("truepositives + falsepositives = 0")
        continue
    if(len(truepositives) + len(falsenegatives) > 0):
        recall = len(truepositives)/(len(truepositives) + len(falsenegatives))
        print("Recall:", recall)
        recall_plot.append(recall)
    else:
        print("truepositives + falsenegatives = 0")
        continue   
    
    if(precision + recall > 0):
        f1 = 2*(precision * recall)/(precision + recall)
        print("F1:",f1)
        f1_plot.append(f1)
    else:
        print("precision + recall = 0")
        f1_plot.append(0)
        continue
    print("--------------------")

    
end_time=datetime.now()
print(f"AR Modelling and Anomaly Analysis of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)
print(f"Average F1 score over {index} runs is = {sum(f1_plot)/len(f1_plot)}")


# In[39]:


df = pd.read_csv(all_csv[0])
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
df_indexed = df.set_index('timestamp')


# In[44]:


X = df_indexed.value
print("\n\nX\n",X)
size = int(len(X) * 0.5)
train, test = X[0:size], X[size:len(X)]
print("\ntrain\n",train,"\ntest\n",test)
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    # train autoregression
    model = AutoReg(history, lags=30)
    model_fit = model.fit()
    # make predictions
    output = model_fit.predict(start=len(history), end=len(history)+1, dynamic=False)
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


# In[ ]:




