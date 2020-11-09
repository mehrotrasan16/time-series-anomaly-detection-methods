#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Read-Data" data-toc-modified-id="Read-Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read Data</a></span></li><li><span><a href="#Calculate-Moving-Average" data-toc-modified-id="Calculate-Moving-Average-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Calculate Moving Average</a></span></li><li><span><a href="#Set-threshold-to-detect-outliers" data-toc-modified-id="Set-threshold-to-detect-outliers-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Set threshold to detect outliers</a></span><ul class="toc-item"><li><span><a href="#Threshold-setting-method:-k-std-devs" data-toc-modified-id="Threshold-setting-method:-k-std-devs-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Threshold setting method: k-std devs</a></span><ul class="toc-item"><li><span><a href="#Calculate-F1-score-of-this-result(702)" data-toc-modified-id="Calculate-F1-score-of-this-result(702)-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Calculate F1 score of this result(702)</a></span></li></ul></li><li><span><a href="#Threshold-setting-method:-Manual" data-toc-modified-id="Threshold-setting-method:-Manual-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Threshold setting method: Manual</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#How-to-check-common-rows-between-two-dataframes" data-toc-modified-id="How-to-check-common-rows-between-two-dataframes-3.2.0.1"><span class="toc-item-num">3.2.0.1&nbsp;&nbsp;</span>How to check common rows between two dataframes</a></span></li></ul></li><li><span><a href="#Calculate-F1-score-of-this-result(546)" data-toc-modified-id="Calculate-F1-score-of-this-result(546)-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Calculate F1 score of this result(546)</a></span></li><li><span><a href="#Do-this-in-a-loop" data-toc-modified-id="Do-this-in-a-loop-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Do this in a loop</a></span></li><li><span><a href="#Do-this-in-a-loop-after-Standardizing-the-Data" data-toc-modified-id="Do-this-in-a-loop-after-Standardizing-the-Data-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Do this in a loop after Standardizing the Data</a></span></li></ul></li></ul></li><li><span><a href="#Stationary-transformation-by-differencing" data-toc-modified-id="Stationary-transformation-by-differencing-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Stationary transformation by differencing</a></span></li><li><span><a href="#Exponential-Moving-Average" data-toc-modified-id="Exponential-Moving-Average-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exponential Moving Average</a></span></li><li><span><a href="#Moving-Average-Model-vs-Moving-Average" data-toc-modified-id="Moving-Average-Model-vs-Moving-Average-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Moving Average Model vs Moving Average</a></span></li><li><span><a href="#Training-an-ARMA-model.-In-a-loop.-With-Standardization" data-toc-modified-id="Training-an-ARMA-model.-In-a-loop.-With-Standardization-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Training an ARMA model. In a loop. With Standardization</a></span></li></ul></div>

# # Read Data

# In[2]:


import os
print(os.getcwd())
import pandas as pd
from matplotlib import pyplot as plt
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from math import sqrt


# In[3]:


start_time = datetime.now()
all_csv = glob.glob(f'../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')


# In[3]:


all_csv[0]


# In[4]:


all_csv[0].split("/")[5].replace('\\','').split(".")[0]


# In[5]:


df= pd.read_csv(all_csv[0])
df


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds


# In[9]:


df_indexed = df.set_index('timestamp')

print(df_indexed.info())


# In[10]:


plt.plot_date(df['timestamp'],df['value'])
plt.gcf().autofmt_xdate()


# In[11]:


values = pd.DataFrame(df.value)
corrmat = pd.concat([values.shift(1),values],axis =1)
corrmat.columns = ['t-1','t']
result = df.corr()
print(result)


# In[12]:


df[df['is_anomaly'] > 0]


# # Calculate Moving Average

# In[80]:


# Tail-rolling average transform
rolling = df.rolling(window=10)
rolling_mean = rolling.mean()

# plot original and transformed dataset
plt.plot_date(df['timestamp'],rolling_mean.value,color="red",fmt="-")
plt.gcf().autofmt_xdate()
#rolling_mean.plot(color='red')
plt.show()


# In[14]:


plt.plot_date(df['timestamp'],df['value'])
plt.plot_date(df['timestamp'],rolling_mean.value,color="red",fmt="-")
plt.gcf().autofmt_xdate()


# # Set threshold to detect outliers

# In[15]:


plt.plot_date(df['timestamp'],df['value'])
plt.plot_date(df['timestamp'],rolling_mean.value,color="red",fmt="-")
plt.gcf().autofmt_xdate()


# In[ ]:





# ## Threshold setting method: k-std devs

# In[16]:


upper_threshold = rolling_mean.value.mean() + rolling_mean.value.std()
lower_threshold = rolling_mean.value.mean() - rolling_mean.value.std()
upper_threshold, lower_threshold


# In[17]:


MAprediction = pd.concat([df['timestamp'],rolling_mean.value],axis=1)
MAprediction.set_index('timestamp')
MAprediction


# In[18]:


df_indexed.join(MAprediction.set_index('timestamp'),on='timestamp',how='inner',lsuffix='_data',rsuffix='_MA')


# In[19]:


outliers = rolling_mean[rolling_mean.value > upper_threshold]
outliers = outliers.append(rolling_mean[rolling_mean.value < lower_threshold])
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)


# ### Calculate F1 score of this result(702)

# $$ F_1 = 2 * \frac{precision + recall}{precision * recall} $$

# $$precision = \frac{true\ positive}{true\ positive + false\ positive}$$

# $$recall = \frac{true\ positive}{true\ positive + false\ negative}$$

# ## Threshold setting method: Manual

# In[34]:


MAprediction.info()


# In[36]:


checkingmatrix = df_indexed.join(MAprediction.set_index('timestamp'), on='timestamp',how='inner',lsuffix='_data',rsuffix='_prediction')
checkingmatrix


# In[45]:


manual_threshold = 1200
outliers = checkingmatrix[checkingmatrix.value_data >= checkingmatrix.value_prediction + manual_threshold]
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)


# In[46]:


#outliers = outliers.append(rolling_mean[rolling_mean.value < -1*(manual_threshold)])


# In[47]:


outliers = outliers.drop_duplicates()
outliers


# #### How to check common rows between two dataframes

# In[58]:


checkingmatrix.merge(outliers, how = 'inner' ,indicator=False)


# ###### How to check rows in checkingmatrix that are not in outliers

# In[63]:


checkingmatrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']


# ###### How to check rows in outliers that are not in checking matrix

# In[60]:


checkingmatrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='right_only']


# In[ ]:


not_outliers = checkingmatrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'] #checkingmatrix[checkingmatrix.value_data <= checkingmatrix.value_prediction + manual_threshold]


# In[ ]:


#not_outliers = not_outliers.append(rolling_mean[rolling_mean.value > -1*(manual_threshold)])


# In[64]:


not_outliers = not_outliers.drop_duplicates()


# In[68]:


not_outliers


# ### Calculate F1 score of this result(546)

# $$ F_1 = 2 * \frac{precision + recall}{precision * recall} $$

# $$precision = \frac{true\ positive}{true\ positive + false\ positive}$$

# $$recall = \frac{true\ positive}{true\ positive + false\ negative}$$

# In[30]:


p = len(df[df.is_anomaly == 1])
p


# In[31]:


n = len(df[df.is_anomaly == 0])
n


# In[69]:


checkingmatrix, outliers, not_outliers


# In[74]:


#join with original df
#check is_anomaly flag - calc truepos,true neg,fslse pos , false neg
#calc f1
#do oit in for loop
truepositives = outliers[outliers.is_anomaly == 1]
falsepostives = outliers[outliers.is_anomaly == 0]
truenegatives = not_outliers[not_outliers.is_anomaly == 0]
falsenegatives = not_outliers[not_outliers.is_anomaly == 1]


# In[78]:


if(len(truepositives) + len(falsepostives) > 0):
    precision = len(truepositives)/(len(truepositives) + len(falsepostives))
    print("Precision: ", precision)
    #precision_plot.append(precision)
else:
    print("truepositives + falsepositives = 0")
    #continue


recall = len(truepositives)/(len(truepositives) + len(falsenegatives))
print("Recall:", recall)
#recall_plot.append(recall)
if(precision + recall > 0):
    f1 = 2*(precision * recall)/(precision + recall)
    print("F1:",f1)
    #f1_plot.append(f1)
else:
    print("precision + recall = 0")
    #continue


# ### Do this in a loop

# In[85]:


start_time = datetime.now() 
f1_plot = []
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
    
    # Tail-rolling average transform
    rolling = df.rolling(window=10)
    rolling_mean = rolling.mean()
        
    # plot original and transformed dataset
    plt.plot_date(df['timestamp'],df['value'])
    plt.plot_date(df['timestamp'],rolling_mean.value,color="red",fmt="-")
    plt.gcf().autofmt_xdate()
    plt.savefig("./MAoutput/" + fname +"MAWindow10")
    plt.show()
    
    MAprediction = pd.concat([df['timestamp'],rolling_mean.value],axis=1)
    MAprediction.set_index('timestamp')
    checkingmatrix = df_indexed.join(MAprediction.set_index('timestamp'), on='timestamp',how='inner',lsuffix='_data',rsuffix='_prediction')
    
    manual_threshold = 1200
    outliers = checkingmatrix[checkingmatrix.value_data >= checkingmatrix.value_prediction + manual_threshold]
    outliers = outliers.drop_duplicates()
    print(f"File: {fname}")
    print(f"threshold:{manual_threshold}")
    print("Outliers:")
    print(outliers)

    not_outliers = checkingmatrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'] #checkingmatrix[checkingmatrix.value_data <= checkingmatrix.value_prediction + manual_threshold]
    #not_outliers = not_outliers.append(rolling_mean[rolling_mean.value > -1*(manual_threshold)])
    not_outliers = not_outliers.drop_duplicates()
    
    truepositives = outliers[outliers.is_anomaly == 1]
    falsepostives = outliers[outliers.is_anomaly == 0]
    truenegatives = not_outliers[not_outliers.is_anomaly == 0]
    falsenegatives = not_outliers[not_outliers.is_anomaly == 1]
    
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
print(f"MA Modelling and Anomaly Analysis of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)


# In[86]:


print(sum(f1_plot)/len(f1_plot))


# ### Do this in a loop after Standardizing the Data

# In[87]:


from sklearn.preprocessing import StandardScaler
from math import sqrt


# In[94]:


start_time = datetime.now() 
f1_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
#     if index > 30:
#          break
    
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
    
    #jugaad, change so that standardized is used from here on.
    for i in range(len(df)):
        df['value'].iloc[i] = standardaized[i]
    
    # Tail-rolling average transform
    rolling = df.rolling(window=10)
    rolling_mean = rolling.mean()
        
    # plot original and transformed dataset
    plt.plot_date(df['timestamp'],df['value'])
    plt.plot_date(df['timestamp'],rolling_mean.value,color="red",fmt="-")
    plt.gcf().autofmt_xdate()
    plt.savefig("./MAoutput/" + fname +"MAWindow10")
    plt.show()
    
    MAprediction = pd.concat([df['timestamp'],rolling_mean.value],axis=1)
    MAprediction.set_index('timestamp')
    checkingmatrix = df.set_index('timestamp').join(MAprediction.set_index('timestamp'), on='timestamp',how='inner',lsuffix='_data',rsuffix='_prediction')
    
    manual_threshold = 1.2#1200
    outliers = checkingmatrix[checkingmatrix.value_data >= checkingmatrix.value_prediction + manual_threshold]
    outliers = outliers.drop_duplicates()
    print(f"File: {fname}")
    print(f"threshold:{manual_threshold}")
    print("Outliers:")
    print(outliers)

    not_outliers = checkingmatrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'] #checkingmatrix[checkingmatrix.value_data <= checkingmatrix.value_prediction + manual_threshold]
    #not_outliers = not_outliers.append(rolling_mean[rolling_mean.value > -1*(manual_threshold)])
    not_outliers = not_outliers.drop_duplicates()
    
    truepositives = outliers[outliers.is_anomaly == 1]
    falsepostives = outliers[outliers.is_anomaly == 0]
    truenegatives = not_outliers[not_outliers.is_anomaly == 0]
    falsenegatives = not_outliers[not_outliers.is_anomaly == 1]
    
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
        f1_plot.append(0)
        continue
    print("--------------------")

    
end_time=datetime.now()
print(f"MA Modelling and Anomaly Analysis of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)


# In[95]:


print(sum(f1_plot)/len(f1_plot))


# # Stationary transformation by differencing

# this is to make sure that we actually only catch the outliers instead of every value above or below the threshold

# Now that you look at it, it's startlingly familiar,that's cause this is subtracting the TS from its seasonal component taking a timestep of the interval argument, here 1. This is another form of STL Additive DEcomposition done in my STL DEcomposition Notebook.

# In[ ]:


#This function removes the trend and seasonsality
#the interval argument could be set to 7 to remove weekly seasonality, 360 for yearly and so on.
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.DataFrame(diff,columns=['value'])


# In[ ]:


diffseries = difference(df.value)
print(diffseries)
plt.plot_date(df['timestamp'][:len(df)-1],diffseries) 
plt.gcf().autofmt_xdate()


# In[ ]:


diffseries.iloc[650:658]


# In[ ]:


diffseries.iloc[890:898]


# In[ ]:


diffseries[diffseries.value > manual_threshold]


# In[ ]:


diffseries[diffseries.value < (-1)*manual_threshold]


# In[ ]:


out = diffseries[diffseries.value > manual_threshold]


# In[ ]:


out = out.append(diffseries[diffseries.value < (-1)*manual_threshold])


# In[ ]:


out


# In[ ]:


# Tail-rolling average transform
rolling2 = diffseries.rolling(window=30)
rolling_mean2 = rolling.mean()

# plot original and transformed dataset
plt.plot_date(df['timestamp'],rolling_mean2.value,color="red",fmt="-")
#rolling_mean.plot(color='red')
plt.show()


# In[ ]:


rolling2


# In[ ]:


plt.plot_date(df['timestamp'],df['value'])
plt.plot_date(df['timestamp'],rolling_mean2.value,color="red",fmt="-")
plt.gcf().autofmt_xdate()


# # Exponential Moving Average

# In[ ]:


# import pandas as pd
# import matplotlib.pyplot as plt

series = pd.read_csv('D://Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark\\synthetic_1.csv')

window_size = 3
mean = series['value'].ewm(window_size).mean()
std = series['value'].ewm(window_size).std()
std[0] = 0 #the first value turns into NaN because of no data

mean_plus_std = mean + std
mean_minus_std = mean - std

is_outlier = (series['value'] > mean_plus_std) | (series['value'] < mean_minus_std)
outliers = series[is_outlier]

plt.plot(series['value'], c = 'b', label = 'Actual Values')
plt.plot(mean, c = 'r', label = 'Exponentially Weighted Moving Average')
plt.plot(mean_plus_std, 'k--', label = 'Prediction Bounds')
plt.plot(mean_minus_std, 'k--')
plt.scatter(outliers.index, outliers['value'], c = 'r', marker = 'o', s = 120, label = 'Outliers')
plt.gcf().set_size_inches(20,20)
plt.legend()


# In[ ]:


outliers


# In[ ]:


outliers[outliers.is_anomaly > 0]


# In[ ]:





# #  Moving Average Model vs Moving Average

# A moving average model is different from calculating the moving average of the time series. It can be calucalted using the ARMA class from scikitlearn.

# The notation for the model involves specifying the order of the model q as a parameter to the MA function, e.g. MA(q). For example, MA(1) is a first-order moving average model.
# 
# The method is suitable for univariate time series without trend and seasonal components.

# In[34]:


from statsmodels.tsa.arima_model import ARMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARMA(data, order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data),dynamic=False)
print(yhat)


# # Training an ARMA model. In a loop. With Standardization

# In[40]:


history


# In[46]:


help(model_fit.predict)
model_fit.predict(len(history), len(history),dynamic=False)


# In[36]:


start_time = datetime.now() 
f1_plot = []
rmse_plot = []
precision_plot = []
recall_plot = []
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 10:
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
    size = int(len(X)*0.5)
    train, test = X[0:size], X[0:len(X)]
    
        
    history = [x for x in train]
    predictions = list()
    
    # train and fit MA model
    for t in range(len(test)):
        # train autoregression
        model = ARMA(train, order=(0, 1))
        model_fit = model.fit(disp=False)
        # make predictions
        output = model_fit.predict(start=len(history), end=len(history)+1, dynamic=False)
        yhat = output.iloc[0]
        predictions.append(yhat)
        obs = X.iloc[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    
        #calculate rmse
    error = sqrt(mean_squared_error(test, predictions))
    print(f'file: {file}')
    print('Test RMSE: %.3f' % error)
    rmse_plot.append(error)

    # plot original and transformed dataset
    plt.plot_date(df['timestamp'],df['std_value'])
    plt.plot_date(df['timestamp'],yhat,color="red",fmt="-")
    plt.gcf().autofmt_xdate()
    plt.savefig("./MAoutput/" + fname +"MAWindow10")
    plt.show()
    
    MAprediction = pd.concat([df['timestamp'],yhat],axis=1)
    MAprediction.rename(columns={0:'value'},inplace=True)
    MAprediction.set_index('timestamp')
    checkingmatrix = df.set_index('timestamp').join(MAprediction.set_index('timestamp'), on='timestamp',how='inner',lsuffix='_data',rsuffix='_prediction')
    
    manual_threshold = 1.2#1200
    outliers = checkingmatrix[checkingmatrix.std_value >= checkingmatrix.value_prediction + manual_threshold]
    outliers = outliers.drop_duplicates()
    print(f"File: {fname}")
    print(f"threshold:{manual_threshold}")
    print("Outliers:")
    print(outliers)
    
    #Line of code to exclude outlier rows and put the rest of the rows into not_outliers
    #Here indicator adds an extra col '_merge' with the value of left_only for rows that are in checking matrix 
    #and right_only for rows only in outliers. and 'both' for rows in both. 
    not_outliers = checkingmatrix.merge(outliers, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'] #checkingmatrix[checkingmatrix.value_data <= checkingmatrix.value_prediction + manual_threshold]
    #not_outliers = not_outliers.append(rolling_mean[rolling_mean.value > -1*(manual_threshold)])
    not_outliers = not_outliers.drop_duplicates()
    
    truepositives = outliers[outliers.is_anomaly == 1]
    falsepostives = outliers[outliers.is_anomaly == 0]
    truenegatives = not_outliers[not_outliers.is_anomaly == 0]
    falsenegatives = not_outliers[not_outliers.is_anomaly == 1]
    
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
print(f"MA Modelling and Anomaly Analysis of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)
print(f"Average F1 score over {index} runs is = {sum(f1_plot)/len(f1_plot)}")


# In[25]:


type(output), output


# In[7]:


X = df_indexed.value
print("\n\nX\n",X)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
print("\ntrain\n",train,"\ntest\n",test)
history = [x for x in train]
predictions = list()
for t in range(len(test)):
 	model = ARMA(history, order=(0, 1))
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


# In[ ]:




