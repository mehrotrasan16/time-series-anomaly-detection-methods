#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span></li></ul></div>

# # Imports

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
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from math import sqrt


# In[2]:


start_time = datetime.now()
all_csv = glob.glob(f'../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')


# In[3]:


df = pd.read_csv(all_csv[0])
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
df_indexed = df.set_index('timestamp')


# In[15]:


X = df_indexed.value
print("\n\nX\n",X)
size = int(len(X) * 0.5)
train, test = X[0:size], X[size:len(X)]
print("\ntrain\n",train,"\ntest\n",test)
history = [x for x in train]
predictions = list()
for t in range(len(test)):
 	model = ARIMA(history, order=(5,1,0))
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
plt.plot(train.tolist())
plt.plot(test.tolist())
plt.plot(predictions, color='red')
plt.show()


# In[17]:


output


# In[7]:


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
    df_indexed['std_value'] = standardaized
    
    # split dataset
    X = df.std_value
    size = int(len(X)*0.5)
    train, test = X[0:size], X[size:len(X)]
    
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test.iloc[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    
    #plot results
    #plt.rcParams.update({'figure.figsize': (10,10)})
    #plt.plot_date(df['timestamp'][size:],df['std_value'][size:])
    plt.plot_date(df['timestamp'][:size],train,fmt="-")
    plt.plot_date(df['timestamp'][:size+1],test,color="orange",fmt="-")
    plt.plot_date(df['timestamp'][size:],predictions,color="red",fmt="-")
    plt.gcf().autofmt_xdate()
    plt.savefig("./ARIMAoutput/" + fname +"arima-5-1-0")
    plt.show()
    
    arimaprediction = pd.concat([df['timestamp'][size:],pd.Series(predictions)],axis=1)
    arimaprediction.rename(columns={0:'value'},inplace=True)
    arimaprediction = arimaprediction.set_index('timestamp')
    checkingmatrix = df.set_index('timestamp').join(arimaprediction, on='timestamp',how='inner',lsuffix='_data',rsuffix='_prediction')
    
end_time=datetime.now()
print(f"ARIMA Modelling and Anomaly Analysis of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)
print(f"Average F1 score over {index} runs is = {sum(f1_plot)/len(f1_plot)}")


# In[10]:


plt.plot_date(df['timestamp'][:size+1],test,color="orange",fmt="-")
plt.gcf().autofmt_xdate()


# In[14]:


len(model_fit.predict())


# In[12]:


test


# In[ ]:




