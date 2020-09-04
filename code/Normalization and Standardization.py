# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:56:54 2020

@author: SanketM
"""


# Standardize and Normalize time series data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt


##NORMALIZATION

# load the dataset and print the first 5 rows
series = pd.read_csv("D:/Temp/time-series/data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/synthetic_1.csv")
series.head()

# Set the datetime as index and convert from unix time to human readable datetime
series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')
series = series.set_index('timestamp')

# pick out values for normalization
vals = series.copy()#pd.DataFrame(series['value'])
vals = vals.drop(columns=['is_anomaly'],axis=1)

print(vals)

#Normalize - train
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(vals)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

#Test transform and inverse transform.
print(vals.value.head())

normalized = scaler.transform(vals)
for i in range(5):
	print(normalized[i])

inversed = scaler.inverse_transform(normalized)
for i in range(5):
	print(inversed[i])

##STANDASRDIZATION

# prepare data for standardization
values = series.copy()
values = values.drop(columns=['is_anomaly'],axis=1)
#values = values.reshape((len(values), 1))

# train the standardization
scaler2 = StandardScaler()
scaler2 = scaler2.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
# standardization the dataset and print the first 5 rows
standardaized = scaler2.transform(values)
for i in range(5):
	print(standardaized[i])

# inverse transform and print the first 5 rows
inversed = scaler.inverse_transform(standardaized)
for i in range(5):
	print(inversed[i])

plt.gcf().set_size_inches(10,10)
plt.subplot(4,1,1)
series.value.hist()
plt.subplot(4,1,2)
plt.plot_date(vals.index,vals.value)
plt.gcf().autofmt_xdate()
plt.subplot(4,1,3)
plt.plot_date(vals.index,normalized)
plt.gcf().autofmt_xdate()
plt.subplot(4,1,4)
plt.plot_date(vals.index,standardaized)
plt.gcf().autofmt_xdate()

###################################

#in a for loop
import glob
from datetime import datetime

start_time = datetime.now()
all_csv = glob.glob(f'../data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')

start_time = datetime.now() 
for index,file in enumerate(all_csv):
    
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index > 3:
        break
    
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    
    ##NORMALIZATION
    
    # load the dataset and print the first 5 rows
    series = pd.read_csv(file)
    series.head()
    
    # Set the datetime as index and convert from unix time to human readable datetime
    series['timestamp'] = pd.to_datetime(series['timestamp'],unit='s')
    series = series.set_index('timestamp')
    
    # pick out values for normalization
    vals = series.copy()#pd.DataFrame(series['value'])
    vals = vals.drop(columns=['is_anomaly'],axis=1)
    
    print(vals)
    
    #Normalize - train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(vals)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    
    #Test transform and inverse transform.
    print(vals.value.head())
    
    normalized = scaler.transform(vals)
    for i in range(5):
    	print(normalized[i])
    
    inversed = scaler.inverse_transform(normalized)
    for i in range(5):
    	print(inversed[i])
    
    ##STANDASRDIZATION
    
    # prepare data for standardization
    values = series.copy()
    values = values.drop(columns=['is_anomaly'],axis=1)
    #values = values.reshape((len(values), 1))
    
    # train the standardization
    scaler2 = StandardScaler()
    scaler2 = scaler2.fit(values)
    print('Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
    # standardization the dataset and print the first 5 rows
    standardaized = scaler2.transform(values)
    for i in range(5):
    	print(standardaized[i])
    
    # inverse transform and print the first 5 rows
    inversed = scaler.inverse_transform(standardaized)
    for i in range(5):
    	print(inversed[i])
    
    plt.gcf().set_size_inches(15,15)
    plt.subplot(4,1,1)
    series.value.hist()
    plt.subplot(4,1,2)
    plt.plot_date(vals.index,vals.value)
    plt.gcf().autofmt_xdate()
    plt.subplot(4,1,3)
    plt.plot_date(vals.index,normalized)
    plt.gcf().autofmt_xdate()
    plt.subplot(4,1,4)
    plt.plot_date(vals.index,standardaized)
    plt.gcf().autofmt_xdate()
    plt.savefig("./NormStanoutput/" + fname +"NormStan")
    plt.clf()
    
end_time = datetime.now()
print(f'Finished processing {len(all_csv)} files. Took {end_time-start_time}')