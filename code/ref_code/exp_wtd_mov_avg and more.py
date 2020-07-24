# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:43:48 2020

@author: SanketM
"""


############# Exponential Weighted Moving Average Code ################

import pandas as pd
import matplotlib.pyplot as plt

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
plt.legend()
plt.gcf().set_size_inches(10,10)

################### Median Absolute Deviation and Standard Deviation ##########
###Src:https://anands.github.io/blog/2015/11/26/outlier-detection-using-python/

from __future__ import division
import numpy

# Sample Dataset
x = [10, 9, 13, 14, 15,8, 9, 10, 11, 12, 9, 0, 8, 8, 25,9,11,10]

# Median absolute deviation
def mad(data, axis=None):
    return numpy.mean(numpy.abs(data - numpy.mean(data, axis)), axis)
_mad = numpy.abs(x - numpy.median(x)) / mad(x)

# Standard deviation
_sd = numpy.abs(x - numpy.mean(x)) / numpy.std(x)

print _mad
print _sd

#after this only left to calculate the IQR
'''Median and Interquartile Deviation Method (IQD)
For this outlier detection method, the median of the residuals is calculated, 
along with the 25th percentile and the 75th percentile. 
The difference between the 25th and 75th percentile is the interquartile deviation (IQD). 
Then, the difference is calculated between each historical value and the residual median. 
If the historical value is a certain number of MAD away from the median of the residuals, 
that value is classified as an outlier. 
The default threshold is 2.22, which is equivalent to 3 standard deviations or MADs.
'''

#################### Mahalobinis Distance Code
##Src: same as above

from __future__ import division
import numpy as np

# Base dataset
dataset = np.array(
        [
          [9,9,10,11,12,13,14,15,16,17,18,19,18,17,11,10,8,7,8],
          [8,6,10,13,12,11,12,12,13,14,1,16,20,21,19,18,11,5,5],
        ])

# target: dataset to be compared
target = [0,0,0,0,10,9,15,11,15,17,13,14,18,17,14,22,11,5,5]

# Avg of SD of each dataset
dataset_std = dataset.std()

# Avg of arrays in dataset
dataset_sum_avg = np.array([0] * len(dataset[0])) # Create a empty dataset
for data in dataset:
    dataset_sum_avg = dataset_sum_avg + ( data / len(dataset)) # Add up all datapoints of dataset

# Substract the target dataset with avg of datapoints sum and divide by SD
data = np.abs(target - dataset_sum_avg) / dataset_std

print data