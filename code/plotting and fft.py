# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:57:35 2020

@author: SanketM
"""


import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
import os
import statsmodels.api as sm
import glob
from datetime import datetime as dt
import numpy as np
from scipy.fftpack import fft

#find our data
start_time = dt.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = dt.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')

#read one file
df= pd.read_csv(all_csv[0])
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds
df.index = df["timestamp"]
print(df)

#get the figure to mess with
#fig = plt.figure(figsize=(50,50))

#USING THE PLOT_DATE FUNCTION - DID NOT EVEN KNOW THAT EXISTED
plt.plot_date(df['timestamp'],df['value'],linestyle='solid')

#USING THE PLT. GET CURRENT FIGURE TO ACCESS THE XAXIS AND THEN USE THE AUTOFMT_XDATE TO ROTATE THE DATES 
plt.gcf().autofmt_xdate()

#USING THE MPL DATES - DATEFORMATTER CLASS TO USE A STRF STRING FORMATTER TO FIX A FORMAT,
date_format = mpl_dates.DateFormatter('%b, %d %Y')

#GET THE CURRENT AXIS(GCA) AND THE SET_MAJOR_FORMATTER TO SET THE FORMAT
axis = plt.gca()
axis.xaxis.set_major_formatter(date_format)

#plt.tight_layout()
plt.show()

#CALCULATE THE FFT OF THE WHOLE SERIES.
x=fft(df['value'].tolist())
print(x)

nrows=1
ncols=1
fig, ax = plt.subplots(nrows,ncols)
ax.plot_date(df['timestamp'],np.abs(x))
ax.set_title("FFT of the A2 Benchmark vanilla synthetic time-series")
ax.set_xlabel("Date")
ax.set_ylabel("fft")
fig.autofmt_xdate()

def get_median_filtered(signal, threshold=3):
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference/ float(median_difference)
    mask = s[0] if s.any() > threshold else threshold
    signal[mask]= np.median(signal)
    return signal

medfilter = get_median_filtered(df['value'].tolist())

fig,ax = plt.subplots(nrows,ncols)
ax.plot_date(df['timestamp'],medfilter)
ax.set_title("median filtered signal")
ax.set_xlabel("Date")
ax.set_ylabel("Median Signal")
fig.autofmt_xdate()

import rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def medianfilter(timeseries):
    try:
        #fractal=importr('fractal')
        ro.r(f'install.packages("fractal")')
        fractal = importr('fractal')
    except:
        ro.r(f'install.packages("fractal")')
        fractal = importr('fractal')

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=fractal.medianFilter(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []

med_filter = medianfilter(df)

fig,ax = plt.subplots(nrows,ncols)
ax.plot_date(df['timestamp'],med_filter)
ax.set_title("median filtered signal by Fractal")
ax.set_xlabel("Date")
ax.set_ylabel("Median Signal")
fig.autofmt_xdate()
