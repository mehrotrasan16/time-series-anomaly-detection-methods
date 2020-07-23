#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Loading-and-Reading-the-data-files" data-toc-modified-id="Loading-and-Reading-the-data-files-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading and Reading the data files</a></span></li><li><span><a href="#For-other-time-series" data-toc-modified-id="For-other-time-series-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>For other time series</a></span><ul class="toc-item"><li><span><a href="#A2-folder---Synthetic-Time-series-data-with-inserted-outliers" data-toc-modified-id="A2-folder---Synthetic-Time-series-data-with-inserted-outliers-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>A2 folder - Synthetic Time-series data with inserted outliers</a></span></li><li><span><a href="#A1-Folder---real-Yahoo-server-traffic-dataset" data-toc-modified-id="A1-Folder---real-Yahoo-server-traffic-dataset-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>A1 Folder - real Yahoo server traffic dataset</a></span></li><li><span><a href="#A3-Folder---Synthetic-data-with-outliers-specified" data-toc-modified-id="A3-Folder---Synthetic-data-with-outliers-specified-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>A3 Folder - Synthetic data with outliers specified</a></span></li><li><span><a href="#A4-Folder---Synthetic-data-with-outliers,-changepoints-and-seasonalities." data-toc-modified-id="A4-Folder---Synthetic-data-with-outliers,-changepoints-and-seasonalities.-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>A4 Folder - Synthetic data with outliers, changepoints and seasonalities.</a></span></li></ul></li><li><span><a href="#Extracting-useful-features-from-Time-Series" data-toc-modified-id="Extracting-useful-features-from-Time-Series-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Extracting useful features from Time-Series</a></span></li></ul></div>

# # Imports

# In[5]:


import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import glob
from datetime import datetime


# # Loading and Reading the data files

# In[6]:


start_time = datetime.now()
all_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A2Benchmark/**/*.csv', recursive=True)
end_time = datetime.now()
print(f'Loaded the paths of {len(all_csv)} files from disk. Took {end_time-start_time}')


# In[7]:


all_csv[0]


# In[8]:


all_csv[0].split("/")[5].replace('\\','').split(".")[0]


# In[9]:


df= pd.read_csv(all_csv[0])
df


# In[10]:


df.describe()


# In[11]:


df.info()


# Now we change the timestamp colump from unix seconds after epoch to a human readable date-time stamp   

# In[12]:


df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')#format='%f' if formatting required upto nanoseconds


# In[13]:


df


# In[14]:


df_indexed = df.set_index('timestamp')

print(df_indexed.info())


# In[15]:


# Additive Decomposition
result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')


# In[16]:


# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()


# In[17]:


threshold=500
residualdf = result_add.resid
outliers = residualdf[residualdf > threshold]
print(f"File: {all_csv[0]}")
print("threshold: 500")
print("Outliers:")
print(outliers)


# Note that the presence of negative and zero values make it such that multiplicative decomposition is not applicable for these time series.

# # For other time series

# Now Let's try doing this for all the other 99 time series in a loop

# ## A2 folder - Synthetic Time-series data with inserted outliers

# In[14]:


start_time = datetime.now() 
f1_plot = []
for index,file in enumerate(all_csv):
    if index%10 == 0:
        print(f'Processing index: {index} of {len(all_csv)}')
    if index >15:
         break
    fname = file.split("/")[5].replace('\\','').split(".")[0]
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df_indexed = df.set_index('timestamp')
    result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig("./STLoutput/A2Benchmark_" + fname +"add_STL")
    print("\n\n\nA2Benchmark_" + fname +"add_STL")
    plt.show()
    threshold=500
    residualdf = result_add.resid
    outliers = residualdf[residualdf > threshold]
    p = df_indexed.loc[df_indexed['is_anomaly'] == 1]
    
    n = df_indexed.loc[df_indexed['is_anomaly'] == 0]
    
    truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]
    
    falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]
    
    truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]
    
    falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]
    
    #Traditional FPR and TPR formmulae
    #tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
    #fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())
    
    #IDEAL Paper based TRP/FPR rates
    fpr = len(falsepositives)/len(n)
    tpr = len(truepositives)/len(p)
    fnr = 1-tpr
    tnr = 1-fpr
    
    precision = len(truepositives)/(tpr + fpr)
    recall = tpr/(tpr + fnr)
    
    f1 = 2 * ((precision * recall)/(precision + recall))
    f1_plot.append(f1)
    
    with open("./STLoutput/"+ fname + ".txt", 'w') as file:
        file.write(f"\n\nFile: {fname}")
        file.write("\nthreshold: 500")
        file.write("\nOutliers:\n")
        file.write(outliers.to_csv())
        
        
    print(f"\n\nFile: {fname}")
    print("threshold: 500")
    print("Outliers:")
    print(outliers)
end_time=datetime.now()
print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")
plt.plot(f1_plot)

# In[15]:


print(f"STL Additive decomposition of Yahoo S5 A2 Benchmark processing complete. Time taken:{end_time-start_time}")


# ## A1 Folder - real Yahoo server traffic dataset

# In[16]:


a1_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/**/*.csv', recursive=True)
# start_time = datetime.now() 
# print(f'Loaded the paths of {len(a1_csv)} files from disk. Begin processing at: {start_time}')
# for index,file in enumerate(a1_csv):
#     if index%10 == 0:
#         print(f'Processing index: {index} of {len(a1_csv)}')
#     if index > 50:
#           break
#     fname = file.split("/")[5].replace('\\','').split(".")[0]
#     df = pd.read_csv(file)
#     df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
#     df_indexed = df.set_index('timestamp')
#     result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
#     # Plot
#     plt.rcParams.update({'figure.figsize': (10,10)})
#     result_add.plot().suptitle('Additive Decompose', fontsize=22)
#     plt.savefig("./STLoutput/A1Benchmark_" + fname +"_add_STL")
#     print("\n\n\nA1Benchmark_" + fname +"_add_STL")
#     plt.show()
#     threshold=500
#     residualdf = result_add.resid
#     outliers = residualdf[residualdf > threshold]
#     with open("./STLoutput/"+ fname + ".txt", 'w') as file:
#         file.write(f"\n\nFile: {fname}")
#         file.write("\nthreshold: 500")
#         file.write("\nOutliers:\n")
#         file.write(outliers.to_csv())
        
        
#     print(f"\n\nFile: {fname}")
#     print("threshold: 500")
#     print("Outliers:")
#     print(outliers)
# end_time=datetime.now()
# print(f"STL Additive decomposition of Yahoo S5 A1 Benchmark processing complete. Time taken:{end_time-start_time}")


# ## A3 Folder - Synthetic data with outliers specified

# In[17]:


a3_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A3Benchmark/A3Benchmark-TS*.csv', recursive=True)
# start_time = datetime.now() 
# for index,file in enumerate(a3_csv):
#     if index%10 == 0:
#         print(f'Processing index: {index} of {len(a3_csv)}')
#     if index > 50:
#          break
#     fname = file.split("/")[5].replace('\\','').split(".")[0]
#     df = pd.read_csv(file)
#     df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
#     df_indexed = df.set_index('timestamp')
#     result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
#     # Plot
#     plt.rcParams.update({'figure.figsize': (10,10)})
#     result_add.plot().suptitle('Additive Decompose', fontsize=22)
#     plt.savefig("./STLoutput/A3Benchmark_" + fname +"_add_STL")
#     print("\n\n\nA3Benchmark_" + fname +"_add_STL")
#     plt.show()
#     threshold=500
#     residualdf = result_add.resid
#     outliers = residualdf[residualdf > threshold]
#     with open("./STLoutput/"+ fname + ".txt", 'w') as file:
#         file.write(f"\n\nFile: {fname}")
#         file.write("\nthreshold: 500")
#         file.write("\nOutliers:\n")
#         file.write(outliers.to_csv())
        
        
#     print(f"\n\nFile: {fname}")
#     print("threshold: 500")
#     print("Outliers:")
#     print(outliers)
# end_time=datetime.now()
# print(f"STL Additive decomposition of Yahoo S5 A3 Benchmark processing complete. Time taken:{end_time-start_time}")


# ## A4 Folder - Synthetic data with outliers, changepoints and seasonalities.

# In[18]:


a4_csv = glob.glob(f'./data/yahoo/dataset/ydata-labeled-time-series-anomalies-v1_0/A3Benchmark/A4Benchmark-TS*.csv', recursive=True)
# start_time = datetime.now() 
# for index,file in enumerate(all_csv):
#     if index%10 == 0:
#         print(f'Processing index: {index} of {len(all_csv)}')
#     if index > 50:
#          break
#     fname = file.split("/")[5].replace('\\','').split(".")[0]
#     df = pd.read_csv(file)
#     df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
#     df_indexed = df.set_index('timestamp')
#     result_add = seasonal_decompose(df_indexed['value'], model='additive', extrapolate_trend='freq')
#     # Plot
#     plt.rcParams.update({'figure.figsize': (10,10)})
#     result_add.plot().suptitle('Additive Decompose', fontsize=22)
#     plt.savefig("./STLoutput/A4Benchmark_" + fname +"_add_STL")
#     print("\n\n\nA4Benchmark_" + fname +"_add_STL")
#     plt.show()
#     threshold=500
#     residualdf = result_add.resid
#     outliers = residualdf[residualdf > threshold]
#     with open("./STLoutput/"+ fname + ".txt", 'w') as file:
#         file.write(f"\n\nFile: {fname}")
#         file.write("\nthreshold: 500")
#         file.write("\nOutliers:\n")
#         file.write(outliers.to_csv())
        
        
#     print(f"\n\nFile: {fname}")
#     print("threshold: 500")
#     print("Outliers:")
#     print(outliers)
# end_time=datetime.now()
# print(f"STL Additive decomposition of Yahoo S5 A4 Benchmark processing complete. Time taken:{end_time-start_time}")


# In[ ]:





# # Extracting useful features from Time-Series

# it is possible to use and derive various features from time-series. we can use these features to compare time series and post-comparison: we can detect any anomalous records/subsequences.

# ![Image of Features](./img/features.png)

# References 78 and 79 deal with this: 78 offers a comparison between time series using a combination of PCA on the abovementioned features and then uses a multidimensional outlier detection 

# to be able to incorporate these features, we must first extract them. acc to the authors of [78], their code included on open source R package on CRAM to automatically extract all the abovementioned features. We can use this library in python via the rpy2 package 

# In[3]:


import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter


# In[30]:


import rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

def extract_features(timeseries):
    try:
        oddstream=importr('oddstream')
    except:
        ro.r(f'install.packages("oddstream")')
        oddstream = importr('oddstream')

    #r_timeseries = pandas2ri.py2ri(timeseries)
    with localconverter(ro.default_converter + pandas2ri.converter):
        for col in timeseries.columns.values:
            timeseries[col]=timeseries[col].astype(str) 
        #r_timeseries = ro.conversion.py2rpy(timeseries)
        features=oddstream.extract_tsfeatures(timeseries)
        #features= ro.conversion.rpy2py(features)
        return features
    return []


# In[31]:


df_indexed


# In[38]:


features = extract_features(df_indexed)
print(features)


# This matrix contains all the needed values except Season, Peak, Trough, and Trend. We can try to get these 

# In[34]:


type(features)


# In[35]:


pandas2ri.ri2py(features)


# In[36]:


type(pandas2ri.ri2py(features))


# In[39]:


features


# In[ ]:

outliers = result_add.resid[result_add.resid > threshold]
not_outliers = result_add.resid[result_add.resid < threshold]

anomalies = df_indexed.join(outliers,on='timestamp',how='inner')

not_anomalies = df_indexed.join(not_outliers,on='timestamp',how='inner')

# In[]:

p = df_indexed.loc[df_indexed['is_anomaly'] == 1]

n = df_indexed.loc[df_indexed['is_anomaly'] == 0]

truepositives = anomalies.loc[anomalies['is_anomaly'] == 1]

falsepositives = anomalies.loc[anomalies['is_anomaly'] == 0]

truenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 0]

falsenegatives = not_anomalies.loc[not_anomalies['is_anomaly'] == 1]

#Traditional FPR and TPR formmulae
#tpr = truepositives.count()/(truepositives.count() + falsenegatives.count())
#fpr = falsepositives.count()/(falsepositives.count() + truenegatives.count())

#IDEAL Paper based TRP/FPR rates
fpr = len(falsepositives)/len(n)
tpr = len(truepositives)/len(p)
fnr = 1-tpr
tnr = 1-fpr

precision = len(truepositives)/(tpr + fpr)
recall = tpr/(tpr + fnr)

f1 = 2 * ((precision * recall)/(precision + recall))


# In[]:
class TSFeatures():
    
     def __repr__(self):
        return f"TSFeatures object with properties:{self.mean}, {self.variance}"

ts1 = TSFeatures()
tsfeatures1  = pandas2ri.ri2py(features) #  array of two rows each representing columns value and is_anomaly features.
print(tsfeatures1[0]) #features for value column
ts1.mean = tsfeatures1[0,0]
ts1.variance = tsfeatures1[0,1]

# In[]:
#Not sure if avobe class is better or enum is better. backing up code here  
    