import glob
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt


starttime = datetime.now()
print(f"Started at : {datetime.now()}")

colnames = ['ID','time','Rad Flow','Fpv Close','Fpv Open', 'High','Bypass','Bpv Open','Bpv Close','class']
dfnasa = pd.read_csv("../data/nasa/shuttle.trn/shuttle.trn",names=colnames,sep=" ")
#dfnasa

colnames = ['ID','time','Rad Flow','Fpv Close','Fpv Open', 'High','Bypass','Bpv Open','Bpv Close','class']
dftest = pd.read_csv("../data/nasa/shuttle.tst",names=colnames,sep=" ")
#dftest


merged = pd.concat([dfnasa,dftest])
print(f'Merged shuttle.trn and shuttle.tst, size: {merged.size}')

removeclass4 = merged.loc[merged['class'] != 4]
print(f'Removed Class 4 records: resulting size: {removeclass4.size}')

labelled = removeclass4.copy()
labelled['outlier'] = np.where(labelled['class'] == 1, 0, 1)

print(labelled.head())

#We don't need the ID column . I think so.
masked = labelled.loc[:,labelled.columns != 'ID']
print(masked[:4])

print('Begin Standardization and Normalization')
# train the standardization
scaler2 = StandardScaler()
standf = masked.copy()
for col in standf.columns:
    if col not in ["time","class","outlier"]:
        standf[col] = scaler2.fit_transform(np.reshape(masked[col].values,(len(masked[col].values),1)))
print('\n\n\nStandardized Dataset')
print('Standardization Mean: %f, StandardDeviation: %f' % (scaler2.mean_, sqrt(scaler2.var_)))
print(standf.head())

#train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(np.reshape(masked[col].values,(len(masked[col]),1)))        
normdf = standf.copy()
for col in normdf.columns:
    if col not in ["time","class","outlier"]:
        normdf[col] = scaler.fit_transform(np.reshape(masked[col].values,(len(masked[col].values),1)))

print('\n\n\nNormalized Dataset')
print(normdf.head())

# for col in masked.columns:
    # #if col != 'time':
    # print(col)
    # plt.plot(masked[col])
    # plt.gcf().autofmt_xdate()
    # plt.show()
    # print("Standardized Value")
    # plt.plot(standf[col])
    # plt.gcf().autofmt_xdate()
    # plt.show()    
    # print("Normalized Standardized value")
    # plt.plot(normdf[col])
    # plt.gcf().autofmt_xdate()
    # plt.show()    

normdf.index = normdf.time
normdf = normdf.sort_index()

normdf.to_csv("../data/nasa/clean-std-norm-nasa.csv")

print(f"""Operations complete:
        1. Removed ID column
        2. Removed Class 4 data points
        3. Added outlier column with value = 0 if data point from class 1 and 0 if data point from classes 2,3,5,6,7
        4. standardized all columns except the time, class and outlier columns
        5. plotted a graph of all columns to understand the effects of normalization on the value of the features(commented this code for now.)
        6. set the time column as the index of the dataset
        7. written standardized, normalized, labelled dataset to clean-std-norm-nasa.csv
        Time taken : {datetime.now()-starttime}""")
