# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:16:44 2020

@author: SanketM
"""


import os
os.chdir("D:\\Temp\\time-series")

import pandas as pd
colnames = ['ID','time','Rad Flow','Fpv Close','Fpv Open', 'High','Bypass','Bpv Open','Bpv Close','class']
df = pd.read_csv("D:\\Temp\\time-series\\data\\nasa\\shuttle.trn\\shuttle.trn",names=colnames,sep=" ")
print(df)

masked = df.loc[:,df.columns != 'class']
print(masked[:4])

###
from statsmodels.tsa.seasonal import seasonal_decompose

#result_add = seasonal_decompose(masked['Rad Flow'],model="additive",extrapolate_trend='freq')

result_add = seasonal_decompose(masked['Rad Flow'],model="multiplicative",extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()