# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:46:28 2020

@author: SanketM
"""


import pandas as pd
import os
import rpy2

 def extract_features(self, timeseries):
        oddstream=importr('oddstream')

        #r_timeseries = pandas2ri.py2ri(timeseries)
        with localconverter(ro.default_converter + pandas2ri.converter):
            for col in timeseries.columns.values:
                timeseries[col]=timeseries[col].astype(str) 
            #r_timeseries = ro.conversion.py2rpy(timeseries)
            features=oddstream.extract_tsfeatures(timeseries)
            #features= ro.conversion.rpy2py(features)
            return features
        return []

