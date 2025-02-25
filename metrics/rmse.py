#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error

data_True = pd.read_excel('../data/REAL.xlsx')
data_Pred = pd.read_excel('../result/Our.xlsx')

# RMSE
print('RMSE')
print(np.sqrt(metrics.mean_squared_error(data_Pred,data_True)))