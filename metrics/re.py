#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error

data_True = pd.read_excel('../data/REAL.xlsx')
data_Pred = pd.read_excel('../result/Our.xlsx')

# RE
preds_index = np.argmax(data_Pred < self.args.end) + 1
trues_index = self.args.battery_EOL[self.args.battery_name][0] - self.args.pred_len - self.args.seq_len + 1
print('RE')
print(preds_index - trues_index)