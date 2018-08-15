# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:28:41 2018

@author: lenovo
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv('ans_touch_number_marked.csv', header = None)
y_pred = df.values
df2 = pd.read_csv('test_touch_number.csv', header = None)
y_true = df2.values

print('micro_F1 = ', metrics.f1_score(y_true, y_pred, average='micro'))
print('macro_F1 = ', metrics.f1_score(y_true, y_pred, average='macro'))