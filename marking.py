# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:05:00 2018

@author: lenovo
"""

import pandas as pd
import numpy as np

df = pd.read_csv('ans_touch_number.csv', header = None)
pred = df.values.tolist()
pred_ori = [i for i in pred]
pred.sort()

threshold = pred[2442]
print(threshold)

pred_res=[]

for i in pred_ori:
    if i < threshold:
        pred_res.append(0)
    else:
        pred_res.append(1)
        
df = pd.DataFrame(pred_res)
df.to_csv('ans_touch_number_marked.csv', index = False)
