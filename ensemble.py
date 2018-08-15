# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:34:56 2018

@author: lenovo
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('data.csv', header = None)
X = df.values
df2 = pd.read_csv('target.csv', header = None)
y = df2.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

df3 = pd.read_csv('ans_svm.csv', header = None)
pred_svm = df3.values
df4 = pd.read_csv('ans_DT.csv', header = None)
pred_DT = df4.values
df5 = pd.read_csv('ans_log_reg.csv', header = None)
pred_neural = df5.values

new_pred = 0.9 * pred_svm + 0.1 * pred_DT + 0.1 * pred_neural
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_true = y_test, y_score = new_pred))
df = pd.DataFrame(new_pred)
df.to_csv('ans_ensamble.csv', index = False)