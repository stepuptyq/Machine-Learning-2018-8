# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:12:19 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv', header = None)
X = df.values
df2 = pd.read_csv('target.csv', header = None)
y = df2.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


pr1 = pd.read_csv('ans/30_G2_5.csv', header = None)
pred1 = pr1.values
pr2 = pd.read_csv('ans/30_G2_10.csv', header = None)
pred2 = pr2.values
pr3 = pd.read_csv('ans/30_G2_50.csv', header = None)
pred3 = pr3.values
pr4 = pd.read_csv('ans/30_G2_100.csv', header = None)
pred4 = pr4.values

yt1 = pd.read_csv('target/30_G2_5.csv', header = None)
y_test1 = yt1.values
yt2 = pd.read_csv('target/30_G2_10.csv', header = None)
y_test2 = yt2.values
yt3 = pd.read_csv('target/30_G2_50.csv', header = None)
y_test3 = yt3.values
yt4 = pd.read_csv('target/30_G2_100.csv', header = None)
y_test4 = yt4.values

#画出roc曲线
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
fpr, tpr, thresholds = roc_curve(y_test1, pred1)
mean_tpr += interp(mean_fpr, fpr, tpr)			#对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0 								#初始处为0
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, 'k--', lw=1, label='ROC_5 (area = %0.2f)' % (roc_auc))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
fpr, tpr, thresholds = roc_curve(y_test2, pred2)
mean_tpr += interp(mean_fpr, fpr, tpr)			#对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0 								#初始处为0
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, 'k--', lw=1, label='ROC_10 (area = %0.2f)' % (roc_auc))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
fpr, tpr, thresholds = roc_curve(y_test3, pred3)
mean_tpr += interp(mean_fpr, fpr, tpr)			#对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0 								#初始处为0
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, 'k--', lw=1, label='ROC_50 (area = %0.2f)' % (roc_auc))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
fpr, tpr, thresholds = roc_curve(y_test4, pred4)
mean_tpr += interp(mean_fpr, fpr, tpr)			#对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0 								#初始处为0
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, 'k--', lw=1, label='ROC_100 (area = %0.2f)' % (roc_auc))

 
#画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
 
mean_tpr /= 1 					#在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0 						#坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)		#计算平均AUC值
#画平均ROC曲线
#print mean_fpr,len(mean_fpr)
#print mean_tpr
#plt.plot(mean_fpr, mean_tpr, 'k--',
#         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
 
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()