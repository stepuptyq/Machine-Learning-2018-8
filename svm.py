# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:45:28 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv('rundata/30_G2_15.csv', header = None)
X = df.values
df2 = pd.read_csv('target.csv', header = None)
y = df2.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.svm import SVC
clf = SVC(probability=True, class_weight='balanced')
clf.fit(X, y)


pred = clf.predict_proba(X_test)[:,1]

# print ("AUC Score (Train): %f" % metrics.roc_auc_score(pred, y_test))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_true = y_test, y_score = pred))

df = pd.DataFrame(pred)
df.to_csv('ans/30_G2_15.csv', index = False)
df = pd.DataFrame(y_test)
df.to_csv('target/30_G2_15.csv', index = False)

#画出roc曲线
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
fpr, tpr, thresholds = roc_curve(y_test, pred)
mean_tpr += interp(mean_fpr, fpr, tpr)			#对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0 								#初始处为0
roc_auc = auc(fpr, tpr)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, 'k--', lw=1, label='ROC (area = %0.2f)' % (roc_auc))
 
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