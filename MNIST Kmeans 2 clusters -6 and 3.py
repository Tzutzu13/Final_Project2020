# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:46:25 2020

@author: yaniv
"""
import sys
import sklearn
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train[(y_train == 3) | (y_train== 6)]
y = y_train[(y_train == 3) | (y_train== 6)]
y[y==3] = 0
y[y==6] = 1

X = x.reshape(len(x),-1)

plt.matshow(x[0])
X = X.astype(float) / 255.

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters = 2)
kmeans.fit(X)
label  =kmeans.labels_
print('accuracy = ', accuracy_score(y, label))
