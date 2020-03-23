# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 02:37:44 2020

@author: yaniv
"""
import timeit
start = timeit.default_timer()
from keras.datasets import mnist
import numpy as np
from time import time
import keras
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
# import metrics
import keras.backend as K
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(10)
import tensorflow as tf

tf.random.set_seed(7)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_train[0:500] #x,y needs same shape
y = y_train[0:500]

x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
x_test = np.divide(x_test, 255.)
x_test = x_test.reshape((x_test.shape[0], -1))

n_clusters = 10
x.shape

X_train = x.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)

result_x = np.array(X_train)
result_y = y
x_train2 = np.array(result_x, copy=True) 
y_train2 = np.array(result_y, copy=True) 

# datagen = ImageDataGenerator(rotation_range=10)
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)

# datagen.fit(result_x)
# np.random.seed(10)
# for x_batch, y_batch in datagen.flow(x_train2,y_train2,batch_size = 200):

# # for x_batch, y_batch in datagen.flow(x_train2,y_train2,batch_size=500):
#     x_train2 = np.concatenate((x_train2, x_batch))
#     y_train2 = np.concatenate((y_train2, y_batch))
#     break

result_x = x_train2
result_y = y_train2        

# result_x  = np.concatenate((result_x, x_train2), axis=0)
# result_y  = np.concatenate((result_y, y_train2), axis=0)

# datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
# datagen.fit(result_x)

# for x_batch, y_batch in datagen.flow(x_train2,y_train2,batch_size=32):
#     x_train2 = np.concatenate((x_train2, x_batch))
#     y_train2 = np.concatenate((y_train2, y_batch))
#     if x_train2.shape[0]>=1000:
#         break
# result_x  = np.concatenate((result_x, x_train2), axis=0)
# result_y  = np.concatenate((result_y, y_train2), axis=0)

x = result_x
x = x.reshape((x.shape[0], -1))

y = result_y
dims = [x.shape[-1], 500, 500, 2000, 10]
init = VarianceScaling(scale=1. / 3., mode='fan_in',distribution='uniform',seed=10)
pretrain_optimizer = SGD(lr=1, momentum=0.9)
# batch_size = 256

def autoencoder(dims, act='relu', init=keras.initializers.glorot_uniform(seed=10)):
# def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
        # x = Dense(dims[i], activation=act, keras.initializers.glorot_normal(seed=10), name='decoder_%d' % i)(x)
        
    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')
autoencoder, encoder = autoencoder(dims, init=init)
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
# autoencoder.fit(x, x, batch_size=batch_size, epochs=300) #, callbacks=cb)
autoencoder_train = autoencoder.fit(x, x,
                epochs=50,
                # batch_size=128,
                # shuffle=True,
                validation_data=(x_test, x_test))
epochs=50
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# autoencoder.save_weights('ae_weights'+'16000AUGMENT'+'seed.h5')
# autoencoder.load_weights('ae_weights'+'16000AUGMENT'+'seed.h5')
# Initialize cluster centers using k-means
np.random.seed(10)

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
correct = 0;
for k in range(10):
    
    t = y_pred == k #clustering
    c = [i for i, b in enumerate(t) if b] #counting 
    u = max(np.bincount(y[c]))
    print('cont=',np.bincount(y[c]))
    print('y=',y[c])
    correct += u

    # n = 20  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(1, n, i + 1)
    #     plt.imshow(x[c[i]].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     print(k)
    
        # display reconstruction
        # ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.show()
    # pairs = np.array([[9,0],[8,1],[7,3],[6,4],[5,2],[4,0],[3,6],[2,9],[1,5],[0,8]])

acc = correct/len(y)