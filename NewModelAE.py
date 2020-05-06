"""
Created on Thu Mar 19 22:05:46 2020

@author: yaniv
"""
import numpy as np
import random as rn
import tensorflow as tf
# tf.set_random_seed(7) # for googleColab
import os
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import sklearn.metrics
import seaborn as sns

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

from keras.initializers import VarianceScaling
tf.random.set_seed(7)
init = VarianceScaling(scale=1. / 3., mode='fan_in',distribution='uniform',seed=10)
act='relu'
input_img = Input(shape=(784,), name='input')

# input_img = Input(784)  # adapt this if using `channels_first` image data format
x = input_img
x = Dense(500, activation=act, kernel_initializer=init, name='encoder_1' )(x)
x = Dense(500, activation=act, kernel_initializer=init, name='encoder_2' )(x)
x = Dense(2000, activation=act, kernel_initializer=init, name='encoder_3')(x)
encoded = Dense(10, activation=act, kernel_initializer=init, name='encoder_4')(x)
# encoded = Dense(dims = 10, kernel_initializer=init, name='encoder_%d'(n_stacks - 1))(x)  # hidden layer, features are extracted from here
x = encoded
x = Dense(2000, activation=act, kernel_initializer=init, name='decoder_4')(x)
x = Dense(500, activation=act, kernel_initializer=init, name='decoder_3')(x)
x = Dense(500, activation=act, kernel_initializer=init, name='decoder_2')(x)
x = Dense(10, activation=act, kernel_initializer=init, name='decoder_1')(x)
x = Dense(28*28, kernel_initializer=init, name='decoder_0')(x)
decoded = x
autoencoder = Model(inputs=input_img, outputs=decoded, name='AE')
Encoder = Model(inputs=input_img, outputs=encoded, name='encoder')

# input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = (x_train.astype('float32') / 255.)[0:500]
x_test = (x_test.astype('float32') / 255.)[0:500]
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
y = y_test[0:1000]
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

autoencoder_train = autoencoder.fit(x_train, x_train,
                epochs=50,
                # batch_size=128,
                # shuffle=True,
                validation_data=(x_test, x_test))

# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 # batch_size=128,
#                 # shuffle=True,
#                 validation_data=(x_test, x_test))

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

autoencoder.summary()
Encoder.summary()

kmeans = KMeans(n_clusters=10, n_init=20)
y_pred = kmeans.fit_predict(Encoder.predict(x_train))
# for i, layer in enumerate(autoencoder.layers):
#     layer.name = 'layer_' + str(i)
# autoencoder.summary()

# np.random.seed(7)

# layer_name = 'layer_7'
# intermediate_layer_model = Model(inputs=autoencoder.input,
#                                  outputs=autoencoder.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(x_train)

# x_pred = intermediate_output.reshape((intermediate_output.shape[0], -1))

# kmeans = KMeans(n_clusters=10, random_state=0).fit(x_pred)

# kmeans = KMeans(n_clusters=10).fit(x_pred)
# np.random.seed(7)

# layer_name = 'layer_7'
# intermediate_layer_model = Model(inputs=autoencoder.input,
#                                  outputs=autoencoder.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(x_test)

# x_pred1 = intermediate_output.reshape((intermediate_output.shape[0], -1))

# y_pred = kmeans.predict(x_pred1)

# encoder = Model(input_img, encoded)

# decoder = Model(encoder.output, decoded)

# encoded_input = Input(shape=(128,))
# # retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[7:]
# # create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))

# autoencode = Model(input_img, decoder)
# confusion_matrix = sklearn.metrics.confusion_matrix(y, y)

# plt.hist(y)

# decoded_imgs = autoencoder.predict(x_test)

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i+n + 1)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


for k in range(10):
    
    t = y_pred == k
    c = [i for i, x in enumerate(t) if x]
    
    # f = y == 7
    # b = [i for i, x in enumerate(f) if x]
    
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[c[i]].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        # ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.show()


# clu = kmeans.cluster_centers_
    
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5);