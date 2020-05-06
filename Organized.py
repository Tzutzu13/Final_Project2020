from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import ReadingAudio
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import random
%matplotlib qt

# from Aug_size import Aug_size

batch_size = 32
num_classes = 2
epochs = 100

# input image dimensions
img_rows, img_cols = 32, 32
(Data,TrueLabels,Precentages,Image) = ReadingAudio.ReadingAudio()
# np.save('Data_Syllables', Data)
# np.save('TrueL', TrueLabels)
# np.save('perc', Precentages)
# np.save('original_image', Image)

Data = np.load('Data_Syllables.npy',allow_pickle = True)
TrueLabels = np.load('TrueL.npy',allow_pickle = True)
Precentages = np.load('perc.npy',allow_pickle = True)
image = np.load('original_image.npy',allow_pickle = True)

image[1].max()
image[1].min()
# for i in range(1,10):
#     # if image
#     plt.figure()
#     plt.imshow(image[i])
i=0
num_complex=0
num_upward=0
x_train = [None] * len(Data)
y_train = [None] * len(Data)

# Chevron_ind = TrueLabels == 'Chevron'
Complex_ind = TrueLabels == 'Complex'
Upward_ind = TrueLabels == 'Upward'



# Chevron_Syllables = Data[Chevron_ind]
Complex_Syllables= Data[Complex_ind][0:322]
Upward_Syllables= Data[Upward_ind]

# Chevron_Labels = np.zeros(len(TrueL[Chevron_ind])) # chevron label = 0
Upward_Labels = np.zeros(len(TrueLabels[Upward_ind])) # Upward label = 0
Complex_Labels = np.ones(len(TrueLabels[Complex_ind])) # Complex label = 1


data_Labels = np.concatenate((Upward_Labels,Complex_Labels[0:322]))

x_train = np.concatenate((Upward_Syllables,Complex_Syllables))
y_train = data_Labels
# for ind in range(0, len(Data)): 
#     if TrueLabels[ind]== 'Complex':
#         x_train[i]=Data[ind]
#         y_train[i]= 1 # Complex = 1
#         i+=1
#         num_complex+=1
#     if TrueLabels[ind]== 'Upward':
#         x_train[i]=Data[ind]
#         y_train[i]= 0 # Upward = 0
#         i+=1
#         num_upward+=1

x_train=[x for x in x_train if x is not None]
y_train=[x for x in y_train if x is not None]

random.seed(4)
random.shuffle(x_train)
random.seed(4)
random.shuffle(y_train)

num_of_test_samples=int(np.floor(0.15*len(x_train)))
x_test=x_train[0:num_of_test_samples]
x_train=x_train[(num_of_test_samples+1):len(x_train)]

y_test=y_train[0:num_of_test_samples]
y_train=y_train[(num_of_test_samples+1):len(y_train)]

y_test = np.asarray(y_test)
y_train = np.asarray(y_train)
x_train=np.asarray(x_train)
x_test=np.asarray(x_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape1 = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape1 = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train_cat = keras.utils.to_categorical(y_train)
# y_train_cat = y_train.astype('float32')
y_test_cat = keras.utils.to_categorical(y_test)
# y_test_cat = y_test.astype('float32')

# x = x_train.reshape(x_train.shape[0],-1)

# Girls model:
from numpy.random import seed
seed(1)
import tensorflow as tf
# from tensorflow import set_random_seed
tf.random.set_seed(2)
# tf.random.set_random_seed(2)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape1))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

CallBack=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

None in y_test_cat
# keras.optimizers.Adagrad()
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.Adagrad(lr=0.012, epsilon=None, decay=0.0),
              optimizer=keras.optimizers.Adagrad(lr=0.012),
              metrics=['accuracy'])

history=model.fit(x_train, y_train_cat,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=.2,
          callbacks=[CallBack])

import pandas as pd
pred = model.predict_classes(x_test)
from sklearn.metrics import classification_report, confusion_matrix
# classification_report(y_test,y_pred)
confm = confusion_matrix(y_test, pred)
df_cm = pd.DataFrame(confm)
import PrettyConfusionMetricsPlot
from PrettyConfusionMetricsPlot import pretty_plot_confusion_matrix
pretty_plot_confusion_matrix(df_cm)
plt.title('Complex vs Chevron - girls model')

losses = pd.DataFrame(model.history.history)
losses.plot()
#  end of girls model
    
#  our model
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import keras.backend as K
import seaborn as sns
import sklearn.metrics
%matplotlib qt
x = x_train.reshape(x_train.shape[0],-1)

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=7)
pca.fit(x)
print(pca.explained_variance_ratio_[0:7].sum())

print(pca.singular_values_)

c = pca.transform(x)

dims = [x.shape[-1], 500, 500, 2000, 6]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                       distribution='uniform',seed=10)
pretrain_optimizer = SGD(lr=1, momentum=0.9)
def autoencoder(dims, act='relu', init=keras.initializers.
                glorot_uniform(seed=10)):
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
autoencoder_train = autoencoder.fit(x, x,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_split = 0.2)

epochs=100
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

autoencoder.summary()

np.random.seed(10)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)
x_te = x_test.reshape(x_test.shape[0],-1)
# y_pred = kmeans.fit_predict(encoder.predict(x_te))
#  end of our model
pcac = pca.transform(x_te)
y_pred = kmeans.fit_predict(pcac)
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
# classification_report(y_test,y_pred)
confm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confm)
import PrettyConfusionMetricsPlot
from PrettyConfusionMetricsPlot import pretty_plot_confusion_matrix
pretty_plot_confusion_matrix(df_cm)

