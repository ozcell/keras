from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import cv2

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

batch_size = 100
nb_classes = 10
nb_epoch = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28,28)
X_test = X_test.reshape(10000, 28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train[0:5000], nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train2=np.zeros((5000,224,224),dtype='float32')
X_test2=np.zeros((10000,224,224),dtype='float32')

print('np zeros')

for i in range(len(X_train)/12):
    X_train2[i,:,:]=cv2.resize(X_train[i,:,:],(224,224))
for i in range(len(X_test)):
    X_test2[i,:,:]=cv2.resize(X_test[i,:,:],(224,224))

print('to 224')

X_train3=np.stack((X_train2,X_train2,X_train2), axis=1)
X_test3=np.stack((X_test2,X_test2,X_test2), axis=1)

print('to rgb')

model = VGG_16('/home/o/ozsel/git/keras/my_work/vgg16_weights.h5')

print('model 1 loaded')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

print('model 1 compiled')

model2 = Sequential()
model2.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model2.add(Convolution2D(64, 3, 3, weights=model.layers[1].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(64, 3, 3, weights=model.layers[3].get_weights(), activation='relu'))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(128, 3, 3, weights=model.layers[6].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(128, 3, 3, weights=model.layers[8].get_weights(), activation='relu'))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(256, 3, 3, weights=model.layers[11].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(256, 3, 3, weights=model.layers[13].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(256, 3, 3, weights=model.layers[15].get_weights(), activation='relu'))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, weights=model.layers[18].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, weights=model.layers[20].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, weights=model.layers[22].get_weights(), activation='relu'))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, weights=model.layers[25].get_weights(),activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, weights=model.layers[27].get_weights(), activation='relu'))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, weights=model.layers[29].get_weights(), activation='relu'))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(Flatten())
model2.add(Dense(4096, weights=model.layers[32].get_weights(), activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(4096, weights=model.layers[34].get_weights(),activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))

print('model 2 loaded')

model2.compile(optimizer=sgd, loss='categorical_crossentropy')

print('model 2 compiled')

for i in range(36):
    model2.layers[i].trainable = False

print('model 2 freezed')

del model, X_train2, X_test2

history2 = model2.fit(X_train3, Y_train,
                      batch_size=batch_size, nb_epoch=50,
                      show_accuracy=True, verbose=1,
                      validation_data=(X_test3, Y_test))
