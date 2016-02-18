from __future__ import absolute_import
from __future__ import print_function
import os
import itertools
import numpy as np
np.random.seed(1337)  # for reproducibility

from kerosene.datasets import svhn2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.scol import LinDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.constraints import maxnorm

'''
    Train a convnet on the cropped Stanford Street View House Numbers dataset.

    http://ufldl.stanford.edu/housenumbers/

    This version can get to 93.05% test accuracy after 12 epochs.
    35 seconds per epoch on a GeForce GTX 680 GPU.

    Or you can try:
      USE_EXTRA=1 python svhn2.py

    to also train on the much larger set that includes "extra" data.

    With this extra data, this gets to 96.40% test accuracy after 12 epochs.
    273 seconds per epoch on a GeForce GTX 680 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 50

# standard split is 73,257 train / 26,032 test
(X_train, y_train), (X_test, y_test) = svhn2.load_data()

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

# input image dimensions
_, img_channels, img_rows, img_cols = X_train.shape

model = Sequential()
model.add(Dropout(0.1, input_shape=(img_channels, img_rows, img_cols)))
model.add(Convolution2D(96, 5, 5, border_mode='same',
                        W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 5, 5, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 5, 5, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2048, W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
#model.add(Dropout(0.5))
model.add(Activation('softmax'))
#model.add(LinDense(nb_classes, dropout_rate=0.5))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.003, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

if "USE_EXTRA" in os.environ:
    # svhn2 extra split has an additional 531,131 (!) examples
    (X_extra, y_extra), = svhn2.load_data(sets=['extra'])
    X_train = np.concatenate([X_train, X_extra])
    y_train = np.concatenate([y_train, y_extra])

y_train = y_train%10
y_test = y_test%10

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

#model.save_weights('weights_svhn_cnn_1024_100_050_sgd.h5', overwrite=True)

print('Test score:', score[0])
print('Test accuracy:', score[1])
