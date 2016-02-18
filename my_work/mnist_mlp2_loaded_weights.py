'''Train a simple deep NN on the MNIST dataset.

Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, LinDense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm

import scipy.io


batch_size = 100
nb_classes = 100
nb_epoch = 3334

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dropout(0.2, input_shape=(784,)))
model.add(Dense(2048, W_constraint=maxnorm(2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, W_constraint=maxnorm(2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, W_constraint=maxnorm(2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=2)

model.load_weights('./weights/mod/mlp_2048x3_100_0.00_mod.h5')

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

prediction = model.predict_classes(X_test, verbose=1)
scipy.io.savemat('mlp_2048x3_100_0.00.mat', mdict={'prediction': prediction})

print('Test score:', score[0])
print('Test accuracy:', score[1])