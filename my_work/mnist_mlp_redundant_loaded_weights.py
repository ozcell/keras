'''Train a simple deep NN on the MNIST dataset.

Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, LinDense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.utils.np_utils import accuracy


batch_size = 100
nb_classes = 10
nb_epoch = 100

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

# model = Sequential()
# model.add(Dense(2048, input_shape=(784,), W_constraint=maxnorm(2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2048, W_constraint=maxnorm(2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2048, W_constraint=maxnorm(2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Dense(1000))
# model.add(Activation('softmax'))
# model.add(LinDense(10))

model = Graph()
model.add_input(name='input1', input_shape=(784,))
# model.add_input(name='input2', input_shape=(784,))
model.add_node(Dense(2048, activation='relu', W_constraint=maxnorm(2)), name='dense1', input='input1')
model.add_node(Dense(2048, activation='relu', W_constraint=maxnorm(2)), name='dense2', input='dense1')
model.add_node(Dense(2048, activation='relu', W_constraint=maxnorm(2)), name='dense3', input='dense2')
# model.add_node(Dense(40, activation='relu'), name='dense4', input='dense3')

model.add_node(Dense(2048, activation='relu', W_constraint=maxnorm(2)), name='dense5', input='input1')
model.add_node(Dense(2048, activation='relu', W_constraint=maxnorm(2)), name='dense6', input='dense5')
model.add_node(Dense(2048, activation='relu', W_constraint=maxnorm(2)), name='dense7', input='dense6')
# model.add_node(Dense(10, activation='relu'), name='dense8', input='dense7')

model.add_node(Dense(40, activation='softmax'), name='dense9', inputs=['dense3', 'dense7'], merge_mode='sum')

model.add_node(LinDense(10), name='lin', input='dense9')

# model.add_node(Dense(10, activation='softmax'), name='dense9', input='dense4')


model.add_output(name='output', input='lin')

rms = RMSprop()
model.compile(optimizer='adadelta', loss={'output':'categorical_crossentropy'})

model.load_weights('weights.h5')

acc = accuracy(Y_test,
               np.round(np.array(model.predict({'input1': X_test},
                                               batch_size=batch_size)['output'])))

print('Test accuracy:', acc)