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
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.scol import LinDense
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD

from six.moves import cPickle
import scipy.io

batch_size = 100
nb_classes = 10
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

model.add(Dense(nb_classes*10, W_constraint=maxnorm(2)))
model.add(Dropout(0.5))
model.add(Activation('softmax'))
model.add(LinDense(nb_classes))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    show_accuracy=True, verbose=2,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=2)

results = []
results.append(np.asarray(history.history.get('loss')))
results.append(np.asarray(history.history.get('val_loss')))
results.append(np.asarray(history.history.get('acc')))
results.append(np.asarray(history.history.get('val_acc')))

model.save_weights('weights_2048x3_100_050.h5', overwrite=True)

f = open('history_2048x3_100_050.save', 'wb')
cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#prediction = model.predict_classes(X_test, verbose=2)
#scipy.io.savemat('predictions_2048x3_10.mat', mdict={'prediction': prediction})

print('Test score:', score[0])
print('Test accuracy:', score[1])
