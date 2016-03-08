# coding: utf-8

'''Train a simple deep NN on the MNIST dataset of varying sizes.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.scol import LinDense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm

import scipy as sc
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import sys

from six.moves import cPickle
import gzip

#%matplotlib inline

batch_size = 100

def set_dataset(dataset='MNIST'):
    if dataset == 'MNIST':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        nb_classes = 10

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
    elif dataset == 'SVHN':
        pass
    elif dataset == 'CIFAR10':
        pass
    elif dataset == 'CIFAR100':
        pass

    return X_train, Y_train, X_test, Y_test, nb_classes


def define_single_layer_mlp(input_shape, input_dropout=0.2,
                            nb_layers=1, nb_units_per_layer=256,
                            dropout=0.5, W_constraint=None,
                            SCOL=False, SCOL_k=10, SCOL_p=0.5,
                            sgd_lr=0.01, sgd_momentum=0.90):
    model = Sequential()
    model.add(Dropout(input_dropout, input_shape=input_shape))
    for i in range(nb_layers):
        model.add(Dense(nb_units_per_layer, W_constraint=W_constraint))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))

    if SCOL == True:
        model.add(Dense(nb_classes*SCOL_k))
        model.add(Dropout(SCOL_p))
        model.add(Activation('softmax'))
        model.add(LinDense(nb_classes,dropout_rate=SCOL_p))
    else:
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))


    sgd = SGD(lr=sgd_lr, decay=1e-6, momentum=sgd_momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def plot_side_by_side(dictionary):
    fig = plot.figure(figsize=(16,4))

    left = fig.add_subplot(121)
    right = fig.add_subplot(122)

    left.plot(np.asarray(dictionary.get('val_loss')),
                        label='val_loss', linewidth=2)
    left.plot(np.asarray(dictionary.get('loss')),
                        label='loss', linewidth=2)
    #left.set_ylim(0,0.25)
    left.legend()

    right.plot(100*(1-np.asarray(dictionary.get('val_acc'))),
                                label='val_acc', linewidth=2)
    right.plot(100*(1-np.asarray(dictionary.get('acc'))),
                                label='acc', linewidth=2)
    #right.set_ylim(0,4)
    right.legend()

    return fig

def mnist_dataset_size(nb_weight_updates=120000):

    results = []
    i=0

    for dataset_size in [100, 500, 1000, 5000, 10000, 30000, 60000]:
        nb_epoch = [40000, 10000, 10000, 1000, 1000, 2000, 1000]#nb_weight_updates/(dataset_size/batch_size)

        model1 = define_single_layer_mlp((X_train.shape[1], ), 0, 2, 2048, 0, None,
                                        False)
        model2 = define_single_layer_mlp((X_train.shape[1], ), 0, 2, 2048, 0, None,
                                        True, 10, 0)
        model3 = define_single_layer_mlp((X_train.shape[1], ), 0, 2, 2048, 0, None,
                                        True, 10, 0.5)

        history1 = model1.fit(X_train[0:dataset_size,:], Y_train[0:dataset_size,:],
                   batch_size=batch_size, nb_epoch=nb_epoch[i],
                   show_accuracy=True, verbose=2,
                   validation_data=(X_test, Y_test))

        history2 = model2.fit(X_train[0:dataset_size,:], Y_train[0:dataset_size,:],
                   batch_size=batch_size, nb_epoch=nb_epoch[i],
                   show_accuracy=True, verbose=2,
                   validation_data=(X_test, Y_test))

        history3 = model3.fit(X_train[0:dataset_size,:], Y_train[0:dataset_size,:],
                   batch_size=batch_size, nb_epoch=nb_epoch[i],
                   show_accuracy=True, verbose=2,
                   validation_data=(X_test, Y_test))

        results.append([])
        results[i].append(np.asarray(history1.history.get('loss')))
        results[i].append(np.asarray(history1.history.get('val_loss')))
        results[i].append(np.asarray(history1.history.get('acc')))
        results[i].append(np.asarray(history1.history.get('val_acc')))

        results[i].append(np.asarray(history2.history.get('loss')))
        results[i].append(np.asarray(history2.history.get('val_loss')))
        results[i].append(np.asarray(history2.history.get('acc')))
        results[i].append(np.asarray(history2.history.get('val_acc')))

        results[i].append(np.asarray(history3.history.get('loss')))
        results[i].append(np.asarray(history3.history.get('val_loss')))
        results[i].append(np.asarray(history3.history.get('acc')))
        results[i].append(np.asarray(history3.history.get('val_acc')))

        i=i+1

    return results

X_train, Y_train, X_test, Y_test, nb_classes = set_dataset('MNIST')
results = mnist_dataset_size(int(sys.argv[1]))

f = open('histories_1.pkl', 'wb')
cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
