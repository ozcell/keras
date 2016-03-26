'''Train a simple deep NN on the MNIST dataset.

Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function, absolute_import, division
import numpy as np
import scipy as sc
import scipy.io
import matplotlib.pyplot as plot
import matplotlib.cm as cm
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation
from keras.layers.scol import LinDense, LinDenseTrainable
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm
from six.moves import cPickle

import keras.activations as activations
import keras.initializations_scol as initializations_scol
from keras import backend as K
import theano as M

class LinDenseTrainable2(Layer):
    input_ndim = 2

    def __init__(self, output_dim, init='identity_vstacked', activation='linear', weights=None,
                 input_dim=None, **kwargs):

        self.init = initializations_scol.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(LinDenseTrainable2, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W = identity_vstacked2((input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        #self.trainable_weights = [self.W]
        self.non_trainable_weights = [self.W,self.b]
        self.W.trainable = False
        self.b.trainable = False

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        A = M.tensor.max(X[:,0:10],axis=1)
        B = M.tensor.stack((M.tensor.max(X[:,0:10],axis=1),
                 M.tensor.max(X[:,10:20],axis=1),
                 M.tensor.max(X[:,20:30],axis=1),
                 M.tensor.max(X[:,30:40],axis=1),
                 M.tensor.max(X[:,40:50],axis=1),
                 M.tensor.max(X[:,50:60],axis=1),
                 M.tensor.max(X[:,60:70],axis=1),
                 M.tensor.max(X[:,70:80],axis=1),
                 M.tensor.max(X[:,80:90],axis=1),
                 M.tensor.max(X[:,90:100],axis=1)),axis=1)


        output = self.activation(B)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'input_dim': self.input_dim}
        base_config = super(LinDenseTrainable2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_vstacked2(shape, scale=1):
    scale = shape[1]/shape[0]
    a = np.identity(shape[1])
    for i in range(1, int(1/scale)):
        a = np.vstack((a, np.identity(shape[1])))
    return K.variable(a)

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
                            nb_layers=1, nb_units_per_layer=256, dropout=0.5, W_constraint=None,
                            SCOL=False, SCOL_k=10, SCOL_p=0.5,
                            sgd_lr=0.01, sgd_momentum=0.95):
    model = Sequential()
    model.add(Dropout(input_dropout, input_shape=input_shape))
    for i in range(nb_layers):
        model.add(Dense(nb_units_per_layer, W_constraint=W_constraint))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))

    if SCOL == True:
        model.add(Dense(nb_classes*SCOL_k, W_constraint=W_constraint))
        model.add(Dropout(SCOL_p))
        model.add(Activation('softmax'))
        model.add(LinDenseTrainable2(nb_classes))
    else:
        model.add(Dense(nb_classes, W_constraint=W_constraint))
        model.add(Activation('softmax'))


    sgd = SGD(lr=sgd_lr, decay=1e-6, momentum=sgd_momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model

batch_size = 100
nb_epoch = 3334

X_train, Y_train, X_test, Y_test, nb_classes = set_dataset('MNIST')

model = define_single_layer_mlp((X_train.shape[1], ), 0.2, 3, 2048, 0.5, maxnorm(2), True, 10, 0.5)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    show_accuracy=True, verbose=2,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=2)

results= []
results.append(np.asarray(history.history.get('loss')))
results.append(np.asarray(history.history.get('val_loss')))
results.append(np.asarray(history.history.get('acc')))
results.append(np.asarray(history.history.get('val_acc')))

model.save_weights('weights_2048x3_100_050_max_for_review.h5', overwrite=True)

f = open('histories_2048x3_100_050_max_for_review.save', 'wb')
cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

print('Test score:', score[0])
print('Test accuracy:', score[1])
