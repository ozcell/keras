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
from keras.layers.core import Layer, MaskedLayer, Dense, Dropout, Activation
from keras.layers.scol import LinDense, LinDenseTrainable
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm, nonneg, unitnorm
import keras.activations as activations
import keras.constraints as constraints
import keras.initializations_scol as initializations_scol
from keras import backend as K
import theano as M

from six.moves import cPickle

class LinDenseTrainable2(Layer):
    input_ndim = 2

    def __init__(self, output_dim, init='identity_vstacked', activation='linear', weights=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):

        self.init = initializations_scol.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.initial_weights = weights

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(LinDenseTrainable2, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W = identity_vstacked2((input_dim, self.output_dim))
        self.W2 = identity_vstacked3((input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        #self.trainable_weights = [self.W]
        self.non_trainable_weights = [self.W,self.b]
        self.trainable_weights = [self.W]
        self.W.trainable = True
        self.b.trainable = False

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(K.dot(X, (self.W*self.W2) + self.b))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'input_dim': self.input_dim,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None
                  }
        base_config = super(LinDenseTrainable2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Constraint(object):
    def __call__(self, p):
        return p

    def get_config(self):
        return {"name": self.__class__.__name__}


class MaxNorm2(Constraint):
    '''Constrain the weights incident to each hidden unit to have a norm less than or equal to a desired value.

    # Arguments
        m: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, m=2, axis=0):
        self.m = m
        self.axis = axis

    def __call__(self, p):
        norms = K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True))
        desired = K.clip(norms, 0, self.m)
        p = p * (desired / (K.epsilon() + norms))
        return p

    def get_config(self):
        return {"name": self.__class__.__name__,
                "m": self.m,
                "axis": self.axis}


class NonNeg2(Constraint):
    '''Constrain the weights to be non-negative.
    '''
    def __call__(self, p):
        p *= K.cast(0.1>= p >= 0., K.floatx())
        return p


class UnitNorm2(Constraint):
    '''Constrain the weights incident to each hidden unit to have unit norm.

    # Arguments
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).
    '''
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        return p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {"name": self.__class__.__name__,
                "axis": self.axis}


def identity_vstacked2(shape, scale=1):
    scale = shape[1]/shape[0]
    a = np.identity(shape[1])
    for i in range(1, int(1/scale)):
        a = np.vstack((a, np.identity(shape[1])))
    return K.variable(np.multiply(a, np.random.normal(loc=scale, scale=scale/3, size=shape)))

def identity_vstacked3(shape, scale=1):
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
        model.add(LinDenseTrainable2(nb_classes, W_constraint=NonNeg2()))
    else:
        model.add(Dense(nb_classes, W_constraint=W_constraint))
        model.add(Activation('softmax'))


    sgd = SGD(lr=sgd_lr, decay=1e-6, momentum=sgd_momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model

batch_size = 100
nb_epoch = 3334

X_train, Y_train, X_test, Y_test, nb_classes = set_dataset('MNIST')

model = define_single_layer_mlp((X_train.shape[1], ), 0.2, 3, 2048, 0.5, maxnorm(2), True, 2, 0.2)

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

model.save_weights('weights_2048x3_20_020_trainable_for_review.h5', overwrite=True)

f = open('histories_2048x3_20_020_trainable_for_review.save', 'wb')
cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

print('Test score:', score[0])
print('Test accuracy:', score[1])