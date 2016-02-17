# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from collections import OrderedDict
import copy
from six.moves import zip

from .. import backend as K
from .. import activations, initializations, initializations_scol, regularizers, constraints
from ..regularizers import ActivityRegularizer

import marshal
import types
import sys


class LinDense(keras.layers.core.Layer):
    '''Just your regular fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, output_dim, dropout_rate=1, init='identity_vstacked', activation='linear', weights=None,
                 input_dim=None, **kwargs):

        self.init = initializations_scol.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
	self.dropout_rate = dropout_rate

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(LinDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

	self.W = self.init((input_dim, self.output_dim))*(1/1-self.dropout_rate)
        self.b = K.zeros((self.output_dim,))
        self.params = [self.W, self.b]
        self.trainable = False

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(K.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'input_dim': self.input_dim,
		  'dropout_rate': self.dropout_rate}
        base_config = super(LinDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
