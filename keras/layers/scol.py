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

class Layer(object):
    '''Abstract base layer class.

    All Keras layers accept certain keyword arguments:

        trainable: boolean. Set to "False" before model compilation
            to freeze layer weights (they won't be updated further
            during training).
        input_shape: a tuple of integers specifying the expected shape
            of the input samples. Does not includes the batch size.
            (e.g. `(100,)` for 100-dimensional inputs).
        batch_input_shape: a tuple of integers specifying the expected
            shape of a batch of input samples. Includes the batch size
            (e.g. `(32, 100)` for a batch of 32 100-dimensional inputs).
    '''
    def __init__(self, **kwargs):
        allowed_kwargs = {'input_shape',
                          'trainable',
                          'batch_input_shape',
                          'cache_enabled'}
        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg
        if 'input_shape' in kwargs:
            self.set_input_shape((None,) + tuple(kwargs['input_shape']))
        if 'batch_input_shape' in kwargs:
            self.set_input_shape(tuple(kwargs['batch_input_shape']))
        if 'trainable' in kwargs:
            self._trainable = kwargs['trainable']
        if not hasattr(self, 'params'):
            self.params = []
        self._cache_enabled = True
        if 'cache_enabled' in kwargs:
            self._cache_enabled = kwargs['cache_enabled']

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value

    def __call__(self, X, mask=None, train=False):
        # set temporary input
        tmp_input = self.get_input
        tmp_mask = None
        if hasattr(self, 'get_input_mask'):
            tmp_mask = self.get_input_mask
            self.get_input_mask = lambda _: mask
        self.get_input = lambda _: X
        Y = self.get_output(train=train)
        # return input to what it was
        if hasattr(self, 'get_input_mask'):
            self.get_input_mask = tmp_mask
        self.get_input = tmp_input
        return Y

    def set_previous(self, layer, connection_map={}):
        '''Connect a layer to its parent in the computational graph.
        '''
        assert self.nb_input == layer.nb_output == 1, 'Cannot connect layers: input count and output count should be 1.'
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), ('Incompatible shapes: layer expected input with ndim=' +
                                                                str(self.input_ndim) +
                                                                ' but previous layer has output_shape ' +
                                                                str(layer.output_shape))
        if layer.get_output_mask() is not None:
            assert self.supports_masked_input(), 'Cannot connect non-masking layer to layer with masked output.'
        self.previous = layer
        self.build()

    def build(self):
        '''Instantiation of layer weights.

        Called after `set_previous`, or after `set_input_shape`,
        once the layer has a defined input shape.
        Must be implemented on all layers that have weights.
        '''
        pass

    @property
    def trainable(self):
        if hasattr(self, '_trainable'):
            return self._trainable
        else:
            return True

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self):
        return 1

    @property
    def input_shape(self):
        # if layer is not connected (e.g. input layer),
        # input shape can be set manually via _input_shape attribute.
        if hasattr(self, 'previous'):
            return self.previous.output_shape
        elif hasattr(self, '_input_shape'):
            return self._input_shape
        else:
            raise Exception('Layer is not connected. Did you forget to set "input_shape"?')

    def set_input_shape(self, input_shape):
        if type(input_shape) not in [tuple, list]:
            raise Exception('Invalid input shape - input_shape should be a tuple of int.')
        input_shape = tuple(input_shape)
        if hasattr(self, 'input_ndim') and self.input_ndim:
            if self.input_ndim != len(input_shape):
                raise Exception('Invalid input shape - Layer expects input ndim=' +
                                str(self.input_ndim) +
                                ', was provided with input shape ' + str(input_shape))
        self._input_shape = input_shape
        self.input = K.placeholder(shape=self._input_shape)
        self.build()

    @property
    def output_shape(self):
        # default assumption: tensor shape unchanged.
        return self.input_shape

    def get_output(self, train=False):
        return self.get_input(train)

    def get_input(self, train=False):
        if hasattr(self, 'previous'):
            # to avoid redundant computations,
            # layer outputs are cached when possible.
            if hasattr(self, 'layer_cache') and self.cache_enabled:
                previous_layer_id = '%s_%s' % (id(self.previous), train)
                if previous_layer_id in self.layer_cache:
                    return self.layer_cache[previous_layer_id]
            previous_output = self.previous.get_output(train=train)
            if hasattr(self, 'layer_cache') and self.cache_enabled:
                previous_layer_id = '%s_%s' % (id(self.previous), train)
                self.layer_cache[previous_layer_id] = previous_output
            return previous_output
        elif hasattr(self, 'input'):
            return self.input
        else:
            raise Exception('Layer is not connected' +
                            'and is not an input layer.')

    def supports_masked_input(self):
        '''Whether or not this layer respects the output mask of its previous
        layer in its calculations.
        If you try to attach a layer that does *not* support masked_input to
        a layer that gives a non-None output_mask(), an error will be raised.
        '''
        return False

    def get_output_mask(self, train=None):
        '''For some models (such as RNNs) you want a way of being able to mark
        some output data-points as "masked",
        so they are not used in future calculations.
        In such a model, get_output_mask() should return a mask
        of one less dimension than get_output()
        (so if get_output is (nb_samples, nb_timesteps, nb_dimensions),
        then the mask is (nb_samples, nb_timesteps),
        with a one for every unmasked datapoint,
        and a zero for every masked one.

        If there is *no* masking then it shall return None.
        For instance if you attach an Activation layer (they support masking)
        to a layer with an output_mask, then that Activation shall
        also have an output_mask.
        If you attach it to a layer with no such mask,
        then the Activation's get_output_mask shall return None.

        Some layers have an output_mask even if their input is unmasked,
        notably Embedding which can turn the entry "0" into
        a mask.
        '''
        return None

    def set_weights(self, weights):
        '''Set the weights of the layer.

        weights: a list of numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).
        '''
        assert len(self.params) == len(weights), ('Provided weight array does not match layer weights (' +
                                                  str(len(self.params)) + ' layer params vs. ' +
                                                  str(len(weights)) + ' provided weights)')
        for p, w in zip(self.params, weights):
            if K.get_value(p).shape != w.shape:
                raise Exception('Layer shape %s not compatible with weight shape %s.' % (K.get_value(p).shape, w.shape))
            K.set_value(p, w)

    def get_weights(self):
        '''Return the weights of the layer,
        as a list of numpy arrays.
        '''
        weights = []
        for p in self.params:
            weights.append(K.get_value(p))
        return weights

    def get_config(self):
        '''Return the parameters of the layer, as a dictionary.
        '''
        config = {'name': self.__class__.__name__}
        if hasattr(self, '_input_shape'):
            config['input_shape'] = self._input_shape[1:]
        if hasattr(self, '_trainable'):
            config['trainable'] = self._trainable
        config['cache_enabled'] =  self.cache_enabled
        return config

    def get_params(self):
        consts = []
        updates = []

        if hasattr(self, 'regularizers'):
            regularizers = self.regularizers
        else:
            regularizers = []

        if hasattr(self, 'constraints') and len(self.constraints) == len(self.params):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity())
        elif hasattr(self, 'constraint') and self.constraint:
            consts += [self.constraint for _ in range(len(self.params))]
        else:
            consts += [constraints.identity() for _ in range(len(self.params))]

        if hasattr(self, 'updates') and self.updates:
            updates += self.updates

        return self.params, regularizers, consts, updates

    def count_params(self):
        '''Return the total number of floats (or ints)
        composing the weights of the layer.
        '''
        return sum([K.count_params(p) for p in self.params])


class LinDense(Layer):
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
