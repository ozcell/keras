from __future__ import absolute_import
from . import backend as K
from ..regularizers import Regularizer
from .utils.generic_utils import get_from_module
import warnings


class L3L4Regularizer(Regularizer):
    """Regularizer for L3 and L4 regularization.

    # Arguments
        l3: Float; L3 regularization factor.
        l4: Float; L4 regularization factor.
    """

    def __init__(self, l3=0., l4=0.):
        self.l3 = K.cast_to_floatx(l3)
        self.l4 = K.cast_to_floatx(l4)

    def __call__(self, x):
        regularization = 0

        size = K.shape(x)[1]

        x_pos = x * K.cast(x>0., K.floatx())
        x_pos_sum = K.sum(x_pos, axis=0)

        M = K.dot(x_pos.T, x_pos)
        N = K.dot(x_pos_sum.reshape((size,1)), x_pos_sum.reshape((1,size)))

        sumM = K.sum(M)
        sumN = K.sum(N)

        A = sumM - K.sum(K.theano.tensor.nlinalg.diag(M))
        B = (size-1)*K.sum(K.theano.tensor.nlinalg.diag(M))
        C = sumN - K.sum(K.theano.tensor.nlinalg.diag(N))
        D = (size-1)*K.sum(K.theano.tensor.nlinalg.diag(N))

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.similarity = (A/B)
        self.fullness = (C/D)

        if self.l3:
            regularization += self.l3 * (A/B)
        if self.l4:
            regularization += self.l4 * (1-(C/D))

        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l3': float(self.l3),
                'l4': float(self.l4)}


# Aliases.

ActivityRegularizer = L3L4Regularizer

def activity_l3(l=0.01):
    return L3L4Regularizer(l3=l)


def activity_l4(l=0.01):
    return L3L4Regularizer(l4=l)


def activity_l3l4(l3=0.01, l4=0.01):
    return L3L4Regularizer(l3=l3, l4=l4)


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
