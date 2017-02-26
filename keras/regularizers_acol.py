from __future__ import absolute_import
from . import backend as K
from .regularizers import Regularizer
from .utils.generic_utils import get_from_module
import warnings

Tr = K.theano.tensor.nlinalg.trace

class ACOLRegularizer(Regularizer):
    """Regularizer for ACOL.

    # Arguments
        c1: Float; affinity factor.
        c2: Float; balance factor.
        c3: Float; coactivity factor.
        c4: Float; L2 regularization factor.
    """

    def __init__(self, c1=0., c2=0., c3=0., c4=0.):
        self.c1 = K.variable(c1)
        self.c2 = K.variable(c2)
        self.c3 = K.variable(c3)
        self.c4 = K.variable(c4)

    def __call__(self, x):
        regularization = 0
        Z = x
        n = K.shape(Z)[1]

        Z_bar = Z * K.cast(x>0., K.floatx())
        v = K.sum(Z_bar, axis=0).reshape(1,n)

        U = K.dot(Z_bar.T, Z_bar)
        V = K.dot(v.T, v)

        affinity = (K.sum(U) - Tr(U))/((n-1)*Tr(U))
        balance = (K.sum(V) - Tr(V))/((n-1)*Tr(V))
        coactivity = K.sum(U) - Tr(U)

        if self.c1:
            regularization += self.c1 * affinity
        if self.c2:
            regularization += self.c2 * (1-balance)
        if self.c3:
            regularization += self.c3 * coactivity
        if self.c4:
            regularization += K.sum(self.c4 * K.square(Z))

        self.affinity = affinity
        self.balance = balance
        self.coactivity = coactivity
        self.reg = regularization

        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c1': float(self.c1),
                'c2': float(self.c2),
                'c3': float(self.c3),
                'c4': float(self.c4)}

# Aliases.

ActivityRegularizer = ACOLRegularizer


def activity_ACOL(c1=1., c2=1., c3=0., c4=0.000001,):
    return ACOLRegularizer(c1=c1, c2=c2, c3=c3, c4=c4)


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
