from __future__ import absolute_import, division
import numpy as np
from . import backend as K
from .utils.generic_utils import get_from_module

def identity_vstacked(shape, scale=1, name=None, dim_ordering='th'):
    scale = shape[1]/shape[0]
    a = np.identity(shape[1])
    for i in range(1, int(1/scale)):
        a = np.vstack((a, np.identity(shape[1])))
    return K.variable(a, name=name)

def get(identifier, **kwargs):
    return get_from_module(identifier, globals(),
                           'initialization', kwargs=kwargs)        
