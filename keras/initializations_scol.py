from __future__ import absolute_import, division
import numpy as np
from . import backend as K

def identity_vstacked(shape, scale=1):
    scale = shape[1]/shape[0]
    a = np.identity(shape[1])
    for i in range(1, int(1/scale)):
        a = np.vstack((a, np.identity(shape[1])))
    return K.variable(scale * a)

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initialization')
