import chainer as ch
import numpy as np


def tensor(x):
    return ch.as_array(np.asarray(x))


def sum(x, axis=None):
    ret = ch.as_array((ch.functions.sum(np.asfarray(x),
                                        axis=axis)))
    if "int" in str(np.asarray(x).dtype):
        ret = ret.astype(np.int32)
    return ret
