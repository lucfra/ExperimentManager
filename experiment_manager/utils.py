"""
Contains some misc utility functions
"""
from functools import reduce

import numpy as np


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.

    :param obj:
    :return: A `list`
    """
    return obj if isinstance(obj, list) else [obj]


def as_tuple_or_list(obj):
    """
    Make sure that `obj` is a tuple or a list and eventually converts it into a list with a single element

    :param obj:
    :return: A `tuple` or a `list`
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def merge_dicts(*dicts):
    return reduce(lambda a, nd: {**a, **nd}, dicts, {})


def to_one_hot_enc(seq, dimension=None):
    da_max = dimension or np.max(seq) + 1

    def create_and_set(_p):
        _tmp = np.zeros(da_max)
        _tmp[_p] = 1
        return _tmp

    return np.array([create_and_set(_v) for _v in seq])


def filter_vars(var_name, scope):
    import tensorflow as tf
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=scope.name) if v.name.endswith('%s:0' % var_name)]
