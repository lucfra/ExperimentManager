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


def flatten_list(lst):
    from itertools import chain
    return list(chain(*lst))


def filter_vars(var_name, scope):
    import tensorflow as tf
    return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=scope.name if hasattr(scope, 'name') else scope)
            if v.name.endswith('%s:0' % var_name)]


def name_from_vars(var_dict, *vars_):
    new_k_v = {}
    for v in vars_:
        for k, vv in var_dict.items():
            if v == vv:
                new_k_v[k] = str(v)
    return '_'.join(flatten_list(list(sorted(new_k_v.items()))))


def GPU_CONFIG():
    import tensorflow as tf
    CONFIG_GPU_GROWTH = tf.ConfigProto(allow_soft_placement=True)
    CONFIG_GPU_GROWTH.gpu_options.allow_growth = True
    return CONFIG_GPU_GROWTH
