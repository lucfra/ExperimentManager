"""
Contains some misc utility functions
"""
from functools import reduce
import collections

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
    """
    Unfortunately this method doesn't return a very specific name....It gets a little messy

    :param var_dict:
    :param vars_:
    :return:
    """
    new_k_v = {}
    for v in vars_:
        for k, vv in var_dict.items():
            if v == vv:
                new_k_v[k] = v
    return name_from_dict(new_k_v)


def name_from_dict(_dict):
    string_dict = {str(k): str(v) for k, v in _dict.items()}
    return _tf_string_replace('_'.join(flatten_list(list(sorted(string_dict.items())))))


def _tf_string_replace(_str):
    """
    Replace chars that are not accepted by tensorflow namings (eg. variable_scope)

    :param _str:
    :return:
    """
    return _str.replace('[', '_p_').replace(']', '_p_').replace(',', '_c_')


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def GPU_CONFIG():
    import tensorflow as tf
    CONFIG_GPU_GROWTH = tf.ConfigProto(allow_soft_placement=True)
    CONFIG_GPU_GROWTH.gpu_options.allow_growth = True
    return CONFIG_GPU_GROWTH
