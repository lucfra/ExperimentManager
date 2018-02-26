"""
Contains some misc utility functions
"""
from functools import reduce
import collections
import multiprocessing

import numpy as np


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.

    :param obj:
    :return: A `list`
    """
    return obj if isinstance(obj, list) else [obj]


def maybe_call(obj, *args, **kwargs):
    """
    Calls obj with args and kwargs and return its result if obj is callable, otherwise returns obj.
    """
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


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


def get_rand_state(rand):
    """
    Utility methods for getting a `RandomState` object.

    :param rand: rand can be None (new State will be generated),
                    np.random.RandomState (it will be returned) or an integer (will be treated as seed).

    :return: a `RandomState` object
    """
    if isinstance(rand, np.random.RandomState):
        return rand
    elif isinstance(rand, (int, np.ndarray, list)) or rand is None:
        return np.random.RandomState(rand)
    else:
        raise ValueError('parameter rand {} has wrong type'.format(rand))


def GPU_CONFIG():
    import tensorflow as tf
    CONFIG_GPU_GROWTH = tf.ConfigProto(allow_soft_placement=True)
    CONFIG_GPU_GROWTH.gpu_options.allow_growth = True
    return CONFIG_GPU_GROWTH


# SOME SCORING UTILS FUNCTIONS

half_int = lambda _m: 1.96*np.std(_m)/np.sqrt(len(_m))


def mean_std_ci(measures, mul=1., tex=False):
    """
    Computes mean, standard deviation and 95% half-confidence interval for a list of measures.

    :param measures: list
    :param mul: optional multiplication coefficient (e.g. for percentage)
    :param tex: if True returns mean +- half_conf_interval for latex
    :return: a list or a string in latex
    """
    measures = np.array(measures)*mul
    ms = np.mean(measures), np.std(measures), half_int(measures)
    return ms if not tex else r"${:.2f} \pm {:.2f}$".format(ms[0], ms[2])


def leaky_relu(x, alpha, name=None):
    """
    Implements leaky relu with negative coefficient `alpha`
    """
    import tensorflow as tf
    with tf.name_scope(name, 'leaky_relu_{}'.format(alpha)):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def execute(target, *args, **kwargs):
    pr = multiprocessing.Process(target=target, args=args, kwargs=kwargs)
    pr.start()
    return pr

