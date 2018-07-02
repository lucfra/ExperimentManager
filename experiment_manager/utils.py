"""
Contains some misc utility functions
"""
from functools import reduce
import collections
import multiprocessing

import numpy as np

import pickle

import os

from collections import OrderedDict, Callable

try:
    import pysftp
except ImportError as e:
    print(e)
    pysftp = None


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


def maybe_get(obj, i):
    return obj[i] if hasattr(obj, '__getitem__') else obj


def merge_dicts(*dicts):
    return reduce(lambda a, nd: {**a, **nd}, dicts, {})


def to_one_hot_enc(seq, dimension=None):
    da_max = dimension or int(np.max(seq)) + 1
    _tmp = np.zeros((len(seq), da_max))
    _tmp[range(len(_tmp)), np.array(seq, dtype=int)] = 1
    return _tmp
    #
    # def create_and_set(_p):
    #     _tmp = np.zeros(da_max)
    #     _tmp[int(_p)] = 1
    #     return _tmp
    #
    # return np.array([create_and_set(_v) for _v in seq])


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


def name_from_dict(_dict, *exclude_names):
    string_dict = {str(k): str(v) for k, v in _dict.items() if k not in exclude_names}
    return _tf_string_replace('_'.join(flatten_list(list(sorted(string_dict.items())))))


def _tf_string_replace(_str):
    """
    Replace chars that are not accepted by tensorflow namings (eg. variable_scope)

    :param _str:
    :return:
    """
    return _str.replace('[', 'p').replace(']', 'q').replace(',', 'c').replace('(', 'p').replace(')', 'q').replace(
        ' ', '')


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

half_int = lambda _m: 1.96*np.std(_m)/np.sqrt(len(_m)-1)


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


def save_on_server(filename, obj, remote_path=None, server='10.255.9.75', user='franceschi', pwd='luca',
                   keep_local=False):
    splitted_filename = filename.split(os.path.sep)
    if len(splitted_filename) > 1:
        filename = splitted_filename[-1]
        if remote_path is None: remote_path = ''
        remote_path += os.path.sep.join(splitted_filename[:-1])

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    with pysftp.Connection(server, username=user, password=pwd) as sftp:
        if remote_path:
            splitted = remote_path.split(os.path.sep)
            joined = [os.path.sep.join(splitted[:k]) for k in range(1, len(splitted) + 1)]
            [sftp.mkdir(j) for j in joined if not sftp.exists(j)]
        with sftp.cd(remote_path):
            sftp.put(filename)

    if not keep_local:
        os.remove(filename)


def open_from_server(filename, remote_path=None, server='10.255.9.75', user='franceschi', pwd='luca',
                     keep_local=False):
    splitted_filename = filename.split(os.path.sep)
    if len(splitted_filename) > 1:
        filename = splitted_filename[-1]
        if remote_path is None: remote_path = ''
        remote_path += os.path.sep.join(splitted_filename[:-1])
    with pysftp.Connection(server, username=user, password=pwd) as sftp:
        with sftp.cd(remote_path):
            sftp.get(filename)
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    if not keep_local:
        os.remove(filename)
    return obj


def get_global_step(name='GlobalStep', init=0):
    import tensorflow as tf
    return tf.get_variable(name, initializer=init, trainable=False,
                           collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


