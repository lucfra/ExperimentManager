"""
Contains (for the moment) static convenience methods for recording quantities
"""
import rfho as rf
import tensorflow as tf
import numpy as np
from functools import wraps
from experiment_manager.savers.save_and_load import Saver
from experiment_manager.utils import as_list
import sys

from experiment_manager.datasets.structures import NAMED_SUPPLIER


class on_hyperiteration:
    """
    context for record_hyper at each hyperiteration
    """

    def __init__(self, saver, *record_what, append_string='', do_print=None, collect_data=None, every=1):
        self.saver = saver
        self.append_string = append_string
        if self.append_string: self.append_string = '__' + self.append_string
        self.do_print = do_print or saver.do_print
        self.collect_data = collect_data or saver.collect_data

        self._unwrapped = []

        self._record_what = record_what or []
        self._processed_items = []

        self._step = 0
        self.every = every

    def __enter__(self):
        self._wrap()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.collect_data:
            if exc_tb:
                self.saver.save_text(str((str(exc_type), str(exc_val), str(exc_tb))),
                                     'exception' + self.append_string)

            self.saver.pack_save_dictionaries(append_string=self.append_string)

        self._unwrap()

        # TODO is this a good thing? or should we leave it to do manually
        self.saver.clear_items()
        if self.saver.timer: self.saver.timer.stop()

    def _wrap(self):
        self._unwrapped.append(rf.HyperOptimizer.initialize)
        rf.HyperOptimizer.initialize = self._initialize_wrapper(rf.HyperOptimizer.initialize)

        self._unwrapped.append(rf.HyperOptimizer.run)
        rf.HyperOptimizer.run = self._saver_wrapper(rf.HyperOptimizer.run)  # mmm...

    def _unwrap(self):
        rf.HyperOptimizer.initialize = self._unwrapped[0]
        rf.HyperOptimizer.run = self._unwrapped[1]

    def _saver_wrapper(self, f):
        @wraps(f)
        def _saver_wrapped(*args, **kwargs):
            res = f(*args, **kwargs)
            self._execute_save(res, *args, **kwargs)
            return res

        return _saver_wrapped

    def _initialize_wrapper(self, f):  # this should be good since
        @wraps(f)
        def _initialize_wrapped(*args, **kwargs):
            first_init = f(*args, **kwargs)
            # add savers just at the first initialization
            if first_init:
                self._processed_items += rf.flatten_list(
                    [Saver.process_items(*e(*args, **kwargs)) for e in self._record_what])
                # self._execute_save('INIT', *args, **kwargs)

            return first_init

        return _initialize_wrapped

    # noinspection PyUnusedLocal
    def _execute_save(self, res, *args, **kwargs):  # maybe args and kwargs could be useful...
        if self._step % self.every == 0:
            self.saver.save(step=self._step, append_string=self.append_string,
                            processed_items=self._processed_items,
                            do_print=self.do_print, collect_data=self.collect_data,
                            _res=res)
        self._step += 1


# noinspection PyPep8Naming
class on_run(on_hyperiteration):
    def __init__(self, saver, *record_what, append_string='', do_print=None, collect_data=None, every=1000):
        super().__init__(saver, *record_what, append_string=append_string, do_print=do_print, collect_data=collect_data,
                         every=every)
        self._uninitialized = True

    def _wrap(self):
        self._unwrapped.append(tf.Session.run)
        tf.Session.run = self._saver_wrapper(tf.Session.run)

    def _unwrap(self):
        tf.Session.run = self._unwrapped[0]

    def _execute_save(self, res, *args, **kwargs):
        if self._uninitialized:
            self._processed_items += rf.flatten_list(
                [Saver.process_items(*e(*args, **kwargs)) for e in self._record_what])
            self._uninitialized = False
        self._unwrap()  # otherwise we get infinite recursion!
        super()._execute_save(res, *args, **kwargs)
        self._wrap()


# noinspection PyClassHasNoInit,PyPep8Naming
class on_forward(on_hyperiteration):  # context class
    """
    Saves at every iteration (before call of method `step_forward`)
    """

    def _wrap(self):
        self._unwrapped.append(rf.HyperOptimizer.initialize)
        rf.HyperOptimizer.initialize = self._initialize_wrapper(rf.HyperOptimizer.initialize)

        self._unwrapped.append(rf.ForwardHG.step_forward)
        rf.ForwardHG.step_forward = self._saver_wrapper(rf.ForwardHG.step_forward)  # mmm...

        # self._unwrapped.append(rf.ReverseHG.)

    def _unwrap(self):
        rf.HyperOptimizer.initialize = self._unwrapped[0]
        rf.ForwardHG.step_forward = self._unwrapped[1]


def direct(*items):
    """
    Everything passed in items is passed directly to `Saver.

    :param items:
    :return:
    """

    # noinspection PyUnusedLocal
    def _call(*args, **kwargs):
        return items

    return _call


def norms_of_z():
    """

    :return:
    """

    def _call(*args, **kwargs):
        hg = args[0]
        if isinstance(hg, rf.HyperOptimizer): hg = hg.hyper_gradients  # guess most common case
        assert isinstance(hg, rf.ForwardHG)
        _rs = tensors(*hg.zs, op=tf.norm)(args, kwargs)
        return _rs

    return _call


def norms_of_d_dynamics_d_hypers(fd=None):
    """
    In `ForwardHG` records the norm of the partial derivatives of the dynamics w.r.t. the hyperparameters.

    :param fd:
    :return:
    """
    if fd is None: fd = lambda stp, rs: rs

    def _call(*args, **kwargs):
        hg = args[0]
        if isinstance(hg, rf.HyperOptimizer):
            hg = hg.hyper_gradients  # guess most common case
        assert isinstance(hg, rf.ForwardHG)
        _rs = tensors(*hg.d_dynamics_d_hypers, op=tf.norm,
                      fd=fd,
                      condition=lambda stp, rs: rs != 'INIT')(args, kwargs)
        return _rs

    return _call


def hyperparameters():
    """
    Simple one! record_hyper all hyperparameter values, assuming the usage of `HyperOptimizer`

    :return: a function
    """

    # noinspection PyUnusedLocal
    def _call(*args, **kwargs):
        hyper_optimizer = args[0]
        assert isinstance(hyper_optimizer, rf.HyperOptimizer)
        return rf.flatten_list(
            [rf.simple_name(hyp), hyp]
            for hyp in hyper_optimizer.hyper_list)

    return _call


def hypergradients():
    """
    Record all hypergradient values, assuming the usage of `HyperOptimizer`

    :return:
    """

    # noinspection PyUnusedLocal
    def _call(*args, **kwargs):
        hyper_optimizer = args[0]
        assert isinstance(hyper_optimizer, rf.HyperOptimizer)
        return rf.flatten_list(
            ['grad::' + rf.simple_name(hyp), hyper_optimizer.hyper_gradients.hyper_gradients_dict[hyp]]
            for hyp in hyper_optimizer.hyper_list)

    return _call


def tensors(*_tensors, key=None, scope=None, name_contains=None,
            rec_name='', op=tf.identity, fd=None, condition=True):
    """
    Little more difficult... attempts to record_hyper tensor named name

    :param name_contains: record_hyper all tensors which name contains this string. Can be a list.
    :type condition: bool | function
    :param condition: optional condition for triggering the saving of tensors, can have different
                        signatures
    :param _tensors: varargs of tensor names
    :param scope: optional for collections
    :param key: to record_hyper collections
    :param op: optional operation to apply to each tensor
    :param rec_name: optional name to prepend to all tensors recorded by this
    :param fd: # given to _process_feed_dicts_for_rec
    :return:
    """
    if isinstance(fd, str) and not rec_name: rec_name = fd

    def _call(*args, **_kwargs):
        nonlocal rec_name, _tensors
        if _tensors:
            _tensors = [tf.get_default_graph().get_tensor_by_name(tns + ':0') if isinstance(tns, str)
                        else tns for tns in _tensors]
        elif key:
            _tensors = tf.get_collection(key, scope=scope)
        elif name_contains:
            _names = rf.flatten_list([[n.name for n in tf.get_default_graph().as_graph_def().node
                                       if nc in n.name] for nc in as_list(name_contains)])
            return _tensors(*_names, rec_name=rec_name, op=op, fd=fd, condition=True)(*args, **_kwargs)
        else:
            raise NotImplemented

        if rec_name: rec_name += '::'  # maybe find a better way

        _rs2 = rf.flatten_list([rec_name + rf.simple_name(tns.name),
                                op(tns),
                                _process_feed_dicts_for_rec(fd, *args, **_kwargs),
                                condition]
                               for tns in _tensors)
        return _rs2

    return _call


# noinspection PyUnresolvedReferences
def exponential_running_average(*what, weight, condition=True):
    """
    computes exp running average till time k-1

    :param what:
    :param weight:
    :param condition:
    :return:
    """
    record_name = lambda name: '%s::running_average' % name
    is_number = lambda _v: isinstance(_v, (np.number, np.ndarray, float, int))

    # noinspection PyUnusedLocal
    def _call(*args, **kwargs):
        items = []
        for e in what:

            def _outer_comp(_e):
                rec_name = record_name(_e)

                # noinspection PyUnusedLocal
                def _compute(_, __, saver):
                    if saver.last_save_dict:
                        if is_number(saver.last_save_dict[rec_name]):
                            return weight * saver.last_save_dict[rec_name] + (1 - weight) * saver.last_save_dict[_e]
                        elif is_number(saver.last_save_dict[_e]):
                            return saver.last_save_dict[_e]
                        else:
                            return 'Not available'
                    else:
                        return 'Not available'

                return _compute

            items += [record_name(e), _outer_comp(e), condition]
        return items

    return _call


def number_of_nodes(condition=True):
    """
    records the number of node in the default graph, for debug purposes.....


    :return:
    """

    # noinspection PyUnusedLocal
    def _call(*_, **__):
        return ['# nodes', lambda _: len([n.name for n in tf.get_default_graph().as_graph_def().node]),
                condition]

    return _call


def model(_model, condition=True):
    """
    Should save the model(s) in a useful way..

    :return:
    """
    def _save_model(step, _, _saver):
        if _saver.collect_data:
            _saver.save_model(_model, step=step)
            return 'SAVED'
    return direct('Model', _save_model, condition)


def setting():  # TODO I have no precise idea on how to implement this...
    """
    Should save experiment meta-info like params, dataset, beginning/end...
    name of experiment function, git version and so on.

    :return:
    """
    raise NotImplemented()


# noinspection PyUnusedLocal
def _process_feed_dicts_for_rec(fd, *args, **kwargs):
    # TODO add more functionality...
    """

    # try with dictionary of form (string (simple name of placeholder), data)

    # Added named suppliers from rf.datasets ... introduce dynamic fetching in saver [fd can be str now]

    :param fd:
    :param args:  # might be useful??
    :param kwargs:
    :return:
    """
    if isinstance(fd, str):
        # it cannot remain a string otherwise is processed in a different way by
        # saver.add_items
        def _call():
            return NAMED_SUPPLIER[fd]

        return _call

    if fd is None or callable(fd) or isinstance(fd, str): return fd

    def _std_process_dict(_dict):
        return {tf.get_default_graph().get_tensor_by_name(n + ':0'): v for n, v in _dict.items()}

    def _fds():
        if isinstance(fd, dict):
            _rs = _std_process_dict(fd)

        elif isinstance(fd, (list, tuple)):  # (x, y, dataset)
            if len(fd) == 3 and isinstance(fd[2], rf.Dataset):  # very common scenario
                _rs = {tf.get_default_graph().get_tensor_by_name(fd[0] + ':0'): fd[2].data,
                       tf.get_default_graph().get_tensor_by_name(fd[1] + ':0'): fd[2].target,
                       }
            else:
                raise NotImplemented('not understood')
        else:
            raise NotImplemented('not understood')

        return _rs

    return _fds


class SMOS:
    """
    Save model on score"""

    def __init__(self, score_name, compare=None, init_best=-np.inf):
        self.score_name = score_name
        self.previous_score = init_best

        if not compare:
            compare = lambda ps, ns: ns > ps

        self.compare = compare

    def __call__(self, stp, _, saver, _partial_record):
        # if not _partial_record: return False  # nothing yet
        if self.score_name not in _partial_record:
            print('SMOS warning: %s not found in partial_record, score must be computed before this',
                  file=sys.stderr)
        score = _partial_record[self.score_name]
        if not isinstance(score, str):  # to avoids SKIP and/or other caught errors in saver.last_record
            if self.compare(self.previous_score, score):
                self.previous_score = score
                return True
        else: return False
