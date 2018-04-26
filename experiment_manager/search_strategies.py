import numpy as np
import time
from experiment_manager.utils import name_from_dict


def log_unif(a, b, integer=False):
    def _call():
        sml = np.exp(np.random.uniform(np.log(a), np.log(b)))
        return int(sml) if integer else sml
    return _call


def unif(a, b, integer=False):
    """
    Returns a function that samples form uniform distribution form a to b
    :param a:
    :param b:
    :param integer:
    :return:
    """
    def _call():
        sml = np.random.uniform(a, b)
        return int(sml) if integer else sml
    return _call


def random(names, *args, max_time=500000):
    st_time = time.time()
    while time.time() - st_time < max_time:
        e = [arg() for arg in args]  # sample
        dict_ = {n: _e for n, _e in zip(names, e)}
        yield name_from_dict(dict_), dict_


def grid(names, *rngs, types=None, _id=0, total=1, _verbose=True):
    _grid = np.array(np.meshgrid(*rngs)).T.reshape((-1, len(rngs)))
    d = len(_grid)
    slice_of_the_grid = _grid[int(_id/total*d): int((_id+1)/total*d)]
    if isinstance(names[0], tuple):
        print(names)
        types = [n[1] for n in names]
        names = [n[0] for n in names]

    if not types:
        types = [lambda _x: _x]*len(names)
    if _verbose:
        print('Process ', _id, ' will search in:')
        print(slice_of_the_grid)
    for e in slice_of_the_grid:
        dict_ = {n: tp(_e) for n, _e, tp in zip(names, e, types)}
        yield name_from_dict(dict_), dict_


def grid_dict(dictionary, types=None,_id=0, total=1, _verbose=True):
    return grid(list(dictionary.keys()), *list(dictionary.values()), types=types,
                _id=_id, total=total, _verbose=_verbose)
