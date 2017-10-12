import numpy as np
import time


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


def random(*args, max_time=500000):
    st_time = time.time()
    while time.time() - st_time < max_time:

        yield [arg() for arg in args]


def grid(*rngs, _id=0, total=1, _verbose=True):
    _grid = np.array(np.meshgrid(*rngs)).T.reshape((-1, len(rngs)))
    d = len(_grid)
    slice_of_the_grid = _grid[int(_id/total*d): int((_id+1)/total*d)]
    if _verbose:
        print('Process ', _id, ' will search in:')
        print(slice_of_the_grid)
    for e in slice_of_the_grid:
        yield e
