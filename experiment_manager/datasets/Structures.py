from functools import reduce

import numpy as np


class Datasets:
    """
    Simple object for standard datasets. Has the field `train` `validation` and `test` and support indexing
    """

    def __init__(self, train=None, validation=None, test=None):
        self.train = train
        self.validation = validation
        self.test = test
        self._lst = [train, validation, test]

    def setting(self):
        return {k: v.setting() if hasattr(v, 'setting') else None for k, v in vars(self).items()}

    def __getitem__(self, item):
        return self._lst[item]

    def __len__(self):
        return len([_ for _ in self._lst if _ is not None])

    @staticmethod
    def from_list(list_of_datasets):
        """
        Generates a `Datasets` object from a list.

        :param list_of_datasets: list containing from one to three dataset
        :return:
        """
        train, valid, test = None, None, None
        train = list_of_datasets[0]
        if len(list_of_datasets) > 3:
            print('There are more then 3 Datasets here...')
            return list_of_datasets
        if len(list_of_datasets) > 1:
            test = list_of_datasets[-1]
            if len(list_of_datasets) == 3:
                valid = list_of_datasets[1]
        return Datasets(train, valid, test)

    @staticmethod
    def stack(*datasets_s):
        """
        Stack some datasets calling stack for each dataset.

        :param datasets_s:
        :return: a new dataset
        """
        return Datasets.from_list([Dataset.stack(*[d[k] for d in datasets_s if d[k] is not None])
                                   for k in range(3)])


def _maybe_cast_to_scalar(what):
    return what[0] if len(what) == 1 else what


def convert_sparse_matrix_to_sparse_tensor(X):
    import tensorflow as tf
    import scipy.sparse as sc_sp
    if isinstance(X, sc_sp.csr.csr_matrix):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
    else:
        coo, indices = X, [X.row, X.col]
    # data = np.array(coo.data, dtype=)
    return tf.SparseTensor(indices, tf.constant(coo.data, dtype=tf.float32), coo.shape)


NAMED_SUPPLIER = {}


def merge_dicts(*dicts):
    return reduce(lambda a, nd: {**a, **nd}, dicts, {})


class Dataset:
    """
    Class for managing a single dataset, includes data and target fields and has some utility functions.
     It allows also to convert the dataset into tensors and to store additional information both on a
     per-example basis and general infos.
    """

    def __init__(self, data, target, sample_info=None, info=None):
        """

        :param data: Numpy array containing data
        :param target: Numpy array containing targets
        :param sample_info: either an array of dicts or a single dict, in which case it is cast to array of
                                  dicts.
        :param info: (optional) dictionary with further info about the dataset
        """
        self._tensor_mode = False

        self._data = data
        self._target = target
        if self._data is not None:  # in meta-dataset data and target can be unspecified
            if sample_info is None:
                sample_info = {}
            self.sample_info = np.array([sample_info] * self.num_examples) \
                if isinstance(sample_info, dict) else sample_info

            assert self.num_examples == len(self.sample_info), str(self.num_examples) + ' ' + str(len(self.sample_info))
            assert self.num_examples == self._shape(self._target)[0]

        self.info = info or {}

    def _shape(self, what):
        return what.get_shape().as_list() if self._tensor_mode else what.shape

    def setting(self):
        """
        for save setting purposes, does not save the actual data

        :return:
        """
        return {
            'num_examples': self.num_examples,
            'dim_data': self.dim_data,
            'dim_target': self.dim_target,
            'info': self.info
        }

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def num_examples(self):
        """

        :return: Number of examples in this dataset
        """
        return self._shape(self.data)[0]

    @property
    def dim_data(self):
        """

        :return: The data dimensionality as an integer, if input are vectors, or a tuple in the general case
        """
        return _maybe_cast_to_scalar(self._shape(self.data)[1:])

    @property
    def dim_target(self):
        """

        :return: The target dimensionality as an integer, if targets are vectors, or a tuple in the general case
        """
        shape = self._shape(self.target)
        return 1 if len(shape) == 1 else _maybe_cast_to_scalar(shape[1:])

    def convert_to_tensor(self, keep_sparse=True):
        import scipy.sparse as sc_sp
        import tensorflow as tf
        SPARSE_SCIPY_MATRICES = (sc_sp.csr.csr_matrix, sc_sp.coo.coo_matrix)
        matrices = ['_data', '_target']
        for att in matrices:
            if keep_sparse and isinstance(self.__getattribute__(att), SPARSE_SCIPY_MATRICES):
                self.__setattr__(att, convert_sparse_matrix_to_sparse_tensor(self.__getattribute__(att)))
            else:
                self.__setattr__(att, tf.convert_to_tensor(self.__getattribute__(att), dtype=tf.float32))
        self._tensor_mode = True

    def create_supplier(self, x, y, other_feeds=None, name=None):
        """
        Return a standard feed dictionary for this dataset.

        :param name: if not None, register this supplier in dict NAMED_SUPPLIERS (this can be useful for instance
                        when recording with rf.Saver)
        :param x: placeholder for data
        :param y: placeholder for target
        :param other_feeds: optional other feeds (dictionary or None)
        :return: a callable.
        """
        if not other_feeds: other_feeds = {}

        # noinspection PyUnusedLocal
        def _supplier(step=None):
            """

            :param step: unused, just for making it compatible with `HG` and `Saver`
            :return: the feed dictionary
            """
            if isinstance(self.data, WindowedData):
                data = self.data.generate_all()

            return {**{x: self.data, y: self.target}, **other_feeds}

        if name:
            NAMED_SUPPLIER[name] = _supplier

        return _supplier

    @staticmethod
    def stack(*datasets):
        """
        Assuming that the datasets have same structure, stacks data, targets and other info

        :param datasets:
        :return: stacked dataset
        """
        return Dataset(data=vstack([d.data for d in datasets]),
                       target=stack_or_concat([d.target for d in datasets]),
                       sample_info=stack_or_concat([d.sample_info for d in datasets]),
                       info={k: [d.info.get(k, None) for d in datasets]
                             for k in merge_dicts(*[d.info for d in datasets])})


def stack_or_concat(list_of_arays):
    func = np.concatenate if list_of_arays[0].ndim == 1 else np.vstack
    return func(list_of_arays)


def vstack(lst):
    """
    Vstack that considers sparse matrices

    :param lst:
    :return:
    """
    import scipy.sparse as sc_sp
    import scipy as sp
    return sp.vstack(lst) if sp and isinstance(lst[0], sc_sp.csr.csr_matrix) else np.vstack(lst)


def pad(_example, _size): return np.concatenate([_example] * _size)


class WindowedData(object):
    def __init__(self, data, row_sentence_bounds, window=5, process_all=False):
        """
        Class for managing windowed input data (like TIMIT).

        :param data: Numpy matrix. Each row should be an example data
        :param row_sentence_bounds:  Numpy matrix with bounds for padding. TODO add default NONE
        :param window: half-window size
        :param process_all: (default False) if True adds context to all data at object initialization.
                            Otherwise the windowed data is created in runtime.
        """
        import intervaltree as it
        self.window = window
        self.data = data
        base_shape = self.data.shape
        self.shape = (base_shape[0], (2 * self.window + 1) * base_shape[1])
        self.tree = it.IntervalTree([it.Interval(int(e[0]), int(e[1]) + 1) for e in row_sentence_bounds])
        if process_all:
            print('adding context to all the dataset', end='- ')
            self.data = self.generate_all()
            print('DONE')
        self.process_all = process_all

    def generate_all(self):
        return self[:]

    def __getitem__(self, item):  # TODO should be right for all the common use... But better write down a TestCase
        if hasattr(self, 'process_all') and self.process_all:  # keep attr check!
            return self.data[item]
        if isinstance(item, int):
            return self.get_context(item=item)
        if isinstance(item, tuple):
            if len(item) == 2:
                rows, columns = item
                if isinstance(rows, int) and isinstance(columns, int):  # TODO check here
                    # do you want the particular element?
                    return self.get_context(item=rows)[columns]
            else:
                raise TypeError('NOT IMPLEMENTED <|>')
            if isinstance(rows, slice):
                rows = range(*rows.indices(self.shape[0]))
            return np.vstack([self.get_context(r) for r in rows])[:, columns]
        else:
            if isinstance(item, slice):
                item = range(*item.indices(self.shape[0]))
            return np.vstack([self.get_context(r) for r in item])

    def __len__(self):
        return self.shape[0]

    def get_context(self, item):
        interval = list(self.tree[item])[0]
        # print(interval)
        left, right = interval[0], interval[1]
        left_pad = max(self.window + left - item, 0)
        right_pad = max(0, self.window - min(right, len(self) - 1) + item)  # this is to cope with reduce datasets
        # print(left, right, item)

        # print(left_pad, right_pad)
        base = np.concatenate(self.data[item - self.window + left_pad: item + self.window + 1 - right_pad])
        if left_pad:
            base = np.concatenate([pad(self.data[item], left_pad), base])
        if right_pad:
            base = np.concatenate([base, pad(self.data[item], right_pad)])
        return base
