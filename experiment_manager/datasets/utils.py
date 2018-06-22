from experiment_manager.utils import *
import experiment_manager as em

import tensorflow as tf
import scipy.sparse as sc_sp
import scipy as sp


def redivide_data(datasets, partition_proportions=None, shuffle=False, filters=None,
                  maps=None, balance_classes=False, rand=None):
    """
    Function that redivides datasets. Can be use also to shuffle or filter or map examples.

    :param rand:
    :param balance_classes: # TODO RICCARDO
    :param datasets: original datasets, instances of class Dataset (works with get_data and get_targets for
                        compatibility with mnist datasets
    :param partition_proportions: (optional, default None)  list of fractions that can either sum up to 1 or less
                                    then one, in which case one additional partition is created with
                                    proportion 1 - sum(partition proportions).
                                    If None it will retain the same proportion of samples found in datasets
    :param shuffle: (optional, default False) if True shuffles the examples
    :param filters: (optional, default None) filter or list of filters: functions with signature
                        (data, target, index) -> boolean (accept or reject the sample)
    :param maps: (optional, default None) map or list of maps: functions with signature
                        (data, target, index) ->  (new_data, new_target) (maps the old sample to a new one,
                        possibly also to more
                        than one sample, for data augmentation)
    :return: a list of datasets of length equal to the (possibly augmented) partition_proportion
    """

    rnd = em.get_rand_state(rand)

    all_data = vstack([get_data(d) for d in datasets])
    all_labels = stack_or_concat([get_targets(d) for d in datasets])

    all_infos = np.concatenate([d.sample_info for d in datasets])

    N = all_data.shape[0]

    if partition_proportions:  # argument check
        partition_proportions = list([partition_proportions] if isinstance(partition_proportions, float)
                                     else partition_proportions)
        sum_proportions = sum(partition_proportions)
        assert sum_proportions <= 1, "partition proportions must sum up to at most one: %d" % sum_proportions
        if sum_proportions < 1.: partition_proportions += [1. - sum_proportions]
    else:
        partition_proportions = [1. * get_data(d).shape[0] / N for d in datasets]

    if shuffle:
        if sp and isinstance(all_data, sp.sparse.csr.csr_matrix): raise NotImplementedError()
        # if sk_shuffle:  # TODO this does not work!!! find a way to shuffle these matrices while
        # keeping compatibility with tensorflow!
        #     all_data, all_labels, all_infos = sk_shuffle(all_data, all_labels, all_infos)
        # else:
        permutation = np.arange(all_data.shape[0])
        rnd.shuffle(permutation)

        all_data = all_data[permutation]
        all_labels = np.array(all_labels[permutation])
        all_infos = np.array(all_infos[permutation])

    if filters:
        if sp and isinstance(all_data, sp.sparse.csr.csr_matrix): raise NotImplementedError()
        filters = as_list(filters)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for fiat in filters:
            data_triple = [xy for i, xy in enumerate(data_triple) if fiat(xy[0], xy[1], xy[2], i)]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    if maps:
        if sp and isinstance(all_data, sp.sparse.csr.csr_matrix): raise NotImplementedError()
        maps = as_list(maps)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for _map in maps:
            data_triple = [_map(xy[0], xy[1], xy[2], i) for i, xy in enumerate(data_triple)]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    N = all_data.shape[0]
    assert N == all_labels.shape[0]

    calculated_partitions = reduce(
        lambda v1, v2: v1 + [sum(v1) + v2],
        [int(N * prp) for prp in partition_proportions],
        [0]
    )
    calculated_partitions[-1] = N

    print('datasets.redivide_data:, computed partitions numbers -',
          calculated_partitions, 'len all', N, end=' ')

    new_general_info_dict = {}
    for data in datasets:
        new_general_info_dict = {**new_general_info_dict, **data.info}

        if balance_classes:
            new_datasets = []
            forbidden_indices = np.empty(0, dtype=np.int64)
            for d1, d2 in zip(calculated_partitions[:-1], calculated_partitions[1:-1]):
                indices = np.array(get_indices_balanced_classes(d2 - d1, all_labels, forbidden_indices))
                dataset = em.Dataset(data=all_data[indices], target=all_labels[indices],
                                  sample_info=all_infos[indices],
                                  info=new_general_info_dict)
                new_datasets.append(dataset)
                forbidden_indices = np.append(forbidden_indices, indices)
                test_if_balanced(dataset)
            remaining_indices = np.array(list(set(list(range(N))) - set(forbidden_indices)))
            new_datasets.append(em.Dataset(data=all_data[remaining_indices], target=all_labels[remaining_indices],
                                        sample_info=all_infos[remaining_indices],
                                        info=new_general_info_dict))
        else:
            new_datasets = [
                em.Dataset(data=all_data[d1:d2], target=all_labels[d1:d2], sample_info=all_infos[d1:d2],
                        info=new_general_info_dict)
                for d1, d2 in zip(calculated_partitions, calculated_partitions[1:])
                ]

        print('DONE')

        return new_datasets


def get_indices_balanced_classes(n_examples, labels, forbidden_indices):
    N = len(labels)
    n_classes = len(labels[0])

    indices = []
    current_class = 0
    for i in range(n_examples):
        index = np.random.random_integers(0, N - 1, 1)[0]
        while index in indices or index in forbidden_indices or np.argmax(labels[index]) != current_class:
            index = np.random.random_integers(0, N - 1, 1)[0]
        indices.append(index)
        current_class = (current_class + 1) % n_classes

    return indices


def test_if_balanced(dataset):
    labels = dataset.target
    n_classes = len(labels[0])
    class_counter = [0] * n_classes
    for l in labels:
        class_counter[np.argmax(l)] += 1
    print('exemple by class: ', class_counter)


def maybe_cast_to_scalar(what):
    return what[0] if len(what) == 1 else what


def pad(_example, _size): return np.concatenate([_example] * _size)


def stack_or_concat(list_of_arays):
    func = np.concatenate if list_of_arays[0].ndim == 1 else np.vstack
    return func(list_of_arays)


def vstack(lst):
    """
    Vstack that considers sparse matrices

    :param lst:
    :return:
    """
    return sp.vstack(lst) if sp and isinstance(lst[0], sc_sp.csr.csr_matrix) else np.vstack(lst)


def convert_sparse_matrix_to_sparse_tensor(X):
    if isinstance(X, sc_sp.csr.csr_matrix):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
    else:
        coo, indices = X, [X.row, X.col]
    # data = np.array(coo.data, dtype=)
    return tf.SparseTensor(indices, tf.constant(coo.data, dtype=tf.float32), coo.shape)


def get_data(d_set):
    if hasattr(d_set, 'images'):
        data = d_set.images
    elif hasattr(d_set, 'data'):
        data = d_set.data
    else:
        raise ValueError("something wrong with the dataset %s" % d_set)
    return data


def get_targets(d_set):
    if hasattr(d_set, 'labels'):
        return d_set.labels
    elif hasattr(d_set, 'target'):
        return d_set.target
    else:
        raise ValueError("something wrong with the dataset %s" % d_set)
