"""
Module for loading datasets
"""

from collections import defaultdict, OrderedDict
from os.path import join

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import experiment_manager as em
import os
import numpy as np
import threading
import h5py
import sys

from experiment_manager.datasets.utils import redivide_data

try:
    from sklearn.datasets import make_classification
except ImportError:
    make_classification = lambda *args: print('make_classification NOT LOADED, NEED SKLEARN', file=sys.stderr)

from_env = os.getenv('DATASETS_FOLDER')
if from_env:
    DATA_FOLDER = from_env
else:
    print('Environment variable DATASETS_FOLDER not found. Variables HELP_WIN and HELP_UBUNTU contain info.')
    DATA_FOLDER = os.getcwd()
    _COMMON_BEGIN = "You can set environment variable DATASETS_FOLDER to" \
                    "specify root folder in which you store various datasets. \n"
    _COMMON_END = """\n
    You can also skip this step... \n
    In this case all load_* methods take a FOLDER path as first argument. \n
    Bye."""
    HELP_UBUNTU = _COMMON_BEGIN + """
    Bash command is: export DATASETS_FOLDER='absolute/path/to/dataset/folder \n
    Remember! To add the global variable kinda permanently in your system you should add export command in
    bash.bashrc file located in etc folder, if you want to set it globally, or .bashrc in your home directory
    if you want to set it only locally.
    """ + _COMMON_END

    HELP_WIN = _COMMON_BEGIN + """
    Cmd command is: Set DATASETS_FOLDER absolute/path/to/dataset/folder  for one session. \n
    To set it permanently use SetX instead of Set (and probably reboot system)
    """ + _COMMON_END

print('Data folder is', DATA_FOLDER)

# kind of private
TIMIT_DIR = os.path.join(DATA_FOLDER, 'timit4python')
XRMB_DIR = os.path.join(DATA_FOLDER, 'XRMB')
IROS15_BASE_FOLDER = os.path.join(DATA_FOLDER, os.path.join('dls_collaboration', 'Learning'))

# easy to find!
IRIS_TRAINING = os.path.join(DATA_FOLDER, 'iris', "training.csv")
IRIS_TEST = os.path.join(DATA_FOLDER, 'iris', "test.csv")
MNIST_DIR = os.path.join(DATA_FOLDER, "mnist_data")
CALTECH101_30_DIR = os.path.join(DATA_FOLDER, "caltech101-30")
CALTECH101_DIR = os.path.join(DATA_FOLDER, "caltech")
CENSUS_TRAIN = os.path.join(DATA_FOLDER, 'census', "train.csv")
CENSUS_TEST = os.path.join(DATA_FOLDER, 'census', "test.csv")
CIFAR10_DIR = os.path.join(DATA_FOLDER, "CIFAR-10")
CIFAR100_DIR = os.path.join(DATA_FOLDER, "CIFAR-100")
REALSIM = os.path.join(DATA_FOLDER, "realsim")

# scikit learn datasets
SCIKIT_LEARN_DATA = os.path.join(DATA_FOLDER, 'scikit_learn_data')

IMAGENET_BASE_FOLDER = join(DATA_FOLDER, 'imagenet')
MINI_IMAGENET_FOLDER = join(DATA_FOLDER, join('imagenet', 'mini_v1'))
MINI_IMAGENET_FOLDER_RES84 = join(DATA_FOLDER, join('imagenet', 'mini_res84'))
MINI_IMAGENET_FOLDER_V2 = join(DATA_FOLDER, join('imagenet', 'mini_v2'))
MINI_IMAGENET_FOLDER_V3 = join(DATA_FOLDER, join('imagenet', 'mini_v3'))

OMNIGLOT_RESIZED = join(DATA_FOLDER, 'omniglot_resized')


def balanced_choice_wr(a, num, rand=None):
    rand = em.utils.get_rand_state(rand)
    lst = [len(a)] * (num // len(a)) + [num % len(a)]
    return np.concatenate(
        [rand.choice(a, size=(d,), replace=False) for d in lst]
    )


def mnist(folder=None, one_hot=True, partitions=None, filters=None, maps=None, shuffle=False):
    if not folder: folder = MNIST_DIR
    datasets = read_data_sets(folder, one_hot=one_hot)
    train = em.Dataset(datasets.train.images, datasets.train.labels, name='MNIST')
    validation = em.Dataset(datasets.validation.images, datasets.validation.labels, name='MNIST')
    test = em.Dataset(datasets.test.images, datasets.test.labels, name='MNIST')
    res = [train, validation, test]
    if partitions:
        res = redivide_data(res, partition_proportions=partitions, filters=filters, maps=maps, shuffle=shuffle)
        res += [None] * (3 - len(res))
    return em.Datasets.from_list(res)


def omni_light(folder=join(DATA_FOLDER, 'omniglot-light'), add_bias=False):
    """
    Extract from omniglot dataset with rotated images, 100 classes,
    3 examples per class in training set
    3 examples per class in validation set
    15 examples per class in test set
    """
    file = h5py.File(os.path.join(folder, 'omni-light.h5'), 'r')
    return em.Datasets.from_list([
        em.Dataset(np.array(file['X_ft_tr']), np.array(file['Y_tr']),
                   info={'original images': np.array(file['X_orig_tr'])}, add_bias=add_bias),
        em.Dataset(np.array(file['X_ft_val']), np.array(file['Y_val']),
                   info={'original images': np.array(file['X_orig_val'])}, add_bias=add_bias),
        em.Dataset(np.array(file['X_ft_test']), np.array(file['Y_test']),
                   info={'original images': np.array(file['X_orig_test'])}, add_bias=add_bias)
    ])


load_omni_light = omni_light


def meta_omniglot(folder=OMNIGLOT_RESIZED, std_num_classes=None, std_num_examples=None,
                  one_hot_enc=True, _rand=0, n_splits=None):
    """
    Loading function for Omniglot dataset in learning-to-learn version. Use image data as obtained from
    https://github.com/cbfinn/maml/blob/master/data/omniglot_resized/resize_images.py

    :param folder: root folder name.
    :param std_num_classes: standard number of classes for N-way classification
    :param std_num_examples: standard number of examples (e.g. for 1-shot 5-way should be 5)
    :param one_hot_enc: one hot encoding
    :param _rand: random seed or RandomState for generate training, validation, testing meta-datasets
                    split
    :param n_splits: num classes per split
    :return: a Datasets of MetaDataset s
    """
    class OmniglotMetaDataset(em.MetaDataset):

        def __init__(self, info=None, rotations=None, name='Omniglot', num_classes=None, num_examples=None):
            super().__init__(info, name=name, num_classes=num_classes, num_examples=num_examples)
            self._loaded_images = defaultdict(lambda: {})
            self._rotations = rotations or [0, 90, 180, 270]
            self.load_all()

        def generate_datasets(self, rand=None, num_classes=None, num_examples=None):
            rand = em.get_rand_state(rand)

            if not num_examples: num_examples = self.kwargs['num_examples']
            if not num_classes: num_classes = self.kwargs['num_classes']

            clss = self._loaded_images if self._loaded_images else self.info['classes']

            random_classes = rand.choice(list(clss.keys()), size=(num_classes,), replace=False)
            rand_class_dict = {rnd: k for k, rnd in enumerate(random_classes)}

            _dts = []
            for ns in em.as_tuple_or_list(num_examples):
                classes = balanced_choice_wr(random_classes, ns, rand)

                all_images = {cls: list(clss[cls]) for cls in classes}
                data, targets, sample_info = [], [], []
                for c in classes:
                    rand.shuffle(all_images[c])
                    img_name = all_images[c][0]
                    all_images[c].remove(img_name)
                    sample_info.append({'name': img_name, 'label': c})
                    data.append(clss[c][img_name])
                    targets.append(rand_class_dict[c])

                if self.info['one_hot_enc']:
                    targets = em.to_one_hot_enc(targets, dimension=num_classes)

                _dts.append(em.Dataset(data=np.array(np.stack(data)), target=targets, sample_info=sample_info,
                                       info={'all_classes': random_classes}))
            return em.Datasets.from_list(_dts)

        def load_all(self):
            from scipy.ndimage import imread
            from scipy.ndimage.interpolation import rotate
            _cls = self.info['classes']
            _base_folder = self.info['base_folder']

            for c in _cls:
                all_images = list(_cls[c])
                for img_name in all_images:
                    img = imread(join(_base_folder, join(c, img_name)))
                    img = 1. - np.reshape(img, (28, 28, 1)) / 255.
                    for rot in self._rotations:
                        img = rotate(img,  rot, reshape=False)
                        self._loaded_images[c + os.path.sep + 'rot_' + str(rot)][img_name] = img

    alphabets = os.listdir(folder)

    labels_and_images = OrderedDict()
    for alphabet in alphabets:
        base_folder = join(folder, alphabet)
        label_names = os.listdir(base_folder)  # all characters in one alphabet
        labels_and_images.update({alphabet + os.path.sep + ln: os.listdir(join(base_folder, ln))
                                  # all examples of each character
                                  for ln in label_names})

    # divide between training validation and test meta-datasets
    _rand = em.get_rand_state(_rand)
    all_clss = list(labels_and_images.keys())
    _rand.shuffle(all_clss)
    n_splits = n_splits or (0, 1200, 1300, len(all_clss))

    meta_dts = []
    for start, end in zip(n_splits, n_splits[1:]):
        meta_dts.append(OmniglotMetaDataset(info={
            'base_folder': folder,
            'classes': {k: labels_and_images[k] for k in all_clss[start: end]},
            'one_hot_enc': one_hot_enc
        }, num_classes=std_num_classes, num_examples=std_num_examples))

    return em.Datasets.from_list(meta_dts)


def meta_omniglot_v2(folder=OMNIGLOT_RESIZED, std_num_classes=None, std_num_examples=None,
                  one_hot_enc=True, _rand=0, n_splits=None):
    """
    Loading function for Omniglot dataset in learning-to-learn version. Use image data as obtained from
    https://github.com/cbfinn/maml/blob/master/data/omniglot_resized/resize_images.py

    :param folder: root folder name.
    :param std_num_classes: standard number of classes for N-way classification
    :param std_num_examples: standard number of examples (e.g. for 1-shot 5-way should be 5)
    :param one_hot_enc: one hot encoding
    :param _rand: random seed or RandomState for generate training, validation, testing meta-datasets
                    split
    :param n_splits: num classes per split
    :return: a Datasets of MetaDataset s
    """
    class OmniglotMetaDataset(em.MetaDataset):

        def __init__(self, info=None, rotations=None, name='Omniglot', num_classes=None, num_examples=None):
            super().__init__(info, name=name, num_classes=num_classes, num_examples=num_examples)
            self._loaded_images = defaultdict(lambda: {})
            self._rotations = rotations or [0, 90, 180, 270]
            self._img_array = None
            self.load_all()

        def generate_datasets(self, rand=None, num_classes=None, num_examples=None):
            rand = em.get_rand_state(rand)

            if not num_examples: num_examples = self.kwargs['num_examples']
            if not num_classes: num_classes = self.kwargs['num_classes']

            clss = self._loaded_images if self._loaded_images else self.info['classes']

            random_classes = rand.choice(list(clss.keys()), size=(num_classes,), replace=False)
            rand_class_dict = {rnd: k for k, rnd in enumerate(random_classes)}

            _dts = []
            for ns in em.as_tuple_or_list(num_examples):
                classes = balanced_choice_wr(random_classes, ns, rand)

                all_images = {cls: list(clss[cls]) for cls in classes}
                indices, targets, sample_info = [], [], []
                for c in classes:
                    rand.shuffle(all_images[c])
                    img_name = all_images[c][0]
                    all_images[c].remove(img_name)
                    sample_info.append({'name': img_name, 'label': c})
                    indices.append(clss[c][img_name])
                    targets.append(rand_class_dict[c])

                if self.info['one_hot_enc']:
                    targets = em.to_one_hot_enc(targets, dimension=num_classes)

                print(self._img_array)
                data = self._img_array[indices]

                _dts.append(em.Dataset(data=data, target=targets, sample_info=sample_info,
                                       info={'all_classes': random_classes}))
            return em.Datasets.from_list(_dts)

        def load_all(self):
            from scipy.ndimage import imread
            from scipy.ndimage.interpolation import rotate
            _cls = self.info['classes']
            _base_folder = self.info['base_folder']

            _id = 0
            flat_data = []
            # flat_targets = []
            for c in _cls:
                all_images = list(_cls[c])
                for img_name in all_images:
                    img = imread(join(_base_folder, join(c, img_name)))
                    img = 1. - np.reshape(img, (28, 28, 1)) / 255.
                    for rot in self._rotations:
                        img = rotate(img,  rot, reshape=False)
                        self._loaded_images[c + os.path.sep + 'rot_' + str(rot)][img_name] = _id
                        _id += 1
                        flat_data.append(img)
                        # flat_targets maybe...

            self._img_array = np.stack(flat_data)

        # end of class

    alphabets = os.listdir(folder)

    labels_and_images = OrderedDict()
    for alphabet in alphabets:
        base_folder = join(folder, alphabet)
        label_names = os.listdir(base_folder)  # all characters in one alphabet
        labels_and_images.update({alphabet + os.path.sep + ln: os.listdir(join(base_folder, ln))
                                  # all examples of each character
                                  for ln in label_names})

    # divide between training validation and test meta-datasets
    _rand = em.get_rand_state(_rand)
    all_clss = list(labels_and_images.keys())
    _rand.shuffle(all_clss)
    n_splits = n_splits or (0, 1200, 1300, len(all_clss))

    meta_dts = []
    for start, end in zip(n_splits, n_splits[1:]):
        meta_dts.append(OmniglotMetaDataset(info={
            'base_folder': folder,
            'classes': {k: labels_and_images[k] for k in all_clss[start: end]},
            'one_hot_enc': one_hot_enc
        }, num_classes=std_num_classes, num_examples=std_num_examples))

    return em.Datasets.from_list(meta_dts)


def meta_mini_imagenet(folder=MINI_IMAGENET_FOLDER_V3, sub_folders=None, std_num_classes=None,
                       std_num_examples=None, resize=84, one_hot_enc=True, load_all_images=True, h5=True):
    """
    Load a meta-datasets from Mini-ImageNet. Returns a Datasets of MetaDataset s,

    :param folder: base folder
    :param sub_folders: optional sub-folders in which data is locate
    :param std_num_classes: standard number of classes to be included in each generated per dataset
                            (can be None, default)
    :param std_num_examples: standard number of examples to be picked in each generated per dataset
                            (can be None, default)
    :param resize:  resizing dimension
    :param one_hot_enc:
    :param load_all_images:
    :param h5:  True (default) to use HDF5 files, when False search for JPEG images.
    :return:
    """

    class ImageNetMetaDataset(em.MetaDataset):

        def __init__(self, info=None, name='MiniImagenet', num_classes=None, num_examples=None):
            super().__init__(info, name=name, num_classes=num_classes, num_examples=num_examples)
            self._loaded_images = defaultdict(lambda: {})
            self._threads = []

        def load_all_images(self):
            if h5:
                _file = self.info['file']
                h5m = h5py.File(_file, 'r')
                img_per_class = 600
                for j in range(len(h5m['X'])):
                    self._loaded_images[j // img_per_class][j] = np.array(h5m['X'][j], dtype=np.float32) / 255.
                    # images were stored as int
            else:
                from scipy.ndimage import imread
                from scipy.misc import imresize
                _cls = self.info['classes']
                _base_folder = self.info['base_folder']

                def _load_class(c):
                    all_images = list(_cls[c])
                    for img_name in all_images:
                        img = imread(join(_base_folder, join(c, img_name)), mode='RGB')
                        if self.info['resize']:
                            # noinspection PyTypeChecker
                            img = imresize(img, size=(self.info['resize'], self.info['resize'], 3)) / 255.
                        self._loaded_images[c][img_name] = img

                for cls in _cls:
                    self._threads.append(threading.Thread(target=lambda: _load_class(cls), daemon=True))
                    self._threads[-1].start()

        def check_loaded_images(self, n_min):
            return self._loaded_images and all([len(v) >= n_min for v in self._loaded_images.values()])

        def generate_datasets(self, rand=None, num_classes=None, num_examples=None, wait_for_n_min=None):

            rand = em.get_rand_state(rand)

            if wait_for_n_min:
                import time
                while not self.check_loaded_images(wait_for_n_min):
                    time.sleep(5)

            if not num_examples: num_examples = self.kwargs['num_examples']
            if not num_classes: num_classes = self.kwargs['num_classes']

            clss = self._loaded_images if self._loaded_images else self.info['classes']

            random_classes = rand.choice(list(clss.keys()), size=(num_classes,), replace=False)
            rand_class_dict = {rnd: k for k, rnd in enumerate(random_classes)}

            _dts = []
            for ns in em.as_tuple_or_list(num_examples):
                classes = balanced_choice_wr(random_classes, ns, rand)

                all_images = {cls: list(clss[cls]) for cls in classes}
                data, targets, sample_info = [], [], []
                for c in classes:
                    rand.shuffle(all_images[c])
                    img_name = all_images[c][0]
                    all_images[c].remove(img_name)
                    sample_info.append({'name': img_name, 'label': c})

                    if self._loaded_images:
                        data.append(clss[c][img_name])
                    else:
                        from scipy.misc import imread, imresize
                        data.append(
                            imresize(
                                imread(join(self.info['base_folder'], join(c, img_name)), mode='RGB'),
                                size=(self.info['resize'], self.info['resize'], 3)) / 255.
                        )
                    targets.append(rand_class_dict[c])

                if self.info['one_hot_enc']:
                    targets = em.to_one_hot_enc(targets, dimension=num_classes)

                _dts.append(em.Dataset(data=np.array(np.stack(data)), target=targets, sample_info=sample_info,
                                       info={'all_classes': random_classes}))
            return em.Datasets.from_list(_dts)

        def all_data(self, partition_proportions=None, seed=None):
            if not self._loaded_images:
                self.load_all_images()
                while not self.check_loaded_images(600):
                    time.sleep(5)
            data, targets = [], []
            for k, c in enumerate(sorted(self._loaded_images)):
                data += list(self._loaded_images[c].values())
                targets += [k] * 600
            if self.info['one_hot_enc']:
                targets = em.to_one_hot_enc(targets, dimension=len(self._loaded_images))
            _dts = [em.Dataset(data=np.stack(data), target=np.array(targets), name='MiniImagenet_full')]
            if seed:
                np.random.seed(seed)
            if partition_proportions:
                _dts = redivide_data(_dts, partition_proportions=partition_proportions,
                                     shuffle=True)
            return em.Datasets.from_list(_dts)

    if sub_folders is None: sub_folders = ['train', 'val', 'test']
    meta_dts = []
    for ds in sub_folders:
        if not h5:
            base_folder = join(folder, ds)
            label_names = os.listdir(base_folder)
            labels_and_images = {ln: os.listdir(join(base_folder, ln)) for ln in label_names}
            meta_dts.append(ImageNetMetaDataset(info={
                'base_folder': base_folder,
                'classes': labels_and_images,
                'resize': resize,
                'one_hot_enc': one_hot_enc
            }, num_classes=std_num_classes, num_examples=std_num_examples))
        else:
            file = join(folder, ds + '.h5')
            meta_dts.append(ImageNetMetaDataset(info={
                'file': file,
                'one_hot_enc': one_hot_enc
            }, num_classes=std_num_classes, num_examples=std_num_examples))

    dts = em.Datasets.from_list(meta_dts)
    if load_all_images:
        import time
        [_d.load_all_images() for _d in dts]
        _check_available = lambda min_num: [_d.check_loaded_images(min_num) for _d in dts]
        while not all(_check_available(15)):
            time.sleep(1)  # be sure that there are at least 15 images per class in each meta-dataset
    return dts


def random_classification_datasets(examples=(100, 100, 100), features=20, classes=5, rnd=None):
    rnd = em.get_rand_state(rnd)

    def _form_dataset(scp_dat):
        return em.Dataset(scp_dat[0], em.utils.to_one_hot_enc(scp_dat[1]))

    return em.Datasets.from_list([
        _form_dataset(make_classification(ne, features, n_classes=classes, n_informative=classes, random_state=rnd))
        for ne in em.utils.as_tuple_or_list(examples)
    ])


if __name__ == '__main__':
    # pass
    # mmi = meta_mini_imagenet()
    # # ts = mmi.train.generate_datasets(num_classes=10, num_examples=(123, 39))
    # d1 = mmi.train.all_data(seed=0)
    # print(d1.train.dim_data, d1.train.dim_target)
    #
    # mmiii = meta_mini_imagenet()
    # # ts = mmi.train.generate_datasets(num_classes=10, num_examples=(123, 39))
    # d2 = mmiii.train.all_data(seed=0)
    # print(d2.train.dim_data, d2.train.dim_target)
    #
    # print(np.equal(d1.train.data[0], d2.train.data[0]))

    res = meta_omniglot_v2(std_num_classes=5, std_num_examples=(10, 20))
    dt = res.train.generate_datasets()
    print(dt.train.data)
    # print(res)
