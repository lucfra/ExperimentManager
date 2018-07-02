import glob
import sys

import tensorflow as tf

from experiment_manager.utils import as_list
import time
from collections import OrderedDict, defaultdict
from functools import reduce
from inspect import signature
from contextlib import redirect_stdout, contextmanager

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from IPython.display import IFrame
    import IPython
except:
    print('Looks like IPython is not installed...')
    IFrame, IPython = None, None
import gzip
import os
import _pickle as pickle
import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    print('Might want to install library "tabulate" for a better dictionary printing')
    tabulate = None


def join_paths(*paths):
    return reduce(lambda acc, new_path: os.path.join(acc, new_path), paths)


SAVE_SETTINGS = {
    'NOTEBOOK_TITLE': ''
}

EXP_ROOT_FOLDER = os.getenv('EXPERIMENTS_FOLDER')
if EXP_ROOT_FOLDER is None:
    print('Environment variable EXPERIMENTS_FOLDER not found. Current directory will be used')
    EXP_ROOT_FOLDER = join_paths(os.getcwd(), 'Experiments')
print('Experiment save directory is ', EXP_ROOT_FOLDER)

FOLDER_NAMINGS = {  # TODO should go into a settings file?
    'EXP_ROOT': EXP_ROOT_FOLDER,
    'OBJ_DIR': 'Obj_data',
    'PLOTS_DIR': 'Plots',
    'MODELS_DIR': 'Models',
    'GEPHI_DIR': 'GePhi',
}


def check_or_create_dir(directory, notebook_mode=True, create=True):
    if not os.path.exists(directory) and create:
        os.mkdir(directory)
        print('folder', directory, 'has been created')

    if notebook_mode and SAVE_SETTINGS['NOTEBOOK_TITLE']:
        directory = join_paths(directory, SAVE_SETTINGS['NOTEBOOK_TITLE'])  # += '/' + settings['NOTEBOOK_TITLE']
        if not os.path.exists(directory) and create:
            try:
                os.mkdir(directory)
            except FileExistsError as e:
                print(e, file=sys.stderr)
            print('folder ', directory, 'has been created')
    return directory


def save_fig(name=None, root_dir=None, notebook_mode=True, default_overwrite=False, extension='pdf', **savefig_kwargs):
    if name is None:
        name = plt.gca().get_title()
    assert name is not None, 'no title found in the current axis, set it manually!'

    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['PLOTS_DIR']),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s.%s' % (name, extension))  # directory + '/%s.pdf' % name
    if not default_overwrite and os.path.isfile(filename):
        # if IPython is not None:
        #     IPython.display.display(tuple(IFrame(filename, width=800, height=600)))  # FIXME not working!
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
    # some modified kwargs (like tight)
    savefig_kwargs.setdefault('bbox_inches', 'tight')

    plt.savefig(filename, **savefig_kwargs)
    # print('file saved')


def save_obj(obj, name, root_dir=None, notebook_mode=True, default_overwrite=False, ext='pkgz'):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['OBJ_DIR']),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '{}.{}'.format(name, ext))  # directory + '/%s.pkgz' % name
    if not default_overwrite and os.path.isfile(filename):
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
        print('Overwriting...')
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)
        # print('File saved!')


def save_text(text, name, root_dir=None, notebook_mode=True, default_overwrite=False):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s.txt' % name)  # directory + '/%s.pkgz' % name
    if not default_overwrite and os.path.isfile(filename):
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
        print('Overwriting...')

    with open(filename, "w") as text_file:
        text_file.write(text)


def pkgz_load(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def load_obj(name, root_dir=None, notebook_mode=True):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['OBJ_DIR']),
                                    notebook_mode=notebook_mode, create=False)

    filename = join_paths(directory, name if name.endswith('.pkgz') or name.endswith('pkexp') else name + '.pkgz')
    return pkgz_load(filename)


def load_text(name, root_dir=None, notebook_mode=True):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(root_dir,
                                    notebook_mode=notebook_mode, create=False)

    filename = join_paths(directory, name if name.endswith('.txt') else name + '.txt')
    with open(filename, 'r') as f:
        return f.read()


def save_model(model, step=None, session=None, root_dir=None, notebook_mode=True):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['MODELS_DIR']),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s' % model.name)
    model.save(filename, session=session, global_step=step)


def restore_model(model, step=None, session=None, root_dir=None, notebook_mode=True):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['MODELS_DIR']),
                                    notebook_mode=notebook_mode, create=False)

    filename = join_paths(directory, model.name)
    model.restore(filename, session=session, global_step=step)


def save_adjacency_matrix_for_gephi(matrix, name, root_dir=None, notebook_mode=True, class_names=None):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['GEPHI_DIR']),
                                    notebook_mode=notebook_mode)
    filename = join_paths(directory, '%s.csv' % name)

    m, n = np.shape(matrix)
    assert m == n, '%s should be a square matrix.' % matrix
    if not class_names:
        class_names = [str(k) for k in range(n)]

    left = np.array([class_names]).T
    matrix = np.hstack([left, matrix])
    up = np.vstack([[''], left]).T
    matrix = np.vstack([up, matrix])

    np.savetxt(filename, matrix, delimiter=';', fmt='%s')


def save_setting(local_variables, root_dir=None, excluded=None, default_overwrite=False, collect_data=True,
                 notebook_mode=True, do_print=True, append_string=''):
    dictionary = generate_setting_dict(local_variables, excluded=excluded)
    if do_print:
        if tabulate:
            print(tabulate(dictionary.items(), headers=('settings var names', 'values')))
        else:
            print('SETTING:')
            for k, v in dictionary.items():
                print(k, v, sep=': ')
            print()
    if collect_data: save_obj(dictionary, 'setting' + append_string,
                              root_dir=root_dir,
                              default_overwrite=default_overwrite,
                              notebook_mode=notebook_mode)


def generate_setting_dict(local_variables, excluded=None):
    """
    Generates a dictionary of (name, values) of local variables (typically obtained by vars()) that
    can be saved at the beginning of the experiment. Furthermore, if an object obj in local_variables implements the
    function setting(), it saves the result of obj.setting() as value in the dictionary.

    :param local_variables:
    :param excluded: (optional, default []) variable or list of variables to be excluded.
    :return: A dictionary
    """
    excluded = as_list(excluded) or []
    setting_dict = {k: v.setting() if hasattr(v, 'setting') else v
                    for k, v in local_variables.items() if v not in excluded}
    import datetime
    setting_dict['datetime'] = str(datetime.datetime.now())
    return setting_dict


class Timer:
    """
    Stopwatch class for timing the experiments. Uses `time` module.
    """

    _div_unit = {'ms': 1. / 1000,
                 'sec': 1.,
                 'min': 60.,
                 'hr': 3600.}

    def __init__(self, unit='sec', round_off=True):
        self._starting_times = []
        self._stopping_times = []
        self._running = False
        self.round_off = round_off
        assert unit in Timer._div_unit
        self.unit = unit

    def reset(self):
        self._starting_times = []
        self._stopping_times = []
        self._running = False

    def start(self):
        if not self._running:
            self._starting_times.append(time.time())
            self._running = True
        return self

    def stop(self):
        if self._running:
            self._stopping_times.append(time.time())
            self._running = False
        return self

    def raw_elapsed_time_list(self):
        def _maybe_add_last():
            t2 = self._stopping_times if len(self._starting_times) == len(self._stopping_times) else \
                self._stopping_times + [time.time()]
            return zip(self._starting_times, t2)

        return [t2 - t1 for t1, t2 in _maybe_add_last()]

    def elapsed_time(self):
        res = sum(self.raw_elapsed_time_list()) / Timer._div_unit[self.unit]
        return res if not self.round_off else int(res)


class Saver:
    """
    Class for recording experiment data
    """

    SKIP = 'SKIP'  # skip saving value in save_dict

    @classmethod
    def std(cls, *names, description=False):
        """
        Returns a function that builds a standard Saver, which has filepath given by names and to which is optionally
        possible to append the name of a named object (that is an object that has a filed name) to it.

        :param names: varargs for subfolders.
        :param description: (optional, defualt False) True for input description, str for providing a fixed description
        :return: a function with signature (string | named_object) -> Saver
        """
        nm = lambda _n: _n if isinstance(_n, str) else _n.name
        # def _call(*named_objects):

        return lambda *named_obj: Saver([n for n in names] + [nm(obj) for obj in named_obj],
                                        append_date_to_name=False,
                                        ask_for_description=description,
                                        default_overwrite=True)

    def __init__(self, experiment_names, *items, append_date_to_name=True,
                 root_directory=FOLDER_NAMINGS['EXP_ROOT'],
                 timer=None, do_print=True, collect_data=True, default_overwrite=False,
                 ask_for_description=True):
        """
        Initialize a saver to collect data. (Intended to be used together with OnlinePlotStream.)

        :param experiment_names: string or list of strings which represent the name of the folder (and sub-folders)
                                    experiment oand
        :param items: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
                      The first arg of each tuple should be a string that will be the key of the save_dict.
                      Then there can be either a callable with signature (step) -> None
                      Should pass the various args in ths order:
                          fetches: tensor or list of tensors to compute;
                          feeds (optional): to be passed to tf.Session.run. Can be a
                          callable with signature (step) -> feed_dict
                          options (optional): to be passed to tf.Session.run
                          run_metadata (optional): to be passed to tf.Session.run
        :param timer: optional timer object. If None creates a new one. If false does not register time.
                        If None or Timer it adds to the save_dict an entry time that record_hyper elapsed_time.
                        The time required to perform data collection and saving are not counted, since typically
                        the aim is to record_hyper the true algorithm execution time!
        :param root_directory: string, name of root directory (default ~HOME/Experiments)
        :param do_print: (optional, default True) will print by default `save_dict` each time method `save` is executed
        :param collect_data: (optional, default True) will save by default `save_dict` each time
                            method `save` is executed
        """
        self.last_record = None
        if ask_for_description is True:
            description = input('Experiment description:')
            if description == 'SKIP':
                collect_data = False
        elif isinstance(ask_for_description, str):
            description = str(ask_for_description)
        else:
            description = None

        experiment_names = as_list(experiment_names)
        if append_date_to_name:
            from datetime import datetime
            experiment_names += [str(int(time.time())) + '__' + datetime.today().strftime('%d-%m-%y__%Hh%Mm')]
        self.experiment_names = list(experiment_names)

        if not os.path.isabs(experiment_names[0]):
            self.directory = join_paths(root_directory)  # otherwise assume no use of root_directory
            if collect_data:
                check_or_create_dir(root_directory, notebook_mode=False)
        else:
            self.directory = ''
        for name in self.experiment_names:
            self.directory = join_paths(self.directory, name)
            check_or_create_dir(self.directory, notebook_mode=False, create=collect_data)

        self.do_print = do_print
        self.collect_data = collect_data
        self.default_overwrite = default_overwrite

        assert isinstance(timer, Timer) or timer is None or timer is False, 'timer param not good...'

        if timer is None:
            timer = Timer()

        self.timer = timer

        self.clear_items()

        self.add_items(*items)

        if description and self.collect_data:
            self.save_text(description, 'description')

    @staticmethod
    def name_from_vars(*base_names, loc_vars, exclude=None):
        """
        Creates a name as list of string from base_names and local variables,
        excludes automatically modules from local variables.

        :param base_names:
        :param loc_vars:
        :param exclude:
        :return:
        """
        # noinspection PyUnresolvedReferences
        from types import ModuleType

        if exclude is None: exclude = []
        if not isinstance(exclude, list): exclude = [exclude]
        return list(base_names) + [n + str(v) for n, v in sorted(loc_vars.items())
                                   if v not in exclude and not isinstance(v, ModuleType)]

    # noinspection PyAttributeOutsideInit
    def clear_items(self):
        """
        Removes all previously inserted items

        :return:
        """
        self._processed_items = []
        self._step = -1

    @staticmethod
    def process_items(*items):
        """
        Add items to the save dictionary

        :param items: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
                      The first arg of each tuple should be a string that will be the key of the save_dict.
                      Then there can be either a callable with signature (step) -> result or () -> result
                      or tensorflow things... In this second case you  should pass the following args in ths order:
                          fetches: tensor or list of tensors to compute;
                          feeds (optional): to be passed to tf.Session.run. Can be a
                                            callable with signature (step) -> feed_dict
                                            or () -> feed_dict or `str`. If `str` then search for a registered supplier
                                            in datasets.NAMED_SUPPLIERS
                          options (optional): to be passed to tf.Session.run
                          run_metadata (optional): to be passed to tf.Session.run
        :return: None
        """
        assert len(items) == 0 or isinstance(items[0], str), 'Check items! first arg %s. Should be a string.' \
                                                             'All args: %s' % (items[0], items)

        processed_args = []
        k = 0
        while k < len(items):
            part = [items[k]]
            k += 1
            while k < len(items) and not isinstance(items[k], str):
                part.append(items[k])
                k += 1
            assert len(part) >= 2, 'Check args! Last part %s' % part
            if callable(part[1]):  # python stuff
                if len(part) == 2: part.append(True)  # always true default condition
            else:  # tensorflow stuff
                part += [None] * (6 - len(part))  # representing name, fetches, feeds, options, metadata
                if part[3] is None: part[3] = True  # default condition
            processed_args.append(part)
        # self._processed_items += processed_args
        # return [pt[0] for pt in processed_args]
        return processed_args

    def add_items(self, *items):
        """
        Adds internally items to this saver

        :param items:
        :return:
        """
        processed_items = Saver.process_items(*items)
        self._processed_items += processed_items
        return [pt[0] for pt in processed_items]

    def save(self, step=None, session=None, append_string="", do_print=None, collect_data=None,
             processed_items=None, _res=None):
        """
        Builds and save a dictionary with the keys and values specified at construction time or by method
        `add_items`

        :param processed_items: optional, processed item list (returned by add_items)
                                if None uses internally stored items
        :param session: Optional tensorflow session, otherwise uses default session
        :param step: optional step, if None (default) uses internal step
                        (int preferred, otherwise does not work well with `pack_save_dictionaries`).
        :param append_string: (optional str) string to append at the file name to `str(step)`
        :param do_print: (default as object field)
        :param collect_data: (default as object field)
        :param _res: used internally by context managers

        :return: the dictionary
        """
        import tensorflow
        from tensorflow import get_default_session
        if step is None:
            self._step += 1
            step = self._step
        if not processed_items: processed_items = self._processed_items

        if do_print is None: do_print = self.do_print
        if collect_data is None: collect_data = self.collect_data

        if session:
            ss = session
        else:
            ss = get_default_session()
        if ss is None and do_print: print('WARNING, No tensorflow session available')

        if self.timer: self.timer.stop()

        def _maybe_call(_method, _partial_save_dict):  # this is a bit crazy...
            if not callable(_method): return _method
            if len(signature(_method).parameters) == 0:
                _out = _method()
            elif len(signature(_method).parameters) == 1:
                _out = _method(step)
            elif len(signature(_method).parameters) == 2:
                _out = _method(step, _res)  # internal usage...
            elif len(signature(_method).parameters) == 3:
                _out = _method(step, _res, self)
            else:
                _out = _method(step, _res, self, _partial_save_dict)
            return _maybe_call(_out, _partial_save_dict) if callable(_out) else _out

        def _tf_run_catch_not_initialized(_pt, _partial_save_dict):

            try:
                return ss.run(_pt[1], feed_dict=_maybe_call(_pt[2], _partial_save_dict),
                              options=_pt[4], run_metadata=_pt[5])
            except tensorflow.errors.FailedPreconditionError:
                return 'Not initialized'
            except KeyError:
                return 'Feed dict not found'

        def _compute_value(pt, _partial_save_dict):
            if _maybe_call(pt[2 if callable(pt[1]) else 3], _partial_save_dict):
                return _maybe_call(pt[1], _partial_save_dict) \
                    if callable(pt[1]) else _tf_run_catch_not_initialized(pt, _partial_save_dict)
            else:
                return Saver.SKIP

        def _unnest(_pt, res_):
            if isinstance(res_[0], str):
                # print(res_[0])
                if _pt[0] == 'FLAT':
                    # print('HERE!!!!')
                    save_dict[res_[0]] = res_[1]
                else:
                    save_dict[_pt[0] + '::' + res_[0]] = res_[1]
            elif isinstance(res_[0], (list, tuple)):
                for _r in res_: _unnest(_pt, _r)
            else:
                save_dict[pt[0]] = res_

        save_dict = OrderedDict()
        for pt in processed_items:
            rss = _compute_value(pt, save_dict)
            if isinstance(rss, (list, tuple)):
                _unnest(pt, rss)
            else:
                save_dict[pt[0]] = rss

        if self.timer: save_dict['SKIP::Elapsed time (%s)' % self.timer.unit] = self.timer.elapsed_time()

        if do_print:
            if tabulate:
                print(tabulate([(k.replace('SKIP::', ''), v) for k, v in save_dict.items() if not k.startswith('HIDE')
                                ],
                               headers=('Step %s' % step, '/'.join(self.experiment_names)),
                               floatfmt='.5f'))
                print()
            else:
                print('SAVE DICT:')
                for key, v in save_dict.items():
                    print(key, v, sep=': ')
                print()
        if collect_data:
            self.save_obj(save_dict, str(step) + append_string)

        if self.timer: self.timer.start()

        self.last_record = save_dict

        return save_dict

    def pack_save_dictionaries(self, name='all', append_string='', erase_others=True, save_packed=True):
        """
        Creates an unique file starting from file created by method `save`.
        The file contains a dictionary with keys equal to save_dict keys and values list of values form original files.

        :param name:
        :param append_string:
        :param erase_others:
        :return: The generated dictionary
        """
        import glob
        all_files = sorted(glob.glob(join_paths(
            self.directory, FOLDER_NAMINGS['OBJ_DIR'], '[0-9]*%s.pkgz' % append_string)),
            key=os.path.getctime)  # sort by creation time
        if len(all_files) == 0:
            print('No file found')
            return

        objs = [load_obj(path, root_dir='', notebook_mode=False) for path in all_files]

        # packed_dict = OrderedDict([(k, []) for k in objs[0]])

        # noinspection PyArgumentList
        packed_dict = defaultdict(list, OrderedDict())
        for obj in objs:
            [packed_dict[k].append(v) for k, v in obj.items()]
        if save_packed:
            _nm = append_string if append_string != '' else name
            self.save_obj(packed_dict, name=_nm, ext='pkexp')

        if erase_others:
            [os.remove(f) for f in all_files]

        return packed_dict

    def load_all_packed(self):
        """
        :return: An ordered dictionary {str all_<experiment_name>: loaded records} from all files generated by
                `pack_save_dictionary` ordered by creation time.
        """
        # TODO maybe use os.sep ?
        alls = sorted([e for e in glob.glob(self.directory + '/Obj_data/*.pkexp')], key=os.path.getctime)
        return OrderedDict(
            [('.'.join(_a.split('/')[-1].split('.')[:-1]), self.load_obj(_a)) for _a in alls])

    def record(self, *what, where='hyper', append_string='', every=1):
        # idea [getting better].
        """
        Context manager for saver. saves statistics at the end of every `where` (default hyperiteration)

        :param where:
        :param every:
        :param what:
        :param append_string:
        :return:
        """
        from experiment_manager.savers import records
        associations = {'hyper': records.on_hyperiteration,
                        'run': records.on_run,
                        # 'forward': records.on_forward,
                        'far': records.on_far}
        assert where in associations, 'param where must be one of %s' % list(associations.keys())
        return associations[where](self, *what, append_string=append_string, every=every)

    def load_text(self, name):
        """
        Loads a text file

        :param name:
        :return:
        """
        return load_text(name, root_dir=self.directory, notebook_mode=False)

    def save_text(self, text, name):
        """
        Simply saves text into a .txt file (.txt extension automatically added).
        :param text:
        :param name:
        :return:
        """
        return save_text(text=text, name=name, root_dir=self.directory, default_overwrite=self.default_overwrite,
                         notebook_mode=False)

    def save_fig(self, name=None, extension='pdf', **savefig_kwargs):
        """
        Object-oriented version of `save_fig`

        :param extension:
        :param name: name of the figure (.pdf extension automatically added)
        :return:
        """
        return save_fig(name, root_dir=self.directory, extension=extension,
                        default_overwrite=self.default_overwrite, notebook_mode=False,
                        **savefig_kwargs)

    def save_obj(self, obj, name, ext='pkgz'):
        """
         Object-oriented version of `save_obj`

        :param ext: extension name, default pkgz
        :param obj: object to save
        :param name: name of the file (extension automatically added)
        :return:
        """
        return save_obj(obj, name, root_dir=self.directory,
                        default_overwrite=self.default_overwrite, notebook_mode=False, ext=ext)

    def save_adjacency_matrix_for_gephi(self, matrix, name, class_names=None):
        """
        Object-oriented version of `save_adjacency_matrix_for_gephi`

        :param matrix:
        :param name:
        :param class_names:
        :return:
        """
        return save_adjacency_matrix_for_gephi(matrix, name, root_dir=self.directory,
                                               notebook_mode=False, class_names=class_names)

    def save_setting(self, local_variables, excluded=None, append_string=''):
        """
        Object-oriented version of `save_setting`

        :param local_variables:
        :param excluded:
        :param append_string:
        :return:
        """
        excluded = as_list(excluded or [])
        excluded.append(self)  # no reason to save itself...
        return save_setting(local_variables, root_dir=self.directory, excluded=excluded,
                            default_overwrite=self.default_overwrite, collect_data=self.collect_data,
                            notebook_mode=False, do_print=self.do_print, append_string=append_string)

    def load_obj(self, name):
        """
         Object-oriented version of `load_obj`

        :param name: name of the file (.pkgz extension automatically added)
        :return: unpacked object
        """
        return load_obj(name, root_dir=self.directory, notebook_mode=False)

    def save_model(self, model, step=None, session=None):
        save_model(model, step=step, session=session, root_dir=self.directory, notebook_mode=False)

    def restore_model(self, model, step=None, session=None):
        restore_model(model, step=step, session=session, root_dir=self.directory, notebook_mode=False)

    def save_tf(self, var_list=None, name='save_tf', step=None, session=None, **saver_kwargs):
        tf.train.Saver(var_list=var_list, **saver_kwargs).save(
            session or tf.get_default_session(), join_paths(
                join_paths(self.directory, FOLDER_NAMINGS['MODELS_DIR']), name),
            global_step=step
        )

    def restore_tf(self, var_list=None, name='save_tf', session=None, **saver_kwargs):
        tf.train.Saver(var_list=var_list, **saver_kwargs).restore(
            session or tf.get_default_session(), join_paths(
                join_paths(self.directory, FOLDER_NAMINGS['MODELS_DIR']), name)
        )

    def open(self, filename, mode):
        return open(join_paths(self.directory, filename), mode)

    @contextmanager
    def redirect_stdout(self, filename=None, mode='a'):
        if filename is None: filename = 'stdout.txt'
        with self.open(filename, mode) as f, redirect_stdout(f):
            yield f


# noinspection PyPep8Naming
def Loader(folder_name):
    """
    utility method for creating a Saver with loading intentions,
    does not create timer nor append time to name. just give the folder name
    for the saver

    :param folder_name: (string or list of strings)
                        either absolute or relative, in which case root_directory will be used
    :return: a `Saver` object
    """
    return Saver(folder_name, append_date_to_name=False, timer=False,
                 collect_data=False, ask_for_description=False)
