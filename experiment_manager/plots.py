from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn
import sys

from experiment_manager.utils import merge_dicts

from experiment_manager.savers.save_and_load import Saver
from IPython.display import clear_output as c_out

seaborn.set_style('whitegrid')


# noinspection PyBroadException
def autoplot(saver_or_history, saver=None, append_string='', clear_output=True):
    if clear_output:
        try: c_out()
        except: pass
    if isinstance(saver_or_history, (list, tuple)):
        return [autoplot(soh, saver, append_string, clear_output=False) for soh in saver_or_history]
    if isinstance(saver_or_history, Saver):
        saver = saver_or_history
        try:
            history = saver.load_obj('all%s' % append_string)
        except FileNotFoundError:
            history = saver.pack_save_dictionaries(erase_others=False, save_packed=False,
                                                   append_string=append_string)
        if history is None:
            return 'nothing yet...'
    else: history = saver_or_history

    # noinspection PyBroadException
    def _simple_plot(_title, _label, _v):
        try:
            if isinstance(v, list):
                plt.title(_title.capitalize())
                plt.plot(_v, label=_label.capitalize())
        except:
            # TODO print the cause of exception. or at least the name
            print('Could not plot %s' % _title, file=sys.stderr)

    nest = defaultdict(lambda: {})
    for k, v in history.items():
        k_split = k.split('::')
        k_1 = k_split[1] if len(k_split) > 1 else ''
        nest[k_split[0]] = merge_dicts(nest[k_split[0]], {k_1: v})

    for k, _dict_k in nest.items():
        if k != 'SKIP' and k != 'HIDE':
            plt.figure(figsize=(8, 6))
            for kk, v in _dict_k.items():
                _simple_plot(k, kk, v)
            if all([kk for kk in _dict_k.keys()]): plt.legend(loc=0)
            if saver and saver.collect_data:
                saver.save_fig(k + '_' + append_string)
                saver.save_fig(k + '_' + append_string, extension='png')
            plt.show()
    print('='*50)
    return 'done...'
