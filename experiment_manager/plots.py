import numpy as np
import matplotlib.pyplot as plt
import seaborn

from experiment_manager.savers.save_and_load import Saver

seaborn.set_style('whitegrid')


def autoplot(saver_or_history, saver=None, append_string=''):

    if isinstance(saver_or_history, Saver):
        saver = saver_or_history
        try:
            history = saver.load_obj('all%s' % append_string)
        except FileNotFoundError:
            history = saver.pack_save_dictionaries(erase_others=False, save_packed=False,
                                                   append_string=append_string)
    else: history = saver_or_history

    def _simple_plot(_k, _v):
        split_k = _k.split('::')
        _title = None
        if isinstance(v, list) and isinstance(v[0], (float, int, np.number)):
            _title = split_k[0].capitalize()
            plt.title(_title)
            plt.plot(_v, legend=split_k[1].capitalize() if len(split_k) > 1 else '')
        return _title

    remain_to_process = list(history.keys())
    for k, v in history:
        remain_to_process.remove(k)
        if k in remain_to_process:  # this could be done with a generator.....
                title = _simple_plot(k, v)
                for kk in remain_to_process:
                    if kk.startswith(k.split('::')[0]):
                        remain_to_process.remove(kk)
                        title = _simple_plot(kk, history[kk])
                if saver: saver.save_fig(title)
                plt.show()
