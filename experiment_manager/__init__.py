from experiment_manager.datasets.structures import *
from experiment_manager.utils import *
import experiment_manager.datasets.load as load
from experiment_manager.savers.save_and_load import *
import experiment_manager.savers.records as rec
from experiment_manager.datasets.utils import redivide_data
from experiment_manager.search_strategies import *
from experiment_manager.experiment import *

# noinspection PyBroadException
try:
    from experiment_manager import plots

    import seaborn as sbn

    sbn.set(font_scale=1.3, style='whitegrid')
    from matplotlib import rc

    # rc('text', usetex=True)  problems on servers....
    # rc('font', family='serif')
except:
    plots = None
    sbn = None
    print('plots module loading failure...')

from tensorflow.python.platform import flags as fg


mod_help = lambda hs, dv: '(default: {}) '.format(dv) + hs


# noinspection PyProtectedMember
def _define_helper(flag_name, default_value, docstring, flagtype):
    """Registers 'flag_name' with 'default_value' and 'docstring'."""
    fg._global_parser.add_argument('--' + flag_name,
                                   default=default_value,
                                   help=mod_help(docstring, default_value),
                                   type=flagtype)


# noinspection PyProtectedMember
def DEFINE_boolean(flag_name, default_value, docstring):
    """Defines a flag of type 'boolean'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a boolean.
    docstring: A helpful message explaining the use of the flag.
  """

    # Register a custom function for 'bool' so --flag=True works.
    def str2bool(v):
        return v.lower() in ('true', 't', '1')

    fg._global_parser.add_argument('--' + flag_name,
                                   nargs='?',
                                   const=True,
                                   help=mod_help(docstring, default_value),
                                   default=default_value,
                                   type=str2bool)


fg._define_helper = _define_helper
fg.DEFINE_boolean = DEFINE_boolean
fg.DEFINE_bool = DEFINE_boolean
