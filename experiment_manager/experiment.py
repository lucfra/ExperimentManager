"""
Simple container for useful quantities for a supervised learning experiment, where data is managed
with feed dictionary
"""
import tensorflow as tf


class SLExperiment:

    def __init__(self, datasets, dtype=tf.float32):
        self.datasets = datasets
        self.x = tf.placeholder(dtype, name='x', shape=self._compute_input_shape())
        self.y = tf.placeholder(dtype, name='y', shape=self._compute_output_shape())
        self.dtype = dtype
        self.model = None
        self.errors = {}
        self.scores = {}
        self.optimizers = {}

    # noinspection PyBroadException
    def _compute_input_shape(self):
        try:
            sh = self.datasets.train.dim_data
            return (None, sh) if isinstance(sh, int) else (None,) + sh
        except:
            print('Could not determine input dimension')
            return None

    # noinspection PyBroadException
    def _compute_output_shape(self):
        try:
            sh = self.datasets.train.dim_target
            return (None, sh) if isinstance(sh, int) else (None,) + sh
        except:
            print('Could not determine output dimension')
            return None
