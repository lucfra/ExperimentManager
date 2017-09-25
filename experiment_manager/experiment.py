class SLExperiment:

    def __init__(self, datasets):
        import tensorflow as tf
        self.datasets = datasets
        self.x = tf.placeholder(tf.float32, name='x', shape=self._compute_input_shape())
        self.y = tf.placeholder(tf.float32, name='y', shape=self._compute_output_shape())
        self.model = None
        self.errors = {}
        self.scores = {}
        self.optimizers = {}

    def build_model(self, *model_args, **model_kwargs):
        pass

    def build_errors(self, *error_args, **error_kwargs):
        pass

    def build_optimizer(self, *opt_args, **opt_kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def initialize(self, ss):
        [opt.initialize() if hasattr(opt, 'initialize') else ss.run(opt.initializer)
         for opt in self.optimizers.values()]

    def std_records(self):
        import experiment_manager.savers.records as rec
        return [
            rec.tensors(*self.errors.values(), *self.scores.values(), fd='train'),
            rec.tensors(*self.errors.values(), *self.scores.values(), fd='valid'),
            rec.tensors(*self.errors.values(), *self.scores.values(), fd='test'),
            rec.hyperparameters(),
            rec.hypergradients(),
        ]

    def mah(self, c):
        pass

    def _compute_input_shape(self):
        try:
            sh = self.datasets.train.dim_data
            return (None, sh) if isinstance(sh, int) else (None,) + sh
        except:
            print('Could not determine input dimension')
            return None

    def _compute_output_shape(self):
        try:
            sh = self.datasets.train.dim_target
            return (None, sh) if isinstance(sh, int) else (None,) + sh
        except:
            print('Could not determine output dimension')
            return None
