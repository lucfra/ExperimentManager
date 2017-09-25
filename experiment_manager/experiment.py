class SLExperiment:

    def __init__(self, datasets):
        import tensorflow as tf
        self.datasets = datasets
        self.x, self.y = tf.placeholder(tf.float32, name='x'), tf.placeholder(tf.float32, name='y')
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
