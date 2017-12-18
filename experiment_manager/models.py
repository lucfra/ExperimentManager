import tensorflow as tf
import sys
import tensorflow.contrib.layers as tcl


from experiment_manager import filter_vars, as_tuple_or_list


class Network(object):
    """
    Base object for models
    """

    def __init__(self, _input, name=None, deterministic_initialization=False, reuse=False):
        """
        Creates an object that represent a network. Important attributes of a Network object are

        `var_list`: list of tf.Variables that constitute the parameters of the model

        `inp`: list, first element is `_input` and last should be output of the model. Other entries can be
        hidden layers activations.

        :param _input: tf.Tensor, input of this model.
        """
        super(Network, self).__init__()

        if not name:
            try:
                name = tf.get_variable_scope().name
            except IndexError:
                print('Warning: no name and no variable scope given', sys.stderr)
        self.name = name
        self.reuse = reuse

        self.deterministic_initialization = deterministic_initialization
        self._var_list_initial_values = []
        self._var_init_placeholder = None
        self._assign_int = []
        self._var_initializer_op = None

        self.layers = [_input]
        # self.s = None
        self._tf_saver = None

        with self._variable_scope(reuse):
            self._build()

    def _variable_scope(self, reuse):
        """
        May override default variable scope context, but pay attention since it looks like
        initializer and default initializer form contrib.layers do not work well together (I think this is because
        functions in contrib.layers usually specify an initializer....

        :param reuse:
        :return:
        """
        return tf.variable_scope(self.name, reuse=reuse)

    def __getitem__(self, item):
        """
        Get's the `activation`

        :param item:
        :return:
        """
        return self.layers[item]

    def __add__(self, other):
        if self.name not in tf.get_variable_scope().name:
            print('Warning: adding layers outside model variable scope', file=sys.stderr)
        self.layers.append(other)
        return self

    @property
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)

    @property
    def Ws(self):
        return self.filter_vars('weights')

    @property
    def bs(self):
        return self.filter_vars('biases')

    @property
    def out(self):
        return self[-1]

    def for_input(self, new_input):
        raise NotImplemented()

    def filter_vars(self, var_name):
        return filter_vars(var_name, self.name)

    def initialize(self, session=None):
        """
        Initialize the model. If `deterministic_initialization` is set to true,
        saves the initial weight in numpy which will be used for subsequent initialization.
        This is because random seed management in tensorflow is rather obscure... and I could not
        find a way to set the same seed across different initialization without exiting the session.

        :param session:
        :return:
        """
        # TODO placeholder are not useful at all here... just change initializer of tf.Variables !
        ss = session or tf.get_default_session()
        assert ss, 'No default session'
        if not self._var_initializer_op:
            self._var_initializer_op = tf.variables_initializer(self.var_list)
        ss.run(self._var_initializer_op)
        if self.deterministic_initialization:
            if self._var_init_placeholder is None:
                self._var_init_placeholder = tf.placeholder(tf.float32)
                self._assign_int = [v.assign(self._var_init_placeholder) for v in self.var_list]

            if not self._var_list_initial_values:
                self._var_list_initial_values = ss.run(self.var_list)

            else:
                [ss.run(v_op, feed_dict={self._var_init_placeholder: val})
                 for v_op, val in zip(self._assign_int, self._var_list_initial_values)]

    def _variables_to_save(self):
        return self.var_list

    def _build(self):
        raise NotImplemented()

    @property
    def tf_saver(self):
        if not self._tf_saver:
            self._tf_saver = tf.train.Saver(var_list=self._variables_to_save(), max_to_keep=1)
        return self._tf_saver

    def save(self, file_path, session=None, global_step=None):
        # TODO change this method! save the class (or at least the attributes you can save
        self.tf_saver.save(session or tf.get_default_session(), file_path, global_step=global_step)

    def restore(self, file_path, session=None, global_step=None):
        if global_step: file_path += '-' + str(global_step)
        self.tf_saver.restore(session or tf.get_default_session(), file_path)


class FeedForwardNet(Network):
    def __init__(self, _input, dims, name='FeedForwardNet', activation=tf.nn.relu,
                 var_collections=(tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES),
                 output_weight_initializer=tf.zeros_initializer,
                 deterministic_initialization=False, reuse=False):
        self.dims = as_tuple_or_list(dims)
        self.activation = activation
        self.var_collections = var_collections
        self.output_weight_initializer = output_weight_initializer
        super().__init__(_input, name, deterministic_initialization, reuse)

    def _build(self):
        for d in self.dims[:-1]:
            self + tcl.fully_connected(self.out, d, activation_fn=self.activation,
                                       variables_collections=self.var_collections, trainable=False)
        self._build_output_layer()

    def _build_output_layer(self):
        self + tcl.fully_connected(self.out, self.dims[-1], activation_fn=None,
                                   weights_initializer=self.output_weight_initializer,
                                   variables_collections=self.var_collections, trainable=False)

    def for_input(self, new_input):
        return FeedForwardNet(new_input, self.dims, self.name, self.activation,
                              self.var_collections, self.output_weight_initializer,
                              self.deterministic_initialization, reuse=True)


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=(10, 321))
    with tf.variable_scope('hji'):
        net = Network(x)
        net + tcl.fully_connected(net.out, 40) + tcl.fully_connected(net.out, 51) + tcl.fully_connected(
            net.out, 12
        ) + tcl.fully_connected(net.out, 1)

    print(net.layers)
    print(net.var_list)
