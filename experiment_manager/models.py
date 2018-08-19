import tensorflow as tf
import sys
import tensorflow.contrib.layers as tcl

from experiment_manager import filter_vars, as_tuple_or_list, maybe_get

from tensorflow.python.client.session import register_session_run_conversion_functions


class ParametricFunction:
    def __init__(self, x, params, y, rule, **kwargs):
        self.x = x
        self.y = y
        self.params = params
        self.rule = rule
        self._kwargs = kwargs

    @property
    def var_list(self):
        return self.params

    @property
    def out(self):
        return self.y

    def with_params(self, new_params):
        print(self.rule)
        return self.rule(self.x, new_params, **self._kwargs)

    def for_input(self, new_x):
        print(self.rule)
        return self.rule(new_x, self.params, **self._kwargs)

    def __add__(self, other):
        """
        Note: assumes that the inputs are the same
        """
        assert isinstance(other, ParametricFunction)
        return ParametricFunction(self.x, [self.params, other.params], self.y + other.y,
                                  lambda _inp, _prm, **kwa: kwa['add_arg1'].rule(
                                      _inp, _prm[0], **kwa['add_arg1']._kwargs) + kwa['add_arg2'].rule(
                                      _inp, _prm[1], **kwa['add_arg2']._kwargs),
                                  add_arg1=self, add_arg2=other)

    def __matmul__(self, other):
        """
        Implements mathematical composition: self \circ other
        """
        assert isinstance(other, ParametricFunction)
        return ParametricFunction(other.x, [self.params, other.params], self.for_input(other.y),
                                  lambda _inp, _prm, **kwa: kwa['comp_arg1'].rule(
                                      _inp, _prm[0], **kwa['comp_arg1']._kwargs) @ kwa['comp_arg2'].rule(
                                      _inp, _prm[1], **kwa['comp_arg2']._kwargs),
                                  comp_arg1=self, comp_arg2=other)


tf.register_tensor_conversion_function(ParametricFunction,
                                       lambda value, dtype=None, name=None, as_ref=False:
                                       tf.convert_to_tensor(value.y, dtype, name))

register_session_run_conversion_functions(ParametricFunction,
                                          lambda pf: ([pf.y], lambda val: val[0]))


def _process_initializer(initiazlizers, j, default):
    if callable(initiazlizers):
        return initiazlizers
    elif initiazlizers is not None:
        return maybe_get(initiazlizers, j)
    else:
        return default


def _pass_shape(shape, initializer, j):
    init = maybe_get(initializer, j)
    return None if (hasattr(init, 'shape') or isinstance(init, list)) else shape


def id_pf(x, weights=None):
    """
    Identity as ParametricFunction
    """
    return ParametricFunction(x, [], x, id_pf)


def lin_func(x, weights=None, dim_out=None, activation=None, initializers=None,
             name='lin_model', variable_getter=tf.get_variable):
    assert dim_out or weights
    with tf.variable_scope(name):
        if weights is None:
            weights = [variable_getter('w', initializer=_process_initializer(initializers, 0, tf.zeros_initializer),
                                       shape=_pass_shape((x.shape[1], dim_out), initializers, 0)),
                       variable_getter('b', initializer=_process_initializer(initializers, 1, tf.zeros_initializer),
                                       shape=_pass_shape((dim_out,), initializers, 1))]
        out = x @ weights[0] + weights[1]
        if activation:
            out = activation(out)
        return ParametricFunction(x, weights, out, lin_func, activation=activation)


def ffnn(x, weights=None, dims=None, activation=tf.nn.relu, name='ffnn', initiazlizers=None,
         variable_getter=tf.get_variable):
    """
    Constructor for a feed-forward neural net as Parametric function
    """
    assert dims or weights or isinstance(initiazlizers, list)

    with tf.variable_scope(name):
        params = weights if weights is not None else []
        out = x
        n_layers = len(dims) if dims else len(weights)//2 + 1

        print('begin of ', name, '-'*5)
        for i in range(n_layers-1):
            if weights is None:
                with tf.variable_scope('layer_{}'.format(i + 1)):
                    params += [variable_getter('w',
                                               shape=_pass_shape((dims[i], dims[i+1]), initiazlizers, 2*i),
                                               dtype=tf.float32,
                                               initializer=_process_initializer(initiazlizers, 2*i, None)),
                               variable_getter('b', shape=_pass_shape((dims[i+1],), initiazlizers, 2*i),
                                               initializer=_process_initializer(initiazlizers, 2*i + 1, tf.zeros_initializer))]
            out = out @ params[2*i] + params[2*i + 1]
            if i < n_layers - 2: out = activation(out)
            print(out)
        print('end of ', name, '-'*5)
        return ParametricFunction(x, params, out, ffnn, activation=activation)


def fixed_init_ffnn(x, weights=None, dims=None, activation=tf.nn.relu, name='ffnn', initializers=None,
                    variable_getter=tf.get_variable):
    from time import time
    import numpy as np
    _temp = ffnn(x, weights, dims, activation, 'tempRandomNet' + str(int(time())) + str(np.random.randint(0, 10000)),
                 initializers)
    new_session = False
    session = tf.get_default_session()
    if session is None:
        session = tf.InteractiveSession()
        new_session = True
    tf.variables_initializer(_temp.var_list).run()
    net = ffnn(x, weights, dims, activation, name, session.run(_temp.var_list), variable_getter=variable_getter)
    if new_session: session.close()
    return net


class Network(object):
    # definitely deprecated!
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
    import numpy as np
    x = tf.constant(np.random.rand(2, 3), tf.float32)
    x2 = tf.constant(np.random.rand(2, 3), tf.float32)
    lf = lin_func(x, dim_out=3)


    idx = id_pf(x)

    res_mod = lf + idx

    ss = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    print(res_mod)
    print(ss.run(lf))
    print(ss.run(res_mod))

    res_mod2 = res_mod.for_input(x2)

    print(ss.run(res_mod2))

    res_mod3 = res_mod2 + lin_func(x, dim_out=3, name='lin2') + idx

    tf.global_variables_initializer().run()

    print(ss.run(res_mod3))

    bla = res_mod3.for_input(x)

    print(ss.run(bla))

    print(bla.params)

