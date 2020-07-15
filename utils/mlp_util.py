import tensorflow as tf
import sonnet as snt

# _DEFAULT_INITIALIZER = {"w": tf.contrib.layers.xavier_initializer(), "b": tf.zeros_initializer()}
_DEFAULT_INITIALIZER = {"w": tf.contrib.layers.xavier_initializer(), "b": tf.random_normal_initializer()}


class MlpProp(object):
    """policy with multi-layer perceptron"""
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, *args, **kwargs):
        pass

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class NormalMlpProp(MlpProp):
    """normal policy"""
    def _init(self, ob, ac_shape, hid_size, num_hid_layers, sigma_min, raw_sigma_bias):
        """
        :param ob:
        :param ac_shape:
        :param hid_size:
        :param num_hidden_layers:
        :param sigma_min:
        :param raw_sigma_bias:
        :return:
        """
        with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
            self.prop = snt.nets.MLP(
                output_sizes=[hid_size] * num_hid_layers + [ac_shape * 2],
                activation=tf.nn.tanh,
                initializers=_DEFAULT_INITIALIZER,
                activate_final=False,
                use_bias=True,
                name="normal_pol"
            )

        pdparam = self.prop(ob)
        mu, raw_sigma = tf.split(pdparam, 2, axis=1)
        sigma = tf.maximum(tf.nn.softplus(raw_sigma + raw_sigma_bias), sigma_min)

        self.pd = tf.distributions.Normal(loc=mu, scale=sigma)


class BernoulliMlpProp(MlpProp):
    """bernoulli policy"""
    def _init(self, ob, ac_shape, hid_size, num_hid_layers, logits_bias):
        """
        :param ob:
        :param ac_shape:
        :param hid_size:
        :param num_hid_layers:
        :param logit_bias:
        :return:
        """
        self.prop = snt.nets.MLP(
            output_sizes=[hid_size] * num_hid_layers + [ac_shape],
            activation=tf.nn.tanh,
            initializers=_DEFAULT_INITIALIZER,
            activate_final=False,
            use_bias=True,
            name="bernoulli_pol"
        )

        pdparam = self.prop(ob) + logits_bias

        self.pd = tf.distributions.Bernoulli(logits=pdparam)


class MlpFEFunction(object):
    """value function with multi-layer perceptron"""
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, *args, **kwargs):
        pass

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class SimpleFEFunction(MlpFEFunction):
    """simplest mlp value function"""
    def _init(self, ob, hid_size, num_hid_layers):
        with tf.variable_scope("value_function", reuse=tf.AUTO_REUSE):
            self.fef = snt.nets.MLP(
                output_sizes=[hid_size] * num_hid_layers + [1],
                # activation=tf.nn.tanh,
                activation=tf.nn.relu,
                initializers=_DEFAULT_INITIALIZER,
                activate_final=False,
                use_bias=True,
                name="simple_vf"
            )

class GAEFEFunction(MlpFEFunction):
    """simplest mlp value function"""
    def _init(self, ob, hid_size, num_hid_layers):
        with tf.variable_scope("value_function", reuse=tf.AUTO_REUSE):
            self.fef = snt.nets.MLP(
                output_sizes=[hid_size] * num_hid_layers + [1],
                # activation=tf.nn.tanh,
                activation=tf.nn.relu,
                initializers=_DEFAULT_INITIALIZER,
                activate_final=True,
                use_bias=True,
                name="simple_gae_vf"
            )
        # self.vpred = self.vf(ob)

class LogGAEFEFunction(MlpFEFunction):
    """simplest mlp value function"""
    def _init(self, ob, hid_size, num_hid_layers):
        with tf.variable_scope("value_function", reuse=tf.AUTO_REUSE):
            self.fef = snt.nets.MLP(
                output_sizes=[hid_size] * num_hid_layers + [1],
                # activation=tf.nn.tanh,
                activation=tf.nn.leaky_relu,
                initializers=_DEFAULT_INITIALIZER,
                activate_final=True,
                use_bias=True,
                name="simple_gae_vf"
            )
        # self.vpred = self.vf(ob)
