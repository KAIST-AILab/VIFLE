import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import sonnet as snt

_DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                         "b": tf.zeros_initializer()}
tfd = tfp.distributions
# tfd = tf.contrib.distributions


class ConditionalNormalDistribution(object):
    """A Normal distribution conditioned on Tensor inputs via a fc network."""

    def __init__(self, size, hidden_layer_sizes, sigma_min=1e-5, raw_sigma_bias=0.0,
                 hidden_activation_fn=tf.nn.tanh, initializers=None, fcnet=None,
                 name="conditional_normal_distribution"):
        """Creates a conditional Normal distribution.

        Args:
          size: The dimension of the random variable.
          hidden_layer_sizes: The sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs.
          sigma_min: The minimum standard deviation allowed, a scalar.
          raw_sigma_bias: A scalar that is added to the raw standard deviation
            output from the fully connected network. Set to 0.25 by default to
            prevent standard deviations close to 0.
          hidden_activation_fn: The activation function to use on the hidden layers
            of the fully connected network.
          initializers: The variable intitializers to use for the fully connected
            network. The network is implemented using snt.nets.MLP so it must
            be a dictionary mapping the keys 'w' and 'b' to the initializers for
            the weights and biases. Defaults to xavier for the weights and zeros
            for the biases when initializers is None.
          name: The name of this distribution, used for sonnet scoping.
        """
        self.sigma_min = sigma_min
        self.raw_sigma_bias = raw_sigma_bias
        self.name = name
        if initializers is None:
            initializers = _DEFAULT_INITIALIZERS
        if fcnet is None:
            self.fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [2*size],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + "_fcnet")
        else:
            self.fcnet = fcnet

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a normal distribution based on the inputs."""
        inputs = tf.concat(tensor_list, axis=1)
        outs = self.fcnet(inputs)
        mu, sigma = tf.split(outs, 2, axis=1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias),
                           self.sigma_min)
        return mu, sigma

    def stopped_call(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        mu, sigma = self.condition(args, **kwargs)
        return tf.distributions.Normal(loc=tf.stop_gradient(mu), scale=tf.stop_gradient(sigma))

    def __call__(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        mu, sigma = self.condition(args, **kwargs)
        return tf.distributions.Normal(loc=mu, scale=sigma)


class ConditionalBernoulliDistribution(object):
    """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

    def __init__(self, size, hidden_layer_sizes, bias_init=0.0,
                 hidden_activation_fn=tf.nn.tanh, initializers=None, fcnet=None,
                 name="conditional_bernoulli_distribution"):
        """Creates a conditional Bernoulli distribution.

        Args:
          size: The dimension of the random variable.
          hidden_layer_sizes: The sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs.
          hidden_activation_fn: The activation function to use on the hidden layers
            of the fully connected network.
          initializers: The variable intiializers to use for the fully connected
            network. The network is implemented using snt.nets.MLP so it must
            be a dictionary mapping the keys 'w' and 'b' to the initializers for
            the weights and biases. Defaults to xavier for the weights and zeros
            for the biases when initializers is None.
          bias_init: A scalar or vector Tensor that is added to the output of the
            fully-connected network that parameterizes the mean of this
            distribution.
          name: The name of this distribution, used for sonnet scoping.
        """
        self.bias_init = bias_init
        if initializers is None:
            initializers = _DEFAULT_INITIALIZERS
        if fcnet is None:
            self.fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [size],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + "_fcnet")
        else:
            self.fcnet = fcnet

    def condition(self, tensor_list):
        """Computes the p parameter of the Bernoulli distribution."""
        inputs = tf.concat(tensor_list, axis=1)
        return self.fcnet(inputs) + self.bias_init

    def __call__(self, *args):
        p = self.condition(args)
        return tf.distributions.Bernoulli(logits=p)


class ConditionalRelaxedBernoulliDistribution(object):
    """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

    def __init__(self, size, hidden_layer_sizes, temperature=0.1,
                 hidden_activation_fn=tf.nn.tanh, initializers=None, fcnet=None, bias_init=0.0,
                 name="conditional_relaxed_bernoulli_distribution"):
        """Creates a conditional Bernoulli distribution.

        Args:
          size: The dimension of the random variable.
          hidden_layer_sizes: The sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs.
          hidden_activation_fn: The activation function to use on the hidden layers
            of the fully connected network.
          initializers: The variable intiializers to use for the fully connected
            network. The network is implemented using snt.nets.MLP so it must
            be a dictionary mapping the keys 'w' and 'b' to the initializers for
            the weights and biases. Defaults to xavier for the weights and zeros
            for the biases when initializers is None.
          bias_init: A scalar or vector Tensor that is added to the output of the
            fully-connected network that parameterizes the mean of this
            distribution.
          name: The name of this distribution, used for sonnet scoping.
        """
        self.temperature = temperature
        self.bias_init = bias_init
        if initializers is None:
            initializers = _DEFAULT_INITIALIZERS
        if fcnet is None:
            self.fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [size],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + "_fcnet")
        else:
            self.fcnet = fcnet

    def condition(self, tensor_list):
        """Computes the p parameter of the Bernoulli distribution."""
        inputs = tf.concat(tensor_list, axis=1)
        return self.fcnet(inputs) + self.bias_init

    def unrelax(self, *args):
        p = self.condition(args)
        return tf.distributions.Bernoulli(logits=p)

    def __call__(self, *args):
        p = self.condition(args)
        return tfd.RelaxedBernoulli(temperature=self.temperature, logits=p)


class NormalApproximatePosterior(ConditionalNormalDistribution):
    """A Normally-distributed approx. posterior with res_q parameterization."""

    def condition(self, tensor_list, prior_mu):
        """Generates the mean and variance of the normal distribution.

        Args:
          tensor_list: The list of Tensors to condition on. Will be concatenated and
            fed through a fully connected network.
          prior_mu: The mean of the prior distribution associated with this
            approximate posterior. Will be added to the mean produced by
            this approximate posterior, in res_q fashion.
        Returns:
          mu: The mean of the approximate posterior.
          sigma: The standard deviation of the approximate posterior.
        """
        mu, sigma = super(NormalApproximatePosterior, self).condition(tensor_list)
        return mu + prior_mu, sigma


class ConditionalCategoricalDistribution(object):
    """A Categorical distribution conditioned on Tensor inputs via a fc net."""

    def __init__(self, size, interval, hidden_layer_sizes,
                 hidden_activation_fn=tf.nn.tanh, initializers=None, fcnet=None, bias_init=0.0,
                 name="conditional_categorical_distribution"):
        """Creates a conditional Categorical distribution.

        Args:
          size: The dimension of the random variable.
          hidden_layer_sizes: The sizes of the hidden layers of the fully connected
            network used to condition the distribution on the inputs.
          hidden_activation_fn: The activation function to use on the hidden layers
            of the fully connected network.
          initializers: The variable intiializers to use for the fully connected
            network. The network is implemented using snt.nets.MLP so it must
            be a dictionary mapping the keys 'w' and 'b' to the initializers for
            the weights and biases. Defaults to xavier for the weights and zeros
            for the biases when initializers is None.
          bias_init: A scalar or vector Tensor that is added to the output of the
            fully-connected network that parameterizes the mean of this
            distribution.
          name: The name of this distribution, used for sonnet scoping.
        """
        self.size = size
        self.interval = interval
        self.bias_init = bias_init
        if initializers is None:
            initializers = _DEFAULT_INITIALIZERS
        if fcnet is None:
            self.fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [size * interval],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + "_fcnet")
        else:
            self.fcnet = fcnet

    def condition(self, tensor_list):
        """Computes the p parameter of the Bernoulli distribution."""
        inputs = tf.concat(tensor_list, axis=1)
        raw_return = self.fcnet(inputs) + self.bias_init
        return tf.reshape(raw_return, [-1, self.size, self.interval])

    def __call__(self, *args):
        p = self.condition(args)
        return tf.distributions.Categorical(logits=p)


class MixedDistributions(object):
    def __init__(self, size1, size2, hidden_layer_sizes, temperature, sigma_min=0.0, raw_sigma_bias=0.25, bias_init=0.0,
                 hidden_activation_fn=tf.nn.tanh, initializers=None, fcnet=None,
                 name="flucell"):
        self.size1 = size1
        self.size2 = size2
        self.temperature = temperature
        self.sigma_min = sigma_min
        self.raw_sigma_bias = raw_sigma_bias
        self.bias_init = bias_init
        if fcnet is None:
            self.fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [2*size1 + size2],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + "_fcnet")
        else:
            self.fcnet = fcnet

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a normal distribution based on the inputs."""
        inputs = tf.concat(tensor_list, axis=1)
        outs = self.fcnet(inputs)
        mu, sigma, logits = tf.split(outs, [self.size1, self.size1, self.size2], axis=1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias),
                           self.sigma_min)
        logits = logits + self.bias_init
        return mu, sigma, logits

    def normal(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        mu, sigma, _ = self.condition(args, **kwargs)
        return tf.distributions.Normal(loc=mu, scale=sigma)

    def bern(self, *args, **kwargs):
        _, _, logits = self.condition(args, **kwargs)
        return tf.distributions.Bernoulli(logits=logits)

    def relaxed_bern(self, *args, **kwargs):
        _, _, logits = self.condition(args, **kwargs)
        return tfd.RelaxedBernoulli(temperature=self.temperature, logits=logits)


class LinearGaussianDistributions(object):
    def __init__(self, lin_transform, var, shift=None):
        self.A = np.float32(lin_transform)
        self.var = np.float32(var)
        self.shift = shift

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a normal distribution based on the inputs."""
        inputs = tf.concat(tensor_list, axis=1)
        if self.shift is None:
            mu = inputs @ tf.transpose(self.A)
        else:
            mu = inputs @ tf.transpose(self.A) + self.shift
        return mu, self.var

    def __call__(self, *args, **kwargs):
        """Creates a normal distribution conditioned on the inputs."""
        mu, sigma = self.condition(args, **kwargs)
        return tfd.Normal(loc=mu, scale=sigma)


class SineGaussianDistribution(object):
    def __init__(self, scale, freq, var):
        self.scale = np.float32(scale)
        self.freq = np.float32(freq)
        self.var = np.float32(var)

    def condition(self, tensor_list, **unused_kwargs):
        inputs = tf.concat(tensor_list, axis=1)
        mu = tf.sin(inputs)+self.scale*tf.sin(self.freq*inputs)
        return mu, self.var

    def __call__(self, *args, **kwargs):
        mu, sigma = self.condition(args, **kwargs)
        return tfd.Normal(loc=mu, scale=sigma)


class NonlinearGaussianDistribution(object):
    def __init__(self, lin_transform, var):
        self.A = np.float32(lin_transform)
        self.var = np.float32(var)

    def condition(self, tensor_list, **unused_kwargs):
        inputs = tf.concat(tensor_list, axis=1)
        mu = inputs*inputs + inputs @ tf.transpose(self.A)
        return mu, self.var

    def __call__(self, *args, **kwargs):
        mu, sigma = self.condition(args, **kwargs)
        return tfd.Normal(loc=mu, scale=sigma)


class LinSineGaussianDistribution(object):
    def __init__(self, scale, freq, var, lin_transform):
        self.scale = np.float32(scale)
        self.freq = np.float32(freq)
        self.var = np.float32(var)
        self.lin_transform = np.float32(lin_transform)

    def condition(self, tensor_list, **unused_kwargs):
        inputs = tf.concat(tensor_list, axis=1)
        mu = self.scale*tf.sin(self.freq*inputs) + inputs @ tf.transpose(self.lin_transform)
        return mu, self.var

    def __call__(self, *args, **kwargs):
        mu, sigma = self.condition(args, **kwargs)
        return tfd.Normal(loc=mu, scale=sigma)


class SimpleLinearDistribution(object):
    def __init__(self, dim, var):
        self.a = tf.Variable(tf.zeros([1, dim]), dtype=tf.float32, name="coeff_z")
        self.b = tf.Variable(tf.zeros([1, dim]), dtype=tf.float32, name="coeff_x")
        self.dim = dim
        self.var = var

    def condition(self, tensor_list, **unused_kwargs):
        inputs = tf.concat(tensor_list, axis=1)
        z, x = tf.split(inputs, num_or_size_splits=[self.dim, self.dim], axis=1)
        mu = tf.multiply(z, self.a)+tf.multiply(x, self.b)
        return mu, self.var

    def __call__(self, *args, **kwargs):
        mu, sigma = self.condition(args, **kwargs)
        return tfd.Normal(loc=mu, scale=sigma)

    def get_params(self):
        results = tf.concat([self.a, self.b], axis=1)
        return results


class SwitchDistribution(object):
    def __init__(self, A, dim, sigma):
        self.matrices = A
        self.dim = dim
        self.sigma = sigma

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the p parameter of the Bernoulli distribution."""
        inputs = tf.concat(tensor_list, axis=1)
        mu = tf.eye(self.dim)
        z, x = tf.split(inputs, num_or_size_splits=[self.dim, self.dim], axis=1)

        return mu

    def __call__(self, *args, **kwargs):
        mu = self.condition(args, **kwargs)
        return tfd.Normal(mu, self.sigma)


class RelaxedBernoulliDistribution(object):
    """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

    def __init__(self, dim, temperature=0.1, move=0.1):
        self.dim = dim
        self.temperature = temperature
        self.move = move

    def condition(self, tensor_list):
        """Computes the p parameter of the Bernoulli distribution."""
        inputs = tf.concat(tensor_list, axis=1)
        prob_next = self.move * tf.ones(shape=tf.shape(inputs)) + (1-2*self.move) * inputs
        return prob_next

    def unrelax(self, *args):
        p = self.condition(args)
        return tf.distributions.Bernoulli(probs=p)

    def __call__(self, *args):
        p = self.condition(args)
        return tfd.RelaxedBernoulli(temperature=self.temperature, probs=p)


class SimpleOneHot(object):
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, *args, **kwargs):
        return tfd.OneHotCategorical(logits=args[0] @ self.trans)


class HMM_for_bernoulli(object):
    def __init__(self, params, data_len):
        dim, _, beta, A, scale, freq = params
        init_prob = np.ones([dim,2])*0.5
        trans_prob = np.ones([dim,2,2])
        for i in range(dim):
            trans_prob[i,:,:]=np.array([[0.9,0.1],[0.1,0.9]])
        initial_dist = tf.distributions.Categorical(probs = init_prob)
        trans_dist = tf.distributions.Categorical(probs = trans_prob)
        obs_dist = tf.distributions.Normal(loc=np.array([np.zeros(dim), np.matmul(A, np.ones(dim))+scale*np.sin(freq*np.ones(dim))]).T,scale=beta)
        self.HMM = tfp.distributions.HiddenMarkovModel(initial_dist, trans_dist, obs_dist, data_len)

    def log_prob(self, x):
        return self.HMM.log_prob(x)