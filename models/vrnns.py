from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
from utils import distributions

_DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                         "b": tf.zeros_initializer()}


class VRNNCell(snt.AbstractModule):
    def __init__(self,
                 rnn_cell,
                 data_feat_extractor,
                 latent_feat_extractor,
                 prior,
                 generative,
                 approx_posterior,
                 prop_update,
                 random_seed=None,
                 name="vrnn"):
        super(VRNNCell, self).__init__(name=name)
        self.rnn_cell = rnn_cell
        self.data_feat_extractor = data_feat_extractor
        self.latent_feat_extractor = latent_feat_extractor
        self.prior = prior
        self.approx_posterior = approx_posterior
        self.generative = generative
        self.random_seed = random_seed
        self.encoded_z_size = latent_feat_extractor.output_size
        self.state_size = (self.rnn_cell.state_size, self.encoded_z_size)
        self.prop_update = prop_update

    def zero_state(self, batch_size, dtype):
        """The initial state of the VRNN.

        Contains the initial state of the RNN as well as a vector of zeros
        corresponding to z_0.
        Args:
          batch_size: The batch size.
          dtype: The data type of the VRNN.
        Returns:
          zero_state: The initial state of the VRNN.
        """
        return (self.rnn_cell.zero_state(batch_size, dtype),
                tf.zeros([batch_size, self.encoded_z_size], dtype=dtype))

    def _build(self, observations, state, mask):
        """Computes one timestep of the VRNN.

        Args:
          observations: The observations at the current timestep, a tuple
            containing the model inputs and targets as Tensors of shape
            [batch_size, data_size].
          state: The current state of the VRNN
          mask: Tensor of shape [batch_size], 1.0 if the current timestep is active
            active, 0.0 if it is not active.

        Returns:
          log_q_z: The logprob of the latent state according to the approximate
            posterior.
          log_p_z: The logprob of the latent state according to the prior.
          log_p_x_given_z: The conditional log-likelihood, i.e. logprob of the
            observation according to the generative distribution.
          kl: The analytic kl divergence from q(z) to p(z).
          state: The new state of the VRNN.
        """
        inputs, targets = observations
        rnn_state, prev_latent_encoded = state

        # Encode the data.
        inputs_encoded = self.data_feat_extractor(inputs)
        targets_encoded = self.data_feat_extractor(targets)

        # Run the RNN cell.
        rnn_inputs = tf.concat([inputs_encoded, prev_latent_encoded], axis=1)
        with tf.variable_scope("theta"):
            rnn_out, new_rnn_state = self.rnn_cell(rnn_inputs, rnn_state)

        # Create the prior and approximate posterior distributions.
        ob = tf.concat([rnn_out, targets_encoded], axis=1)
        latent_dist_prior = self.prior(rnn_out)
        latent_dist_q = self.approx_posterior(ob)

        # Sample the new latent state z and encode it.
        latent_state = tf.to_float(latent_dist_q.sample(seed=self.random_seed))
        latent_encoded = self.latent_feat_extractor(latent_state)

        # Calculate probabilities of the latent state according to the prior p
        # and approximate posterior q.

        log_q_z = tf.reduce_sum(latent_dist_q.log_prob(latent_state), axis=-1)
        log_p_z = tf.reduce_sum(latent_dist_prior.log_prob(latent_state), axis=-1)
        analytic_kl = tf.reduce_sum(
            tf.contrib.distributions.kl_divergence(
                latent_dist_q, latent_dist_prior),
            axis=-1)

        # Create the generative dist. and calculate the logprob of the targets.
        generative_dist = self.generative(latent_encoded, rnn_out)
        log_p_x_given_z = tf.reduce_sum(generative_dist.log_prob(targets), axis=-1)

        rew_vpred = self.prop_update.fef_psi.fef(ob)

        return (log_q_z, log_p_z, log_p_x_given_z, analytic_kl,
                (new_rnn_state, latent_encoded),
                (ob, latent_state, rew_vpred))


def create_vrnn(
        name,
        params,
        dataset_type,
        latent_type,
        prop_update,
        data_size,
        latent_size,
        generative_class,
        prior_hidden_sizes,
        gen_hidden_sizes,
        post_hidden_sizes,
        generative_bias_init,
        raw_sigma_bias,
        sigma_min,
        init_temp,
        rnn_hidden_size=None,
        fcnet_hidden_sizes=None,
        encoded_data_size=None,
        encoded_latent_size=None,
        initializers=None,
        random_seed=None):
    temperature = init_temp
    if rnn_hidden_size is None:
        rnn_hidden_size = latent_size
    if fcnet_hidden_sizes is None:
        fcnet_hidden_sizes = [latent_size]
    if encoded_data_size is None:
        encoded_data_size = latent_size
    if encoded_latent_size is None:
        encoded_latent_size = latent_size
    if initializers is None:
        initializers = _DEFAULT_INITIALIZERS
    with tf.variable_scope("phi"):
        if latent_type == "normal":
            approx_posterior = distributions.ConditionalNormalDistribution(
                size=latent_size,
                hidden_layer_sizes=post_hidden_sizes,
                sigma_min=sigma_min,
                raw_sigma_bias=raw_sigma_bias,
                initializers=initializers,
                fcnet=prop_update.prop_phi.prop,
                name="approximate_posterior")
        elif latent_type == "bernoulli":
            approx_posterior = distributions.ConditionalRelaxedBernoulliDistribution(
                size=latent_size,
                hidden_layer_sizes=fcnet_hidden_sizes,
                temperature=temperature,
                initializers=initializers,
                fcnet=prop_update.prop_phi.prop,
                name="approximate_posterior")
        else:
            raise ValueError("Undefined latent type %s" % latent_type)
    if dataset_type == 'pianoroll':
        cell = create_music_cell(name,
                                 latent_type,
                                 prop_update,
                                 approx_posterior,
                                 data_size,
                                 latent_size,
                                 generative_class,
                                 prior_hidden_sizes,
                                 gen_hidden_sizes,
                                 generative_bias_init,
                                 raw_sigma_bias,
                                 sigma_min,
                                 init_temp,
                                 rnn_hidden_size,
                                 fcnet_hidden_sizes,
                                 encoded_data_size,
                                 encoded_latent_size,
                                 initializers,
                                 random_seed)
    elif dataset_type == 'synthetic':
        cell = create_synth_cell(name,
                                 params,
                                 latent_type,
                                 prop_update,
                                 approx_posterior,
                                 data_size,
                                 latent_size,
                                 generative_class,
                                 prior_hidden_sizes,
                                 gen_hidden_sizes,
                                 generative_bias_init,
                                 raw_sigma_bias,
                                 sigma_min,
                                 temperature,
                                 fcnet_hidden_sizes,
                                 initializers,
                                 random_seed)
    else:
        raise ValueError("Undefined dataset_type %s" % dataset_type)
    return cell


def create_music_cell(
        name,
        latent_type,
        prop_update,
        approx_posterior,
        data_size,
        latent_size,
        generative_class,
        prior_hidden_sizes,
        gen_hidden_sizes,
        generative_bias_init,
        raw_sigma_bias,
        sigma_min,
        init_temp,
        rnn_hidden_size,
        fcnet_hidden_sizes,
        encoded_data_size,
        encoded_latent_size,
        initializers,
        random_seed):
    with tf.variable_scope("%s/theta" % name):
        temperature = init_temp
        data_feat_extractor = snt.nets.MLP(
            output_sizes=fcnet_hidden_sizes + [encoded_data_size],
            initializers=initializers,
            name="data_feat_extractor")
        latent_feat_extractor = snt.nets.MLP(
            output_sizes=fcnet_hidden_sizes + [encoded_latent_size],
            initializers=initializers,
            name="latent_feat_extractor")
        if latent_type == "normal":
            prior = distributions.ConditionalNormalDistribution(
                size=latent_size,
                hidden_layer_sizes=prior_hidden_sizes,
                sigma_min=sigma_min,
                raw_sigma_bias=raw_sigma_bias,
                initializers=initializers,
                name="prior")
        elif latent_type == "bernoulli":
            prior = distributions.ConditionalRelaxedBernoulliDistribution(
                size=latent_size,
                hidden_layer_sizes=fcnet_hidden_sizes,
                temperature=temperature,
                initializers=initializers,
                name="prior")
        else:
            raise ValueError("Undefined latent type %s" % latent_type)
        if generative_class == distributions.ConditionalBernoulliDistribution:
            generative = distributions.ConditionalBernoulliDistribution(
                size=data_size,
                hidden_layer_sizes=gen_hidden_sizes,
                initializers=initializers,
                bias_init=generative_bias_init,
                name="generative")
        elif generative_class == distributions.ConditionalNormalDistribution:
            generative = distributions.ConditionalNormalDistribution(
                size=data_size,
                hidden_layer_sizes=gen_hidden_sizes,
                initializers=initializers,
                name="generative")
        else:
            raise ValueError("Undefined generative class %s" % generative_class)
        rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size,
                                           initializer=initializers["w"])
    return VRNNCell(rnn_cell, data_feat_extractor, latent_feat_extractor,
                    prior, generative, approx_posterior, prop_update, random_seed=random_seed)


def create_synth_cell(
        name,
        params,
        latent_type,
        prop_update,
        approx_posterior,
        data_size,
        latent_size,
        generative_class,
        prior_hidden_sizes,
        gen_hidden_sizes,
        generative_bias_init,
        raw_sigma_bias,
        sigma_min,
        temperature,
        fcnet_hidden_sizes,
        initializers,
        random_seed):
    alpha, beta, A, B, scale, freq = params
    if latent_type == "normal":
        prior = distributions.LinSineGaussianDistribution(
            scale=1,
            freq=0,
            var=alpha,
            lin_transform=A
        )
    elif latent_type == "bernoulli":
        prior = distributions.ConditionalRelaxedBernoulliDistribution(
            size=latent_size,
            hidden_layer_sizes=fcnet_hidden_sizes,
            temperature=temperature,
            initializers=initializers,
            name="prior")
    else:
        raise ValueError("Undefined latent type %s" % latent_type)

    generative = distributions.LinSineGaussianDistribution(
        scale=scale,
        freq=freq,
        var=beta,
        lin_transform=B
    )
    return LinGaussianCell(prior, generative, approx_posterior, prop_update, random_seed=random_seed)
