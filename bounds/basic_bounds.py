import tensorflow as tf
import utils.nested_utils as nested


def iwae(cell,
         inputs,
         seq_lengths,
         float_type=tf.float32,
         num_samples=1,  # if num_samples=1, then iwae==elbo
         parallel_iterations=30,
         swap_memory=True):
    """Computes the IWAE lower bound on the log marginal probability.

    This method accepts a stochastic latent variable model and some observations
    and computes a stochastic lower bound on the log marginal probability of the
    observations. The IWAE estimator is defined by averaging multiple importance
    weights. For more details see "Importance Weighted Autoencoders" by Burda
    et al. https://arxiv.org/abs/1509.00519.

    When num_samples = 1, this bound becomes the evidence lower bound (ELBO).

    Args:
        config: FLAGS
        cell: A callable that implements one timestep of the model. See
            models/vrnn.py for an example.
        inputs: The inputs to the model. A potentially nested list or tuple of
            Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
            have a rank at least two and have matching shapes in the first two
            dimensions, which represent time and the batch respectively. At each
            timestep 'cell' will be called with a slice of the Tensors in inputs.
        seq_lengths: A [batch_size] Tensor of ints encoding the length of each
            sequence in the batch (sequences can be padded to a common length).
        num_samples: The number of samples to use.
        parallel_iterations: The number of parallel iterations to use for the
            internal while loop.
        swap_memory: Whether GPU-CPU memory swapping should be enabled for the
            internal while loop.

    Returns:
        log_p_hat: A Tensor of shape [batch_size] containing IWAE's estimate of the
            log marginal probability of the observations.
        kl: A Tensor of shape [batch_size] containing the kl divergence
            from q(z|x) to p(z), averaged over samples.
        log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
            containing the log weights at each timestep. Will not be valid for
            timesteps past the end of a sequence.
        log_ess: A Tensor of shape [max_seq_len, batch_size] containing the log
            effective sample size at each timestep. Will not be valid for timesteps
            past the end of a sequence.
    """
    batch_size = tf.shape(seq_lengths)[0]
    max_seq_len = tf.reduce_max(seq_lengths)
    seq_mask = tf.transpose(
        tf.sequence_mask(seq_lengths, maxlen=max_seq_len, dtype=float_type),
        perm=[1, 0])        # compute mask for inputs. [max_seq_len, batch_size]
    if num_samples > 1:
        inputs, seq_mask = nested.tile_tensors([inputs, seq_mask], [1, num_samples])
    inputs_ta, mask_ta = nested.tas_for_tensors([inputs, seq_mask], max_seq_len)

    t0 = tf.constant(0, tf.int32)
    init_states = cell.zero_state(batch_size * num_samples, float_type)
    ta_names = ['log_weights', 'log_ess']
    tas = [tf.TensorArray(float_type, max_seq_len, name='%s_ta' % n)
           for n in ta_names]   # define tas for while
    log_weights_acc = tf.zeros([num_samples, batch_size], dtype=float_type)
    kl_acc = tf.zeros([num_samples * batch_size], dtype=float_type)
    accs = (log_weights_acc, kl_acc)

    trajectories = {
        "mask": tf.TensorArray(tf.float32, max_seq_len, name="mask"),
        "ob": tf.TensorArray(tf.float32, max_seq_len, name="ob"),
        "ac": tf.TensorArray(tf.float32, max_seq_len, name="ac"),
        "fef_vpred": tf.TensorArray(tf.float32, max_seq_len, name="fef_vpred"),
        "log_p_z": tf.TensorArray(tf.float32, max_seq_len, name="log_p_z"),
        "log_p_x_given_z": tf.TensorArray(tf.float32, max_seq_len, name="log_p_x_given_z"),
        "log_q_z": tf.TensorArray(tf.float32, max_seq_len, name="log_q_z")
    }

    def while_predicate(t, *unused_args):
        return t < max_seq_len

    def while_step(t, rnn_state, tas, accs, trajectories):
        """Implements one timestep of IWAE computation."""
        log_weights_acc, kl_acc = accs
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
        # Run the cell for one step.
        log_q_z, log_p_z, log_p_x_given_z, kl, new_state, rl_term = cell(
            cur_inputs,
            rnn_state,
            cur_mask,
        )
        # Compute the incremental weight and use it to update the current
        # accumulated weight.
        kl_acc += kl * cur_mask
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
        log_alpha = tf.reshape(log_alpha, [num_samples, batch_size])
        log_weights_acc += log_alpha
        # Calculate the effective sample size.
        ess_num = 2 * tf.reduce_logsumexp(log_weights_acc, axis=0)
        ess_denom = tf.reduce_logsumexp(2 * log_weights_acc, axis=0)
        log_ess = ess_num - ess_denom
        # Update the  Tensorarrays and accumulators.
        ta_updates = [log_weights_acc, log_ess]
        new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]
        new_accs = (log_weights_acc, kl_acc)

        new_trajectories = {
            "mask": trajectories["mask"].write(t, cur_mask),
            "ob": trajectories["ob"].write(t, rl_term[0]),
            "ac": trajectories["ac"].write(t, rl_term[1]),
            "fef_vpred": trajectories["fef_vpred"].write(t, rl_term[2]),
            "log_p_z": trajectories["log_p_z"].write(t, log_p_z),
            "log_p_x_given_z": trajectories["log_p_x_given_z"].write(t, log_p_x_given_z),
            "log_q_z": trajectories["log_q_z"].write(t, log_q_z)
        }

        return t + 1, new_state, new_tas, new_accs, new_trajectories

    _, _, tas, accs, trajectories = tf.while_loop(
        while_predicate,
        while_step,
        loop_vars=(t0, init_states, tas, accs, trajectories),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    log_weights, log_ess = [x.stack() for x in tas]
    final_log_weights, kl = accs
    log_p_hat = (tf.reduce_logsumexp(final_log_weights, axis=0) -
                 tf.log(tf.to_float(num_samples)))
    kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)
    log_weights = tf.transpose(log_weights, perm=[0, 2, 1])

    trajectories = {
        "mask": trajectories["mask"].stack(),
        "ob": trajectories["ob"].stack(),
        "ac": trajectories["ac"].stack(),
        "fef_vpred": trajectories["fef_vpred"].stack(),
        "log_p_z": trajectories["log_p_z"].stack(),
        "log_p_x_given_z": trajectories["log_p_x_given_z"].stack(),
        "log_q_z": trajectories["log_q_z"].stack(),
        "log_weights": log_weights,
    }

    return log_p_hat, kl, log_weights, log_ess, trajectories
