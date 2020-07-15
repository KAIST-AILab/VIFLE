import os
import numpy as np
import tensorflow as tf

from proposals import vifle_proposals
from utils import mlp_util
from data import create_datasets
from bounds import basic_bounds
from models import vrnns
from utils import distributions
import utils.tf_util as U


def create_dataset(config, split, shuffle, repeat):
    if config.dataset_type == "pianoroll":
        if split == 'train':
            inputs, targets, lengths, params = create_datasets.create_pianoroll_dataset(
                path=config.dataset_path,
                split=split,
                batch_size=config.batch_size,
                shuffle=shuffle,
                repeat=repeat
            )
        else:
            inputs, targets, lengths, params = create_datasets.create_pianoroll_dataset(
                path=config.dataset_path,
                split=split,
                batch_size=-1,
                shuffle=shuffle,
                repeat=repeat
            )
        # params = mean
        generative_bias_init = -tf.log(1. / tf.clip_by_value(params, 0.0001, 0.9999) - 1)
        generative_distribution_class = distributions.ConditionalBernoulliDistribution
    elif config.dataset_type == "synthetic":
        if split == 'train':
            inputs, targets, lengths, params = create_datasets.create_synthetic_dataset(
                path=config.dataset_path,
                split=split,
                batch_size=config.batch_size,
                data_dim=config.data_dimension,
                shuffle=shuffle,
                repeat=repeat
            )
        else:
            inputs, targets, lengths, params = create_datasets.create_synthetic_dataset(
                path=config.dataset_path,
                split=split,
                batch_size=-1,
                data_dim=config.data_dimension,
                shuffle=shuffle,
                repeat=repeat
            )
        # params =
        generative_bias_init = 0
        generative_distribution_class = distributions.LinSineGaussianDistribution
    else:
        raise ValueError("Undefined dataset type %s" % config.dataset_type)
    return inputs, targets, lengths, params, generative_distribution_class, generative_bias_init


def create_cell(config, dataset_args):
    # get dataset arguments
    # inputs: (sequence_length) x (batch_size) x (data_dimension)
    inputs, targets, lengths, params, generative_distribution_class, generative_bias_init = dataset_args
    prior_hidden_sizes = [config.latent_size] * config.num_hidden_prior
    gen_hidden_sizes = [config.latent_size] * config.num_hidden_gen
    post_hidden_sizes = [config.latent_size] * config.num_hidden_post
    # set parameters and mlp
    latent_size = rnn_hidden_size = encoded_data_size = config.latent_size
    ob_shape = rnn_hidden_size+encoded_data_size
    ac_shape = latent_size
    ob = tf.placeholder(dtype=tf.float32, shape=[None]+[ob_shape], name="ob")

    if config.latent_type == "normal":
        def prop_fn(ob, ac_shape, name='prop_fn'):
            return mlp_util.NormalMlpProp(name=name, ob=ob, ac_shape=ac_shape, hid_size=latent_size,
                                          num_hid_layers=config.num_hidden_post,
                                          sigma_min=config.sigma_min, raw_sigma_bias=config.raw_sigma_bias)
    elif config.latent_type == "bernoulli":
        logits_bias = np.zeros(ac_shape, dtype=float)
        def prop_fn(ob, ac_shape, name='bernoulli_prop'):
            return mlp_util.BernoulliMlpProp(name=name, ob=ob, ac_shape=ac_shape, hid_size=latent_size,
                                             num_hid_layers=config.num_hidden_post,
                                             logits_bias=logits_bias)
    else:
        raise ValueError('Undefined latent type %s' % config.latent_type)
    def fe_fn(ob, name='fe_fn'):
        return mlp_util.LogGAEFEFunction(name=name, ob=ob, hid_size=64, num_hid_layers=2)
    # construct cell
    if config.cell == "vrnn":
        if config.algorithm == 'reparam':
            prop_update = vifle_proposals.REPARAM(ob, ac_shape, prop_fn, fe_fn, config.batch_size, config.num_samples, config.learning_rate)
        elif config.algorithm == 'reinforce':
            prop_update = vifle_proposals.REINFORCE(ob, ac_shape, prop_fn, fe_fn, config.batch_size, config.num_samples, config.learning_rate)
        elif config.algorithm == 'vimco':
            prop_update = vifle_proposals.VIMCO(ob, ac_shape, prop_fn, fe_fn, config.batch_size, config.num_samples, config.learning_rate)
        elif config.algorithm == 'fr':
            prop_update = vifle_proposals.FR(ob, ac_shape, prop_fn, fe_fn, config.batch_size, config.num_samples, config.learning_rate)
        elif config.algorithm == 'vifle':
            prop_update = vifle_proposals.VIFLE(ob, ac_shape, prop_fn, fe_fn, config.batch_size, config.num_samples, config.learning_rate)
        else:
            raise ValueError('Undefined algorithm %s' % config.algorithm)
        cell = vrnns.create_vrnn(
            name=config.cell,
            params=params,
            dataset_type=config.dataset_type,
            latent_type=config.latent_type,
            prop_update=prop_update,
            data_size=inputs.get_shape().as_list()[2],
            latent_size=config.latent_size,
            prior_hidden_sizes=prior_hidden_sizes,
            gen_hidden_sizes=gen_hidden_sizes,
            post_hidden_sizes=post_hidden_sizes,
            generative_class=generative_distribution_class,
            generative_bias_init=generative_bias_init,
            raw_sigma_bias=config.raw_sigma_bias,
            sigma_min=config.sigma_min,
            init_temp=config.temperature
        )
    else:
        raise ValueError('Undefined cell %s' % config.cell)
    return cell


def restore_checkpoint_if_exists(saver, sess, logdir):
    """Looks for a checkpoint and restores the session from it if found.
    Args:
        saver: A tf.train.Saver for restoring the session.
        sess: A TensorFlow session.
        logdir: The directory to look for checkpoints in.
    Returns:
        True if a checkpoint was found and restored, False otherwise.
    """
    checkpoint = tf.train.get_checkpoint_state(logdir)
    if checkpoint:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        full_checkpoint_path = os.path.join(logdir, checkpoint_name)
        saver.restore(sess, full_checkpoint_path)
        return True
    return False


def run(config):
    def create_loss():
        train_dataset_args = create_dataset(config, split="train", shuffle=True, repeat=True)
        test_dataset_args = create_dataset(config, split="test", shuffle=True, repeat=True)
        valid_dataset_args = create_dataset(config, split="valid", shuffle=True, repeat=True)
        inputs, targets, lengths, params, _, _ = train_dataset_args
        test_inputs, test_targets, test_lengths, _, _, _ = test_dataset_args
        valid_inputs, valid_targets, valid_lengths, _, _, _ = valid_dataset_args

        cell = create_cell(config, train_dataset_args)

        if config.bound == "iwae":
            ll_per_seq, kl, log_weight, log_ess, trajectories = \
                basic_bounds.iwae(cell, (inputs, targets), lengths, num_samples=config.num_samples)
        else:
            raise ValueError("Undefined bound %s" % config.bound)

        if config.test_bound == "iwae":
            valid_ll_per_seq, _, _, _, _ = \
                basic_bounds.iwae(cell, (valid_inputs, valid_targets), valid_lengths, num_samples=config.test_num_samples)
        else:
            raise ValueError("Undefined bound %s" % config.test_bound)

        ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
        valid_ll_per_t = tf.reduce_mean(valid_ll_per_seq / tf.to_float(valid_lengths))

        return cell, ll_per_t, valid_ll_per_t, trajectories, lengths

    def create_graph():
        global_step = tf.train.get_or_create_global_step()
        cell, bound, valid_bound, trajectories, lengths = create_loss()
        loss = -bound
        opt = tf.train.AdamOptimizer(config.learning_rate)
        if config.model_train:
            grad_theta = opt.compute_gradients(loss, var_list=tf.trainable_variables("%s/theta" % config.cell))
            train_op_theta = opt.apply_gradients(grad_theta, global_step=global_step)
        else:
            train_op_theta = tf.constant(1)
        if config.algorithm == 'reparam':
            grad_phi = opt.compute_gradients(loss, var_list=tf.trainable_variables('prop_phi'))
            train_op_phi = opt.apply_gradients(grad_phi, global_step=global_step)
        else:
            train_op_phi = tf.constant(1)
        return cell, bound, valid_bound, trajectories, lengths, train_op_theta, train_op_phi, global_step

    valid_best = -1000000

    cell, bound, valid_bound, trajectories, lengths, train_op_theta, train_op_phi, global_step = create_graph()
    sess = U.get_session()
    U.initialize()
    cur_step = 0

    saver = tf.train.Saver(max_to_keep=1)
    valid_saver = tf.train.Saver(max_to_keep=1)
    model_savepath = config.logdir + '/model.ckpt'
    valid_best_model_savepath = config.logdir + '/valid_best/valid_best_model.ckpt'

    if not os.path.exists(config.logdir):
        os.makedirs(config.logdir)
        os.makedirs(config.logdir + '/valid_best')
    ckpt = tf.train.get_checkpoint_state(config.logdir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        valid_saver.restore(sess, ckpt.model_checkpoint_path)
        cur_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print('Model and log loaded! (checkpoint_path=%s, cur_step=%d)' % (ckpt.model_checkpoint_path, cur_step))

    while cur_step < config.max_iter+1:
        if config.algorithm == 'reparam':
            _, _, bound_value, valid_bound_value = sess.run([train_op_theta, train_op_phi, bound, valid_bound])
        elif "reinforce" in config.algorithm or "vimco" in config.algorithm or "vifle" in config.algorithm or "fr" in config.algorithm:
            _, bound_value, raw_seg, valid_bound_value, run_lengths = sess.run([train_op_theta, bound, trajectories, valid_bound, lengths])
            cell.prop_update.update(raw_seg, run_lengths)
        else:
            raise ValueError("Undefined algorithm %s" % config.algorithm)

        if valid_bound_value > valid_best and cur_step > config.init_steps:
            valid_best = valid_bound_value
            valid_best_model_saved_path = valid_saver.save(sess, valid_best_model_savepath, global_step=cur_step)
            print('Model saved: %s' % valid_best_model_saved_path)
        # for save - current work
        if cur_step % config.save_every == 0:
            model_saved_path = saver.save(sess, model_savepath, global_step=cur_step)
            print('Model saved: %s' % model_saved_path)
        cur_step += 1


def run_eval(config):
    def create_loss():
        test_dataset_args = create_dataset(config, split="test", shuffle=True, repeat=True)
        test_inputs, test_targets, test_lengths, _, _, _ = test_dataset_args

        cell = create_cell(config, test_dataset_args)

        test_ll_per_seq, kl, log_weight, log_ess, trajectories = \
            basic_bounds.iwae(cell, (test_inputs, test_targets), test_lengths, num_samples=config.num_samples)

        test_ll_per_t = tf.reduce_mean(test_ll_per_seq / tf.to_float(test_lengths))

        return test_ll_per_t, trajectories, cell, log_weight, test_lengths

    def create_graph():
        global_step = tf.train.get_or_create_global_step()
        test_bound, trajectories, cell, log_weight, test_lengths = create_loss
        return cell, test_bound, global_step, trajectories, test_lengths

    cell, test_bound, global_step, trajectories, test_lengths = create_graph()

    sess = U.get_session()
    U.initialize()
    cur_step = 0

    # saver
    saver = tf.train.Saver(max_to_keep=1)

    if not os.path.exists(config.logdir):
        assert False
    ckpt = tf.train.get_checkpoint_state(config.logdir + '/valid_best')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        cur_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print('Model and log loaded! (checkpoint_path=%s, cur_step=%d)' % (ckpt.model_checkpoint_path, cur_step))
    test_bound_value = sess.run([test_bound])

    print ("##################################")
    print ("VALID_BEST_STEP: %s" % cur_step)
    print ("PARTICLE_NUM: %s" % config.num_samples)
    print ("TEST_BOUND: %s" % test_bound_value[0])
    print("##################################")

