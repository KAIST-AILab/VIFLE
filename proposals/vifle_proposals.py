import tensorflow as tf
import numpy as np

from utils import tf_util as U
from scipy.special import logsumexp


def _convert_raw_seg_to_seg(raw_seg, run_lengths, alg_name, min_value, max_value):
    s, b, p = raw_seg['log_weights'].shape  # sequence length, batch size, number of particles
    act_dim = raw_seg['ac'].shape[-1]
    state_dim = raw_seg['ob'].shape[-1]

    mask = np.transpose(np.reshape(raw_seg['mask'], (s, p, b)),(0,2,1))
    ob = np.transpose(np.reshape(raw_seg['ob'], (s, p, b, state_dim)),(0,2,1,3))
    ac = np.transpose(np.reshape(raw_seg['ac'], (s, p, b, act_dim)),(0,2,1,3))

    log_p_div_q = raw_seg['log_p_z'] + raw_seg['log_p_x_given_z'] - raw_seg['log_q_z']
    log_p_div_q = np.transpose(np.reshape(log_p_div_q, (s, p, b)),(0,2,1))
    log_p_div_q = log_p_div_q * mask

    log_p_xz = raw_seg['log_p_z'] + raw_seg['log_p_x_given_z']
    log_p_xz = np.transpose(np.reshape(log_p_xz, (s, p, b)),(0,2,1))
    log_p_xz = log_p_xz * mask

    initial_pr = np.zeros((1, b, p))
    cum_prs = np.cumsum(log_p_div_q, axis=0)
    cum_prs = np.concatenate((initial_pr, cum_prs[:-1,:,:]), axis=0)    # (s,b,p)

    ob = np.reshape(ob, (s*b*p, state_dim))
    if alg_name == "reinforce":
        current_fef = 0
        future_fef = 0
        fef_weights = 0
        future_fef_weights = 0

        # compute hiah_var_coeff
        high_var_coeff = log_p_div_q - np.log(p)
    elif alg_name == "vimco":
        current_fef = 0
        future_fef = 0
        fef_weights = 0
        future_fef_weights = 0

        # compute hiah_var_coeff
        vimco_numerators = np.zeros((b,1,p))
        vimco_denominators = np.zeros((b, 1, p))

        for i in range(p):
            total_sum = np.sum(log_p_div_q * mask, axis=0)

            vimco_numerators = np.concatenate((vimco_numerators, np.reshape(total_sum, (b,1,p))), axis=1)

            total_sum[:,i] = (np.sum(total_sum, axis=-1) - total_sum[:,i]) / (p-1)
            total_sum = np.reshape(total_sum, (b,1,p))
            vimco_denominators = np.concatenate((vimco_denominators, total_sum), axis=1)

        vimco_numerators = vimco_numerators[:,1:,:]
        vimco_numerators = logsumexp(vimco_numerators, axis=-1)

        vimco_denominators = vimco_denominators[:, 1:, :]
        vimco_denominators = logsumexp(vimco_denominators, axis=-1)
        high_var_coeff = vimco_numerators - vimco_denominators
    elif alg_name == "vifle" or alg_name == "fr":
        current_fef = np.transpose(np.reshape(raw_seg['fef_vpred'], (s, p, b)), (0,2,1))
        current_fef = current_fef * mask  # (s,b,p)

        # future_value: log(V)
        future_fef = np.concatenate((current_fef[1:,:,:], np.zeros((1,b,p))), axis=0)
        fef_weights = np.zeros([s,b])
        future_fef_weights = np.zeros([s,b])
        for i in range(b):
            fef_weights[:run_lengths[i], i] = np.arange(run_lengths[i])[::-1] + 1
            future_fef_weights[:run_lengths[i] - 1, i] = np.arange(run_lengths[i] - 1)[::-1] + 1
        # compute hiah_var_coeff
        if alg_name == "vifle":
            total_sum = np.sum(log_p_div_q * mask, axis=0, keepdims=True)  # (1,b,p)
            total_sum = np.broadcast_to(total_sum, (s,b,p))
            vifle_numerators = np.tile(np.expand_dims(total_sum, axis=3),(1,1,1,p))
            vifle_denominators = np.tile(np.expand_dims(total_sum, axis=3),(1,1,1,p))

            for i in range(p):
                vifle_numerators[:,:,i,i] = (cum_prs + log_p_div_q)[:, :, i] + future_fef[:,:,i] * future_fef_weights
                vifle_denominators[:,:,i,i] = cum_prs[:, :, i] + current_fef[:,:,i] * fef_weights

            vifle_numerators = logsumexp(vifle_numerators, axis=-1)
            vifle_denominators = logsumexp(vifle_denominators, axis=-1)
            high_var_coeff = vifle_numerators - vifle_denominators
        else:
            fr_numerators = cum_prs + log_p_div_q +  future_fef * future_fef_weights
            fr_numerators = logsumexp(fr_numerators, axis=-1)

            fr_denominators = cum_prs + current_fef * fef_weights
            fr_denominators = logsumexp(fr_denominators, axis=-1)
            high_var_coeff = fr_numerators - fr_denominators
    else:
        raise ValueError("Undefined alg_name %s" % alg_name)
    seg = {
        'ob': ob,
        'ac': ac,
        'mask': mask,
        'log_p_xz': log_p_xz,
        'high_var_coeff': high_var_coeff,
        'target_pr': log_p_div_q,
        'future_fef': future_fef,
        'current_fef': current_fef,
        'fef_weights': fef_weights,
        'future_fef_weights': future_fef_weights
    }
    return seg


class REPARAM(object):
    def __init__(self, ob, ac_shape, prop_fn, fe_fn,batch_size, um_samples, learning_rate):
        self.prop_phi = prop_fn(ob, ac_shape, name='prop_phi')
        self.fef_psi = fe_fn(ob, name='fef_psi')


class REINFORCE(object):
    def _rew_update(self, b, p, ob, ac, mask, lengths, fe_fn, fef_stepsize):
        pass

    def __init__(self, ob, ac_shape, prop_fn, fe_fn, batch_size, num_samples, prop_stepsize=1e-4, fef_stepsize=1e-4):
        b, p = batch_size, num_samples

        ac = tf.placeholder(dtype=tf.float32, shape=[None, b, p, ac_shape], name='ac')
        mask = tf.placeholder(dtype=tf.float32, shape=[None, b, p], name='mask')
        log_p_xz = tf.placeholder(dtype=tf.float32, shape=[None, b, p], name='log_p_xz')
        high_var_coeff = tf.placeholder(dtype=tf.float32, shape=[None, b, p], name='high_var_coeff')
        lengths = tf.placeholder(dtype=tf.float32, shape=[b], name='lengths')

        reshape_ac = tf.reshape(ac, (tf.shape(ac)[0] * b * p, tf.shape(ac)[-1]))

        self.prop_phi = prop_fn(ob, ac_shape, name='prop_phi')
        self.fef_psi = fe_fn(ob, name='fef_psi')

        prop_var_list = self.prop_phi.get_trainable_variables()

        log_q = tf.reduce_sum(self.prop_phi.pd.log_prob(reshape_ac), axis=-1)
        log_q = mask * tf.reshape(log_q, (tf.shape(ac)[0], b, p))

        objective_1 = tf.reduce_sum(tf.reduce_sum(high_var_coeff * log_q, axis=-1), axis=0)
        objective_2 = tf.reduce_logsumexp(tf.reduce_sum(log_p_xz - log_q, axis=0), axis=-1) - tf.log(tf.to_float(p))

        adam = tf.train.AdamOptimizer(learning_rate=prop_stepsize)

        objective = tf.reduce_mean(objective_1 + objective_2)
        self.apply_grad = U.function([ob, ac, mask, log_p_xz, high_var_coeff, lengths], adam.minimize(-objective, var_list=prop_var_list))

        self._rew_update(b, p, ob, ac, mask, lengths, fe_fn, fef_stepsize)

    def update(self, raw_seg, lengths, min_value=1e-20, max_value=None):
        seg = _convert_raw_seg_to_seg(raw_seg, lengths, 'reinforce', min_value, max_value)
        self.apply_grad(seg['ob'], seg['ac'], seg['mask'], seg['log_p_xz'], seg['high_var_coeff'], lengths)


class VIMCO(REINFORCE):
    def update(self, raw_seg, lengths, min_value=1e-20, max_value=None):
        seg = _convert_raw_seg_to_seg(raw_seg, lengths, 'vimco', min_value, max_value)
        self.apply_grad(seg['ob'], seg['ac'], seg['mask'], seg['log_p_xz'], seg['high_var_coeff'], lengths)


class VIFLE(REINFORCE):
    def _rew_update(self, b, p, ob, ac, mask, lengths, fe_fn, fef_stepsize):
        target_pr = tf.placeholder(dtype=tf.float32, shape=[None, b, p], name='target_pr')
        future_fef = tf.placeholder(dtype=tf.float32, shape=[None, b, p], name='future_fef')

        fef_weights = tf.placeholder(dtype=tf.float32, shape=[None, b], name='fef_weights')
        future_fef_weights = tf.placeholder(dtype=tf.float32, shape=[None, b], name='future_fef_weights')

        weighted_fef = tf.reshape(tf.squeeze(self.fef_psi.fef(ob)), (tf.shape(ac)[0], b, p)) * tf.expand_dims(fef_weights, axis=2)
        weighted_future_fef = future_fef * tf.expand_dims(future_fef_weights, axis=2)

        rew_vf_var_list = self.fef_psi.get_trainable_variables()
        rew_vf_err = tf.reduce_mean(tf.square((weighted_fef - target_pr - weighted_future_fef)) * mask / tf.expand_dims(tf.expand_dims(lengths, axis=0), axis=2))
        vf_adam = tf.train.AdamOptimizer(learning_rate=fef_stepsize)
        self.apply_rew_vf_grad = U.function([ob, ac, target_pr, mask, future_fef, fef_weights, future_fef_weights, lengths], vf_adam.minimize(rew_vf_err, var_list=rew_vf_var_list))

    def update(self, raw_seg, lengths, min_value=1e-20, max_value=None):
        seg = _convert_raw_seg_to_seg(raw_seg, lengths, 'vifle', min_value, max_value)
        for _ in range(1):
            self.apply_rew_vf_grad(seg['ob'], seg['ac'], seg['target_pr'], seg['mask'], seg['future_fef']
                                   , seg['fef_weights'], seg['future_fef_weights'], lengths)
        self.apply_grad(seg['ob'], seg['ac'], seg['mask'], seg['log_p_xz'], seg['high_var_coeff'], lengths)


class FR(VIFLE):
    def update(self, raw_seg, lengths, min_value=1e-20, max_value=None):
        seg = _convert_raw_seg_to_seg(raw_seg, lengths, 'fr', min_value, max_value)
        for _ in range(1):
            self.apply_rew_vf_grad(seg['ob'], seg['ac'], seg['target_pr'], seg['mask'], seg['future_fef']
                                   , seg['fef_weights'], seg['future_fef_weights'], lengths)
        self.apply_grad(seg['ob'], seg['ac'], seg['mask'], seg['log_p_xz'], seg['high_var_coeff'], lengths)
