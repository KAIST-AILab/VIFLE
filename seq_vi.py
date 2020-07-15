from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import runners
from utils.misc_util import set_global_seeds

DATASET_PATH = "datasets"

SAVE_PATH = "checkpoints"

# default setting
PIANOROLL_DEFAULT_DATA_DIMENSION = 88

# select options
tf.app.flags.DEFINE_string("logdir", SAVE_PATH, "The directory to keep checkpoints in.")
tf.app.flags.DEFINE_string("latent_type", "normal",
                           "Type of latent variables. Currently, 'normal' and 'bernoulli' are supported")
tf.app.flags.DEFINE_string("algorithm", "reparam",
                           "Algorithm to be executed. Currently, 'reparam', 'reinforce', 'vimco', 'fr', and 'vifle' are supported.")
tf.app.flags.DEFINE_string("cell", "vrnn", "The cell choice. Currently 'vrnn' cell is supported.")
tf.app.flags.DEFINE_string("dataset_name", "jsb", "Dataset name")

# Bound options
tf.app.flags.DEFINE_string("bound", "iwae", "The bound to optimize. Can be 'iwae'.")
tf.app.flags.DEFINE_string("test_bound", "iwae", "The bound to optimize. Can be 'iwae'.")

# Dataset and save options
tf.app.flags.DEFINE_string("dataset_type", "pianoroll", "The type of dataset, 'pianoroll', 'gaussian', or 'bernoulli'.")
tf.app.flags.DEFINE_integer("data_dimension", 10, "The dimension of dimension")
tf.app.flags.DEFINE_string("dataset_path", DATASET_PATH, "Path to load the dataset from.")
tf.app.flags.DEFINE_string("mode", 'train', "The mode either 'train' or 'eval' or 'analysis'.")
tf.app.flags.DEFINE_boolean("model_train", True, "Whether to train the model.")
tf.app.flags.DEFINE_integer("init_steps", 20000, "Start steps will not be saved.")
tf.app.flags.DEFINE_integer("save_every",  10000, "Save every")

# Algorithm details
tf.app.flags.DEFINE_integer("latent_size", 10, "The size of the latent state of the model.")
tf.app.flags.DEFINE_integer("num_hidden_prior", 1, "The number of hidden layers for prior.")
tf.app.flags.DEFINE_integer("num_hidden_gen", 1, "The number of hidden layers for generative.")
tf.app.flags.DEFINE_integer("num_hidden_post", 1, "The number of hidden layers for posterior.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 4, "The number of samples (or particles) for multisample algorithms.")
tf.app.flags.DEFINE_integer("test_num_samples", 128, "The number of samples (or particles) for multisample algorithms.")
tf.app.flags.DEFINE_integer("random_seed", 0, "A random seed for seeding the TensorFlow graph / numpy")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of p and q.")
tf.app.flags.DEFINE_float("fef_learning_rate", 1e-4, "The learning rate of future estimate function.")
tf.app.flags.DEFINE_float("sigma_min", 1e-5, "The sigma_min for normal distribution to prevent nan.")
tf.app.flags.DEFINE_float("raw_sigma_bias", 0.0, "The sigma bias for normal distribution to prevent nan.")
tf.app.flags.DEFINE_float("temperature", 10.0, "Temperature for relaxed distributions.")
tf.app.flags.DEFINE_integer("max_iter", 500000, "Maximum number of iteration of update.")

# For simple run
tf.app.flags.DEFINE_integer("pid", 0, "Condor PID")

FLAGS = tf.app.flags.FLAGS


simple_run_settings = []
simple_run_settings.append(('iwae','jsb','reparam', 'normal', 3e-5, 4, 0, 200001))
simple_run_settings.append(('iwae','jsb','vifle', 'normal', 3e-5, 4, 0, 200001))



def main(unused_argv):
    args = simple_run_settings[FLAGS.pid]
    FLAGS.bound, FLAGS.dataset_name, FLAGS.algorithm, FLAGS.latent_type, FLAGS.learning_rate, FLAGS.num_samples,\
    FLAGS.random_seed, FLAGS.max_iter = args
    if FLAGS.dataset_name == 'jsb':
        FLAGS.model_train = True
        FLAGS.dataset_type = 'pianoroll'
        FLAGS.data_dimension = PIANOROLL_DEFAULT_DATA_DIMENSION
        FLAGS.latent_size = 32
        FLAGS.batch_size = 4
        extension = "pkl"
    elif FLAGS.dataset_name == 'nottingham':
        FLAGS.model_train = True
        FLAGS.dataset_type = 'pianoroll'
        FLAGS.data_dimension = PIANOROLL_DEFAULT_DATA_DIMENSION
        FLAGS.latent_size = 64
        FLAGS.batch_size = 4
        extension = "pkl"
    elif FLAGS.dataset_name == 'musedata':
        FLAGS.model_train = True
        FLAGS.dataset_type = 'pianoroll'
        FLAGS.data_dimension = PIANOROLL_DEFAULT_DATA_DIMENSION
        FLAGS.latent_size = 64
        FLAGS.batch_size = 4
        extension = "pkl"
    elif FLAGS.dataset_name == 'pianomidide':
        FLAGS.model_train = True
        FLAGS.dataset_type = 'pianoroll'
        FLAGS.data_dimension = PIANOROLL_DEFAULT_DATA_DIMENSION
        FLAGS.latent_size = 64
        FLAGS.batch_size = 4
        extension = "pkl"
    elif FLAGS.dataset_name == 'gaussian':
        FLAGS.model_train = False
        FLAGS.dataset_type = 'synthetic'
        FLAGS.max_iter = 50000
        extension = "npy"
    elif FLAGS.dataset_name == 'bernoulli':
        FLAGS.model_train = False
        FLAGS.dataset_type = 'synthetic'
        FLAGS.max_iter = 50000
        extension = "npy"
    else:
        raise ValueError("Undefined dataset_name %s" % FLAGS.dataset_name)
    FLAGS.dataset_path = "%s/%s/%s.%s" % (DATASET_PATH, FLAGS.dataset_type, FLAGS.dataset_name, extension)
    FLAGS.logdir = "%s/%s/%s" % (SAVE_PATH, FLAGS.dataset_name, FLAGS.algorithm)

    print('The result will be saved in %s' % FLAGS.logdir)
    set_global_seeds(FLAGS.random_seed)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()
    if FLAGS.mode == "train":
        runners.run(FLAGS)
    elif FLAGS.mode == "eval":
        runners.run_eval(FLAGS)


if __name__ == '__main__':
    tf.app.run()
