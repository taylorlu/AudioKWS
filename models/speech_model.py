from __future__ import division

import math
import numpy as np
import tensorflow as tf
from user_ops import warp_ctc_ops
from modules import conv1d_banks, conv1d, normalize, highwaynet, gru

class SpeechModel(object):

    def __init__(self):
        self._init_inference = False
        self._init_cost = False
        self._init_train = False

    def init_inference(self, config, is_training=False):
        num_banks = config['num_banks']
        hidden_units = config['hidden_units']
        num_highway = config['num_highway']
        norm_type = config['norm_type']
        batch_size = config['batch_size']
        self._input_dim = input_dim = config['input_dim']
        self._output_dim = output_dim = config['alphabet_size']

        self._inputs = tf.placeholder(tf.float32, [batch_size, None, input_dim])
        self._seq_lens = tf.placeholder(tf.int32, shape=batch_size)
        self._out_lens = self._seq_lens

        # TODO, awni, for now on the client to remember to initialize these.
        self._mean = tf.get_variable("mean",
                        shape=input_dim, trainable=False)
        self._std = tf.get_variable("std",
                        shape=input_dim, trainable=False)

        std_inputs = (self._inputs - self._mean) / self._std

        # x = tf.layers.dense(self._inputs, units=hidden_units, activation=tf.nn.relu)  # (n, t, h)

        x = conv1d(self._inputs, hidden_units, 1, scope="conv1d")

        out = conv1d_banks(x, K=num_banks, num_units=hidden_units, norm_type=norm_type,
                           is_training=is_training)  # (n, t, k * h)

        out = tf.layers.max_pooling1d(out, 2, 1, padding="same")  # (n, t, k * h)

        out = conv1d(out, hidden_units, 3, scope="conv1d_1")  # (n, t, h)
        out = normalize(out, type=norm_type, is_training=is_training, activation_fn=tf.nn.relu)
        out = conv1d(out, hidden_units, 3, scope="conv1d_2")  # (n, t, h)

        out += x  # (n, t, h) # residual connections

        for i in range(num_highway):
            out = highwaynet(out, num_units=hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

        rnn_out, state = gru(out, hidden_units, False, seqlens=self._seq_lens)  # (n, t, h)

        self._rnn_state = state
        rnn_out = tf.transpose(rnn_out, [1, 0, 2])

        # Collapse time and batch dims pre softmax.
        rnn_out = tf.reshape(rnn_out, (-1, hidden_units))
        logits, probas = _add_softmax_linear(rnn_out, hidden_units,
                                             output_dim, initializer=tf.contrib.layers.xavier_initializer())
        # Reshape to time-major.
        self._logits = tf.reshape(logits, (-1, batch_size, output_dim))
        self._probas = tf.reshape(probas, (-1, batch_size, output_dim))

        self._init_inference = True

    def init_cost(self):
        assert self._init_inference, "Must init inference before cost."

        self._labels = tf.placeholder(tf.int32)
        self._label_lens = tf.placeholder(tf.int32)

        losses = warp_ctc_ops.warp_ctc_loss(self.logits, self._out_lens,
                                            self._labels, self._label_lens)
        self._cost = tf.reduce_mean(losses)

        self._init_cost = True

    def init_train(self, config):
        assert self._init_inference, "Must init inference before train."
        assert self._init_cost, "Must init cost before train."

        learning_rate = config['learning_rate']
        self._momentum_val = config['momentum']
        max_grad_norm = config['max_grad_norm']
        decay_steps = config['lr_decay_steps']
        decay_rate = config['lr_decay_rate']

        self._momentum = tf.Variable(0.5, trainable=False)
        self._global_step = step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate, step,
                    decay_steps, decay_rate, staircase=True)

        ema = tf.train.ExponentialMovingAverage(0.99, name="avg")
        avg_cost_op = ema.apply([self.cost])
        self._avg_cost = ema.average(self.cost)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        scaled_grads, norm = tf.clip_by_global_norm(grads, max_grad_norm)

        optimizer = tf.train.MomentumOptimizer(self.lr, self._momentum)
        with tf.control_dependencies([avg_cost_op]):
            self._train_op = optimizer.apply_gradients(zip(scaled_grads, tvars),
                                 global_step=step)

        self._grad_norm = norm
        self._init_train = True

    def feed_dict(self, inputs, labels=None):
        """
        Constructs the feed dictionary from given inputs necessary to run
        an operations for the model.

        Args:
            inputs : List of 2D numpy array input spectrograms. Should be
                of shape [input_dim x time]
            labels : List of labels for each item in the batch. Each label
                should be a list of integers. If label=None does not feed the
                label placeholder (for e.g. inference only).

        Returns:
            A dictionary of placeholder keys and feed values.
        """
        sequence_lengths = [d.shape[1] for d in inputs]
        feed_dict = { self._inputs : _batch_major(inputs),
                      self._seq_lens : sequence_lengths}
        if labels:
            values = [l for label in labels for l in label]
            label_lens = [len(label) for label in labels]
            label_dict = { self._labels : values,
                           self._label_lens : label_lens }
            feed_dict.update(label_dict)

        return feed_dict

    def start_momentum(self, session):
        m = self._momentum.assign(self._momentum_val)
        session.run([m])

    def set_mean_std(self, mean, std, session):
        m = self._mean.assign(mean)
        s = self._std.assign(std)
        session.run([m, s])

    @property
    def cost(self):
        assert self._init_cost, "Must init cost."
        return self._cost

    @property
    def avg_cost(self):
        assert self._init_train, "Must init train."
        return self._avg_cost

    @property
    def grad_norm(self):
        assert self._init_train, "Must init train."
        return self._grad_norm

    @property
    def global_step(self):
        assert self._init_train, "Must init train."
        return self._global_step

    @property
    def input_dim(self):
        assert self._init_inference, "Must init inference."
        return self._input_dim

    @property
    def logits(self):
        assert self._init_inference, "Must init inference."
        return self._logits

    @property
    def output_dim(self):
        assert self._init_inference, "Must init inference."
        return self._output_dim

    @property
    def output_lens(self):
        assert self._init_inference, "Must init inference."
        return self._out_lens

    @property
    def probabilities(self):
        assert self._init_inference, "Must init inference."
        return self._probas

    @property
    def state(self):
        assert self._init_inference, "Must init inference."
        return self._rnn_state

    @property
    def train_op(self):
        assert self._init_train, "Must init train."
        return self._train_op

def _add_softmax_linear(inputs, input_dim, output_dim, initializer):
    with tf.variable_scope("softmax", initializer=initializer):
        W_softmax = tf.get_variable("softmax_W",
                        shape=(input_dim, output_dim))
        b_softmax = tf.get_variable("softmax_b", shape=(output_dim),
                        initializer=tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(inputs, W_softmax), b_softmax)
    probas = tf.nn.softmax(logits)
    return logits, probas

def _batch_major(data):
    """
    Reshapes a batch of spectrogram arrays into
    a single tensor [batch_size x max_time x input_dim].

    Args :
        data : list of 2D numpy arrays of [input_dim x time]
    Returns :
        A 3d tensor with shape
        [batch_size x max_time x input_dim]
        and zero pads as necessary for data items which have
        fewer time steps than max_time.
    """
    max_time = max(d.shape[1] for d in data)
    batch_size = len(data)
    input_dim = data[0].shape[0]
    all_data = np.zeros((batch_size, max_time, input_dim),
                        dtype=np.float32)
    for e, d in enumerate(data):
        all_data[e, :d.shape[1], :] = d.T
    return all_data
