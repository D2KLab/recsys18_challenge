# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys

sys.stdout = sys.stderr
import numpy as np
import tensorflow as tf
import ptb_reader as reader

from os import path
from gensim.models import Word2Vec

import sys
sys.path.append('.')
from recommender.baseline import MostPopular
from utils.dataset import Dataset

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small",
                    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_string("sample_file", None,
                    "Must have trained model ready. Only does sampling")
flags.DEFINE_string("seed_for_sample", "spotify:track:1mea3bSkSGXuIRvnydlB5b",
                    "supply seeding phrase here. it must only contain words from vocabulary")
flags.DEFINE_bool("rank", False,
                  "use do_rank instead of do_sample")
flags.DEFINE_string("embs", None,
                    "the path with the embeddings")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def embeddings(vocab_size, size, dataset):
    if FLAGS.embs is None:
        return tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
    else:
        w2v_tracks = Word2Vec.load(path.join(FLAGS.embs, 'word2rec_dry.w2v'))
        w2v_albums = Word2Vec.load(path.join(FLAGS.embs, 'word2rec_dry_albums.w2v'))
        w2v_artists = Word2Vec.load(path.join(FLAGS.embs, 'word2rec_dry_artists.w2v'))

        embs = np.zeros((vocab_size, 300), dtype=np.float32)

        for i in range(1, vocab_size):
            album_id = dataset.tracks_id2album[i]
            artist_id = dataset.tracks_id2artist[i]
            embs[i] = np.concatenate((w2v_tracks.wv[str(i)],
                                      w2v_albums.wv[str(album_id)],
                                      w2v_artists.wv[str(artist_id)]))

        embs_initializer = tf.constant_initializer(embs)

        return tf.get_variable(name="embedding",
                               shape=(vocab_size, 300),
                               initializer=embs_initializer,
                               trainable=False, dtype=data_type())


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, dataset):

        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = embeddings(vocab_size, size, dataset)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_step, [1])
        #           for input_step in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        self.logits = logits
        self.sample = tf.multinomial(logits, 1)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        if config.optimizer == 'RMSPropOptimizer':
            optimizer = tf.train.RMSPropOptimizer(self._lr)
        elif config.optimizer == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer()
        elif config.optimizer == 'MomentumOptimizer':
            optimizer = tf.train.MomentumOptimizer(self._lr, momentum=0.8, use_nesterov=True)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    optimizer = 'GradientDescentOptimizer'
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 10
    hidden_size = 50
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20


class MediumConfig(object):
    """Medium config."""
    optimizer = 'GradientDescentOptimizer'
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20


class LargeConfig(object):
    """Large config."""
    optimizer = 'GradientDescentOptimizer'
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20


class TestConfig(object):
    """Tiny config, for testing."""
    optimizer = 'GradientDescentOptimizer'
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 10
    hidden_size = 3
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20


def do_rank(session, model, data, num_samples):
    """Sampled from the model"""
    samples = []
    state = session.run(model.initial_state)
    fetches = [model.final_state, model.logits]
    logits = None

    # for all the seeds
    for x in data:
        feed_dict = {}
        feed_dict[model.input_data] = [[x]]

        for layer_num, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[layer_num].c
            feed_dict[h] = state[layer_num].h

        state, logits = session.run(fetches, feed_dict)

    sorted_items = np.argsort(logits[0])[::-1]
    i = 0

    while len(samples) < num_samples:
        item = sorted_items[i]
        if item not in data and item != 0:
            samples.append(item)
        i += 1

    assert 0 not in samples
    assert len(samples) == num_samples
    assert len(list(set(samples))) == num_samples

    return samples


def do_sample(session, model, data, num_samples):
    """Sampled from the model"""
    samples = []
    state = session.run(model.initial_state)
    fetches = [model.final_state, model.sample]
    sample = None

    # for all the seeds
    for i, x in enumerate(data):
        feed_dict = {}
        feed_dict[model.input_data] = [[x]]

        for layer_num, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[layer_num].c
            feed_dict[h] = state[layer_num].h

        state, sample = session.run(fetches, feed_dict)

        if i == len(data) - 1:
            while sample is None or sample[0][0] in data or sample[0][0] == 0:
                state, sample = session.run(fetches, feed_dict)

    samples.append(sample[0][0])
    k = 1

    # for all the samples
    while k < num_samples:
        feed_dict = {}
        feed_dict[model.input_data] = [[samples[-1]]]

        for layer_num, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[layer_num].c
            feed_dict[h] = state[layer_num].h

        state, sample = session.run(fetches, feed_dict)

        # avoid suggesting <eos>
        if sample[0][0] == 0:
            continue

        # avoid suggesting a seed
        if sample[0][0] in data:
            continue

        # avoid suggesting duplicated items
        if sample[0][0] in samples:
            continue

        samples.append(sample[0][0])
        k += 1

    assert 0 not in samples
    assert len(samples) == num_samples
    assert len(list(set(samples))) == num_samples

    return samples


def run_epoch(session, model, data, is_train=False, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        if is_train:
            fetches = [model.cost, model.final_state, model.train_op]
        else:
            fetches = [model.cost, model.final_state]

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        for layer_num, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[layer_num].c
            feed_dict[h] = state[layer_num].h

        if is_train:
            cost, state, _ = session.run(fetches, feed_dict)
        else:
            cost, state = session.run(fetches, feed_dict)

        costs += cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def pretty_print(items, id2word):
    uris = []
    for x in items:
        try:
            uris.append(id2word[x])
        except KeyError:
            continue
    return ' '.join(uris)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    dataset = Dataset(FLAGS.data_path)

    # the number of tracks plus <eos>
    vocab_size = len(dataset.tracks_uri2id) + 1

    config = get_config()
    config.vocab_size = vocab_size
    eval_config = get_config()

    eval_config.vocab_size = vocab_size
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    seed_for_sample = FLAGS.seed_for_sample.split()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Train"):

            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, dataset=dataset)
                tf.summary.scalar("Training Loss", m.cost)
                tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, dataset=dataset)
                tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, dataset=dataset)

        saver = tf.train.Saver(name='saver', write_version=tf.train.SaverDef.V2)
        sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=0, save_summaries_secs=0, saver=saver)

        old_valid_perplexity = 10000000000.0

        # sessconfig = tf.ConfigProto(allow_soft_placement=True)
        # sessconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with sv.managed_session() as session:

            if FLAGS.sample_file is not None:

                # TODO should be title2rec
                fallback = MostPopular(dataset, dry=True)
                writer = dataset.writer(FLAGS.sample_file)

                for i, playlist in enumerate(dataset.reader('playlists_test.csv', 'items_test_x.csv')):
                    print('sampling playlist', i)

                    if len(playlist['items']) == 0:
                        fallback.recommend(playlist)
                    else:
                        if FLAGS.rank:
                            playlist['items'] = do_rank(session, mtest, playlist['items'], 500)
                        else:
                            playlist['items'] = do_sample(session, mtest, playlist['items'], 500)

                    writer.write(playlist)

            else:

                train_data, valid_data = reader.ptb_raw_data(dataset)
                print('Distinct terms: %d' % vocab_size)

                for i in range(config.max_max_epoch):

                    print("Seed: %s" % pretty_print([dataset.tracks_uri2id[x] for x in seed_for_sample],
                                                    dataset.tracks_id2uri))
                    print("Sample: %s" % pretty_print(
                        do_sample(session, mtest, [dataset.tracks_uri2id[word] for word in seed_for_sample],
                                  max(5 * (len(seed_for_sample) + 1), 10)), dataset.tracks_id2uri))

                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
                    m.assign_lr(session, config.learning_rate * lr_decay)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity = run_epoch(session, m, train_data, is_train=True, verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                    valid_perplexity = run_epoch(session, mvalid, valid_data)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                    if valid_perplexity < old_valid_perplexity:
                        old_valid_perplexity = valid_perplexity
                        sv.saver.save(session, FLAGS.save_path, i)
                    elif valid_perplexity >= 1.3 * old_valid_perplexity:
                        if len(sv.saver.last_checkpoints) > 0:
                            sv.saver.restore(session, sv.saver.last_checkpoints[-1])
                        break
                    else:
                        if len(sv.saver.last_checkpoints) > 0:
                            sv.saver.restore(session, sv.saver.last_checkpoints[-1])
                        lr_decay *= 0.5

                # test_perplexity = run_epoch(session, mtest, test_data)
                # print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
