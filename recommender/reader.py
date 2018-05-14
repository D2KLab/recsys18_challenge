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


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import time


def _read_items(filename, dataset):

    items = []

    for i, playlist in enumerate(dataset.reader('playlists_%s.csv' % filename, 'items_%s.csv' % filename)):

        items = items + list(map(lambda x: str(x), playlist['items']))

        items.append('<eos>')

        print('read playlist %s: %d' % (filename, i))

    return items


def _build_vocab(data):

    items = list(set(data))
    item_to_id = {item: i for i, item in enumerate(items)}

    return item_to_id


def _file_to_word_ids(data, word_to_id):

    return [word_to_id[word] for word in data if word in word_to_id]


def read_raw_data(dataset=None):
    """Load MPD dataset from data directory "data_path".

    pid,pos,track_id

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """
    print('reading training data')
    train_data = _read_items("training", dataset)
    word_to_id = _build_vocab(train_data)
    train_data = _file_to_word_ids(train_data, word_to_id)
    print('reading validation data')
    valid_data = _read_items("validation", dataset)
    valid_data = _file_to_word_ids(valid_data, word_to_id)
    print('reading test data')
    test_data = _read_items("test", dataset)
    test_data = _file_to_word_ids(test_data, word_to_id)

    vocabulary = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
