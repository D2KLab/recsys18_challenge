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


"""Utilities for parsing MPD files into arrays for the RNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def _read_items(filename, dataset, units='items'):

    items = []

    for i, playlist in enumerate(dataset.reader('playlists_%s.csv' % filename, 'items_%s.csv' % filename)):

        items.extend(playlist[units])

        items.append(0)  # index for <eos>

        print('read playlist %s: %d' % (filename, i))

    return items


def read_raw_data(dataset=None):

    """Load MPD dataset from data directory "data_path".

    pid,pos,track_id

    Args:
      dataset: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_data = {}
    valid_data = {}
    test_data = {}

    print('reading training data')
    train_data['tracks'] = _read_items("training", dataset, units='items')
    train_data['albums'] = _read_items("training", dataset, units='albums')
    train_data['artists'] = _read_items("training", dataset, units='artists')
    print('reading validation data')
    valid_data['tracks'] = _read_items("validation", dataset, units='items')
    valid_data['albums'] = _read_items("validation", dataset, units='albums')
    valid_data['artists'] = _read_items("validation", dataset, units='artists')
    print('reading test data')
    test_data['tracks'] = _read_items("test", dataset, units='items')
    test_data['albums'] = _read_items("test", dataset, units='albums')
    test_data['artists'] = _read_items("test", dataset, units='artists')

    vocabulary = len(dataset.tracks_uri2id) + 1

    assert len(train_data['tracks']) == len(train_data['albums']) == len(train_data['artists'])

    return train_data, valid_data, test_data, vocabulary

"""
def ptb_producer(raw_data, batch_size, num_steps, name=None):
    Iterate on the raw PTB data.

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
    
   

    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):

        raw_data_tracks = tf.convert_to_tensor(raw_data['tracks'], name="tracks", dtype=tf.int32)

        data_len = tf.size(raw_data_tracks)
        batch_len = data_len // batch_size
        data_tracks = tf.reshape(raw_data_tracks[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

        x_tracks = tf.strided_slice(data_tracks, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])

        x_tracks.set_shape([batch_size, num_steps])

        y = tf.strided_slice(data_tracks, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

        raw_data_albums = tf.convert_to_tensor(raw_data['albums'], name="albums", dtype=tf.int32)

        data_albums = tf.reshape(raw_data_albums[0: batch_size * batch_len],
                          [batch_size, batch_len])

        x_albums = tf.strided_slice(data_albums, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])

        x_albums.set_shape([batch_size, num_steps])

        raw_data_artists = tf.convert_to_tensor(raw_data['artists'], name="artists", dtype=tf.int32)

        data_artists = tf.reshape(raw_data_artists[0: batch_size * batch_len],
                          [batch_size, batch_len])

        x_artists = tf.strided_slice(data_artists, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])

        x_artists.set_shape([batch_size, num_steps])

        return x_tracks, x_albums, x_artists, y
"""

def ptb_iterator(raw_data, batch_size, num_steps):

    """Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
      ValueError: if batch_size or num_steps are too high.
    """

    raw_data_tracks = np.array(raw_data['tracks'], dtype=np.int32)
    raw_data_albums = np.array(raw_data['albums'], dtype=np.int32)
    raw_data_artists = np.array(raw_data['artists'], dtype=np.int32)

    data_len = len(raw_data_tracks)
    batch_len = data_len // batch_size

    data_tracks = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_albums = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_artists = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data_tracks[i] = raw_data_tracks[batch_len * i:batch_len * (i + 1)]
        data_albums[i] = raw_data_albums[batch_len * i:batch_len * (i + 1)]
        data_artists[i] = raw_data_artists[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x_tracks = data_tracks[:, i * num_steps:(i + 1) * num_steps]
        x_albums = data_albums[:, i * num_steps:(i + 1) * num_steps]
        x_artists = data_artists[:, i * num_steps:(i + 1) * num_steps]
        y = data_tracks[:, i * num_steps + 1:(i + 1) * num_steps + 1]

        yield (x_tracks, x_albums, x_artists, y)
