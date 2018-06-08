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

import numpy as np


def _read_items(filename, dataset, units='items'):

    items = []

    for i, playlist in enumerate(dataset.reader('playlists_%s.csv' % filename, 'items_%s.csv' % filename)):

        items.extend(playlist[units])
        items.append(0)  # <eos>
        print('read playlist %s: %d' % (filename, i))

    return items


def ptb_raw_data(dataset):

    train_data = {}
    valid_data = {}

    print('reading training data')
    train_data['tracks'] = _read_items("training", dataset, units='items')
    train_data['albums'] = _read_items("training", dataset, units='albums')
    train_data['artists'] = _read_items("training", dataset, units='artists')
    print('reading validation data')
    valid_data['tracks'] = _read_items("validation", dataset, units='items')
    valid_data['albums'] = _read_items("validation", dataset, units='albums')
    valid_data['artists'] = _read_items("validation", dataset, units='artists')

    assert len(train_data['tracks']) == len(train_data['albums']) == len(train_data['artists'])

    return train_data, valid_data


def ptb_iterator(raw_data, batch_size, num_steps):
    # TODO

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
