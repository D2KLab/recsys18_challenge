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


def _read_items(filename, dataset):

    items = []

    for i, playlist in enumerate(dataset.reader('playlists_%s.csv' % filename, 'items_%s.csv' % filename)):

        items.extend(playlist['items'])
        items.append(0)  # <eos>
        print('read playlist %s: %d' % (filename, i))

    return items


def ptb_raw_data(dataset):

    print('reading training data')
    train_data = _read_items("training", dataset)

    print('reading validation data')
    valid_data = _read_items("validation", dataset)

    return train_data, valid_data


def ptb_iterator(raw_data, batch_size, num_steps):

    data_len = len(raw_data)
    batch_len = data_len // batch_size

    data = np.zeros([batch_size, batch_len], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]

        yield (x, y)
