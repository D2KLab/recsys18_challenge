from enum import Enum

from .dataset import Dataset


class Mode(Enum):
    ITEM = 'items'
    ARTIST = 'artists'
    ALBUM = 'albums'


class Iterator:

    def __init__(self, dataset: Dataset, train_playlists: str, train_items: str, mode: Mode):
        self.dataset = dataset
        self.train_playlists = train_playlists
        self.train_items = train_items
        self.mode = mode

    def __iter__(self):
        for playlist in self.dataset.reader(self.train_playlists, self.train_items):
            # Convert IDs to strings
            sentence = list(map(lambda x: str(x), playlist[self.mode.value]))

            yield sentence
