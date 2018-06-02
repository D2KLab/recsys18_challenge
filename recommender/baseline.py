import os.path
import random

from gensim.models import Word2Vec

from utils import sentence
from ._recommender import AbstractRecommender


class MostPopular(AbstractRecommender):

    def __init__(self, dataset, dry=True):
        super().__init__(dataset, dry)

        # Count the items
        items_count = {}

        for playlist in self.dataset.reader(self.train_playlists, self.train_items):
            for item in playlist['items']:
                if item in items_count:
                    items_count[item] += 1
                else:
                    items_count[item] = 1

        self.items_sorted = sorted(items_count, key=items_count.get, reverse=True)

    def recommend(self, playlist):
        items = []
        count = 0

        for item in self.items_sorted:
            if item not in playlist['items']:
                items.append(item)
                count += 1
            if count >= 500:
                break

        playlist['items'] = items


class Random(AbstractRecommender):

    def __init__(self, dataset, dry=True, weighted=False):
        super().__init__(dataset, dry)

        # Load the items
        self.items_training = []

        for playlist in self.dataset.reader(self.train_playlists, self.train_items):
            for item in playlist['items']:
                self.items_training.append(item)

        if weighted is False:
            self.items_training = list(set(self.items_training))

    def recommend(self, playlist):
        items = []

        while len(items) < 500:
            item = random.choice(self.items_training)
            if item not in playlist['items'] and item not in items:
                items.append(item)

        playlist['items'] = items


class Word2Rec(AbstractRecommender):

    def __init__(self, dataset, dry=True, model_file=None, mode=sentence.Mode.ALBUM, fallback=MostPopular):
        super().__init__(dataset, dry)
        self.mode = mode
        self.fallback = fallback(dataset, dry=dry)

        if os.path.isfile(model_file):
            # Load the model
            model = Word2Vec.load(model_file)
        else:
            # Train the model
            sentences = sentence.Iterator(self.dataset, self.train_playlists, self.train_items, mode)
            model = Word2Vec(sentences, workers=4, min_count=0)

            # Save the model
            if model_file is not None:
                model.save(model_file)

        self.model = model.wv
        del model

    def recommend(self, playlist):
        seeds = list(map(lambda x: str(x), playlist[self.mode]))

        if len(seeds) > 0:
            n = 500 - len(seeds)
            max_num_seed = 100
            predictions_and_seeds = self.model.most_similar(positive=seeds, topn=n + max_num_seed)
            predictions_and_seeds = [p for (p, s) in predictions_and_seeds]
            predictions = [p for p in predictions_and_seeds if p not in seeds][0:500]
            playlist[self.mode] = list(map(lambda x: int(x), predictions))
        else:
            self.fallback.recommend(playlist)
