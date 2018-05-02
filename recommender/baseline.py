import random
import os.path
from gensim.models import Word2Vec


class Recommender:

    def __init__(self, dataset, dry=True):
        """
        :param dataset: An instance of the helper class for reading and writing the dataset.
        :param dry: If true the recommender will use the internal dataset instead of the official one.
        """
        self.dataset = dataset
        if dry is True:
            self.train_playlists = 'playlists_training_validation.csv'
            self.train_items = 'items_training_validation.csv'
            self.test_playlists = 'playlists_test.csv'
            self.test_items = 'items_test_x.csv'
        else:
            self.train_playlists = 'playlists.csv'
            self.train_items = 'items.csv'
            self.test_playlists = 'playlists_challenge.csv'
            self.test_items = 'items_challenge.csv'

    def recommend(self, playlist):
        raise NotImplementedError

    def run(self, submission_path):
        submission_writer = self.dataset.writer(submission_path)

        for playlist in self.dataset.reader(self.test_playlists, self.test_items):
            self.recommend(playlist)
            submission_writer.write(playlist)


class MostPopular(Recommender):

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


class Random(Recommender):

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


class Word2Rec(Recommender):

    class MySentence:

        def __init__(self, dataset, train_playlists, train_items):
            self.dataset = dataset
            self.train_playlists = train_playlists
            self.train_items = train_items

        def __iter__(self):
            for playlist in self.dataset.reader(self.train_playlists, self.train_items):
                # Convert IDs to strings
                sentence = list(map(lambda x: str(x), playlist['items']))

                yield sentence

    def __init__(self, dataset, dry=True, model_file=None, fallback=MostPopular):
        super().__init__(dataset, dry)
        self.fallback = fallback(dataset, dry)

        if os.path.isfile(model_file):
            # Load the model
            model = Word2Vec.load(model_file)
        else:
            # Train the model
            sentences = self.MySentence(self.dataset, self.train_playlists, self.train_items)
            model = Word2Vec(sentences, workers=4, min_count=0)

            # Save the model
            if model_file is not None:
                model.save(model_file)

        self.model = model.wv
        del model

    def recommend(self, playlist):
        seeds = list(map(lambda x: str(x), playlist['items']))

        if len(seeds) > 0:
            n = 500 - len(seeds)
            max_num_seed = 100
            predictions_and_seeds = self.model.most_similar(positive=seeds, topn=n + max_num_seed)
            predictions_and_seeds = [p for (p, s) in predictions_and_seeds]
            predictions = [p for p in predictions_and_seeds if p not in seeds][0:500]
            playlist['items'] = list(map(lambda x: int(x), predictions))
        else:
            self.fallback.recommend(playlist)
