import random
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

    def run(self, submission_path):
        raise NotImplementedError


class MostPopular(Recommender):

    def __init__(self, dataset, dry=True):
        super().__init__(dataset, dry)

    def run(self, submission_path):
        # Count the items
        items_count = {}

        for playlist in self.dataset.reader(self.train_playlists, self.train_items):
            for item in playlist['items']:
                if item in items_count:
                    items_count[item] += 1
                else:
                    items_count[item] = 1

        items_sorted = sorted(items_count, key=items_count.get, reverse=True)

        # Create the submission
        submission_writer = self.dataset.writer(submission_path)

        for playlist in self.dataset.reader(self.test_playlists, self.test_items):
            items = []
            count = 0

            for item in items_sorted:
                if item not in playlist['items']:
                    items.append(item)
                    count += 1
                if count >= 500:
                    break

            playlist['items'] = items
            submission_writer.write(playlist)


class Random(Recommender):

    def __init__(self, dataset, dry=True, weighted=False):
        super().__init__(dataset, dry)
        self.weighted = weighted

    def run(self, submission_path):
        # Load the items
        items_training = []

        for playlist in self.dataset.reader(self.train_playlists, self.train_items):
            for item in playlist['items']:
                items_training.append(item)

        if self.weighted is False:
            items_training = list(set(items_training))

        # Create the submission
        submission_writer = self.dataset.writer(submission_path)

        for playlist in self.dataset.reader(self.test_playlists, self.test_items):
            items = []

            while len(items) < 500:
                item = random.choice(items_training)
                if item not in playlist['items'] and item not in items:
                    items.append(item)

            playlist['items'] = items
            submission_writer.write(playlist)


# TODO
class Word2Rec:

    def __init__(self, dataset, model_file):

        self.dataset = dataset

        model = Word2Vec.load(model_file)

        self.model = model.wv

        del model

    def run(self, submission_path):

        # Create the submission

        submission_writer = self.dataset.writer(submission_path)

        for playlist in self.dataset.reader('playlists_test.csv', 'items_test_x.csv'):

            seeds = list(map(lambda x: str(x), playlist['items']))

            length_seeds = len(seeds)

            n = 500 - length_seeds

            max_num_seed = 100

            predictions_and_seeds = self.model.most_similar(positive=seeds, topn=n+max_num_seed)

            predictions_and_seeds = [p for (p,s) in predictions_and_seeds]

            predictions = [p for p in predictions_and_seeds if p not in seeds][0:500]

            playlist['items'] = list(map(lambda x: int(x), predictions)) 

            submission_writer.write(playlist)
