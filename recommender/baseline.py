import random


class MostPopular:

    def __init__(self, dataset):
        self.dataset = dataset

    def run(self, submission_path):
        # Count the items
        items_count = {}

        for playlist in self.dataset.reader('playlists_training_validation.csv', 'items_training_validation.csv'):
            for item in playlist['items']:
                if item in items_count:
                    items_count[item] += 1
                else:
                    items_count[item] = 1

        items_sorted = sorted(items_count, key=items_count.get, reverse=True)

        # Create the submission
        submission_writer = self.dataset.writer(submission_path)

        for playlist in self.dataset.reader('playlists_test.csv', 'items_test_x.csv'):
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


class Random:

    def __init__(self, dataset, weighted=False):
        self.dataset = dataset
        self.weighted = weighted

    def run(self, submission_path):
        # Load the items
        items_training = []

        for playlist in self.dataset.reader('playlists_training_validation.csv', 'items_training_validation.csv'):
            for item in playlist['items']:
                items_training.append(item)

        if self.weighted is False:
            items_training = list(set(items_training))

        # Create the submission
        submission_writer = self.dataset.writer(submission_path)

        for playlist in self.dataset.reader('playlists_test.csv', 'items_test_x.csv'):
            items = []

            while len(items) < 500:
                item = random.choice(items_training)
                if item not in playlist['items'] and item not in items:
                    items.append(item)

            playlist['items'] = items
            submission_writer.write(playlist)
