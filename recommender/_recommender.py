class AbstractRecommender:

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
