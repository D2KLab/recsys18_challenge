import random
import numpy as np
import os.path
from gensim.models import Word2Vec, KeyedVectors
import re
import emot
from pyfasttext import FastText


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


class Title2Rec(Recommender):
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

    def __init__(self, dataset, dry=True, w2rmodel_file=None, pl_model_file=None, ft_model_file=None,
                 ft_vec_file=None, cluster_file=None, num_clusters=100, fallback=MostPopular):
        super().__init__(dataset, dry)
        self.fallback = fallback(dataset, dry)

        if os.path.isfile(ft_model_file):
            self.ft_model = FastText(ft_model_file)
        else:
            self.num_clusters = num_clusters
            self.playlists = self.dataset.reader(self.test_playlists, self.test_items)
            self.w2rmodel = self.get_w2r(self, dataset, dry, w2rmodel_file, fallback)
            self.pl_embs = self.compute_pl_embs(pl_model_file)
            self.clusters = self.compute_clusters(cluster_file, self.pl_embs, num_clusters)
            self.ft_model = self.compute_fasttext(ft_model_file)

        if not os.path.isfile(ft_vec_file):
            self.title_vecs = [self.get_vector_from_title(pl['title']) for pl in self.playlists]
            with open(ft_vec_file, 'w') as file_handler:
                file_handler.write('%d %d\n' % (len(self.title_vecs), len(self.title_vecs[0])))
                for idx, vec in enumerate(self.title_vecs):
                    file_handler.write('%d %s\n' % (idx, ' '.join(vec.astype(np.str))))

        self.pl_vec = KeyedVectors.load_word2vec_format(ft_vec_file, binary=False)

    def get_w2r(self, dataset, dry=True, model_file=None, fallback=MostPopular):
        if os.path.isfile(model_file):
            # Load the model
            model = Word2Vec.load(model_file)
            return model.wv
        else:
            # Train the model
            w2r = Word2Rec(dataset, dry, model_file, fallback)
            return w2r.model

    def compute_fasttext(self, ft_model_file=None):

        documents = [[process_title(pl['title']).strip() for pl in self.playlists[self.clusters == i]]
                     for i in np.arange(self.num_clusters)]
        doc_file = 'documents.txt'
        np.savetxt(doc_file, [' '.join(d) for d in documents], fmt='%s')
        model = FastText()
        model.skipgram(input=doc_file, output=ft_model_file, epoch=100, lr=0.7)
        os.remove(doc_file)

        return model

    def compute_pl_embs(self, pl_model_file):
        if os.path.isfile(pl_model_file):
            # Load the model
            return np.loadtxt(pl_model_file)
        else:
            # Train the model
            _embs = [self.get_vector_from_w2r(playlist) for playlist in self.playlists]
            np.savetxt(pl_model_file, _embs)
            return _embs

    def get_vector_from_w2r(self, playlist):
        _item_embs = map(lambda track_id: self.w2rmodel[str(track_id)], playlist['items'])
        return np.array(_item_embs).mean(axis=0)

    def compute_clusters(self, cluster_file, pl_embs, num_clusters=100):
        if os.path.isfile(cluster_file):
            return np.loadtxt(cluster_file)
        else:
            # from scipy.cluster.hierarchy import linkage, fcluster
            # Z = linkage(pl_embs, 'ward')
            # clusters = fcluster(Z, num_clusters, criterion='maxclust')

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pl_embs)
            clusters = kmeans.predict(pl_embs)
            np.savetxt(cluster_file, clusters, fmt="%u")
            return clusters

    def get_vector_from_title(self, playlist):
        return self.ft_model.get_numpy_sentence_vector(process_title(playlist['title']).strip())

    def recommend(self, playlist, n=500, n_pl=300):
        this_vec = self.get_vector_from_title(playlist)
        seeds = list(map(lambda x: str(x), playlist['items']))

        # get more popular tracks among the 100 most similar playlists
        most_similar_vec = self.pl_vec.most_similar(positive=[this_vec], topn=n_pl)
        most_similar_pl = self.playlists[[v[0] for v in most_similar_vec]]
        predictions_and_seeds = [pl['items'] for pl in most_similar_pl]
        predictions_and_seeds = [item for sublist in predictions_and_seeds for item in sublist] # flatten
        predictions = [p for p in predictions_and_seeds if p not in seeds]
        predictions = sorted(set(predictions), key=predictions.count, reverse=True)[0:n]
        playlist['items'] = list(map(lambda x: int(x), predictions))


def process_title(word=''):
    punctuation_regex = r"[()]?[.,!?;~:]+"
    specials = [r':-?[\)\(]+']

    word = word.lower()

    others = []
    for o in specials:
        m = re.search(o, word)
        if m is not None:
            others.append(m.group(0))
            word = word.replace(m.group(0), '')

    emoji = emot.emoji(word)
    #  detect skin code
    #     to_delete = []
    #     for i, emo in enumerate(emoji):
    #         if u"\U0001F3FB" <= emo['value'] <= u"\U0001F3FF":
    #             emoji[i-1]['value'] += emo['value']
    #             to_delete.append(i)
    #     for i in reversed(to_delete):
    #         del emoji[i]

    if not re.search('[a-zA-Z]', word):
        #  just manage emoji
        emos = list(map(lambda emo: emo['value'], emoji))
        for emo in emos:
            word = word.replace(emo, '')
        return ' '.join(emos) + ' ' + word

    emoticons = emot.emoticons(word)

    # skip all-words emoticons, normally wrong
    emoticons = list(filter(lambda emo: re.search('[^a-z]', emo['value']), emoticons))

    # merge single-char emoticons, normally wrong
    previous = -2
    to_delete = []
    for i, emo in enumerate(emoticons):
        if len(emo['value']) < 2 and emo['location'][0] == previous - 1:
            emoticons[i - 1]['value'] += emo['value']
            to_delete.append(i)
        previous = emo['location'][1]
    for i in reversed(to_delete):
        del emoticons[i]
        # remove remaining single-char emoticons
    emoticons = list(filter(lambda emo: len(emo['value']) >= 2, emoticons))

    emos = list(map(lambda emo: emo['value'], emoji + emoticons))
    for emo in emos:
        word = word.replace(emo, '')

    # punctuation
    #     punctuation = re.findall(punctuation_regex, word)
    #     for p in punctuation:
    #         word = word.replace(p, ' ')

    #     # parentesis
    #     word = re.sub(r'[\(\)]', '', word)

    # multiple spaces
    word = word.replace('  ', ' ').strip()

    # separated letters (i.e. 'w o r k o u t' or 'r & b')
    if re.match(r'^([\w&] )+[\w&]$', word):
        word = word.replace(' ', '')

    # hashtag
    word = re.sub(r'^#', '', word)

    #     if(len(punctuation)>=1):
    #         print(punctuation)
    return ' '.join(emos + others) + ' ' + word
