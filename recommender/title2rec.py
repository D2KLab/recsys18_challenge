import os
import re
from pyfasttext import FastText

import emot
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans

from utils import sentence
from ._recommender import AbstractRecommender
from .baseline import Word2Rec, MostPopular


def index(l, f):
    return next((i for i in np.arange(len(l)) if f(l[i])), None)


class Title2Rec(AbstractRecommender):

    def __init__(self, dataset, dry=True, w2rmodel_file=None, pl_model_file=None, ft_model_file=None,
                 ft_vec_file=None, cluster_file=None, num_clusters=100, fallback=MostPopular):
        super().__init__(dataset, dry=dry)
        self.playlists = self.dataset.reader(self.train_playlists, self.train_items)
        self.playlists = np.array(list(filter(lambda p: p['title'] and len(p['items']) > 0, self.playlists)))

        self.fallback = fallback(dataset, dry=dry)

        if os.path.isfile(ft_model_file):
            self.ft_model = FastText(ft_model_file)
        else:
            self.num_clusters = num_clusters

            self.w2rmodel = self.get_w2r(dataset, dry, w2rmodel_file)
            self.pl_embs = self.compute_pl_embs(pl_model_file)
            self.clusters = self.compute_clusters(cluster_file, self.pl_embs, num_clusters)
            self.ft_model = self.compute_fasttext(ft_model_file)

        if not os.path.isfile(ft_vec_file):
            self.title_vecs = [self.get_vector_from_title(pl) for pl in self.playlists]
            with open(ft_vec_file, 'w') as file_handler:
                file_handler.write('%d %d\n' % (len(self.title_vecs), len(self.title_vecs[0])))
                for idx, vec in enumerate(self.title_vecs):
                    file_handler.write('%d %s\n' % (idx, ' '.join(vec.astype(np.str))))

        self.pl_vec = KeyedVectors.load_word2vec_format(ft_vec_file, binary=False)

    def get_w2r(self, dataset, dry, model_file):
        if os.path.isfile(model_file):
            # Load the model
            model = Word2Vec.load(model_file)
            return model.wv
        else:
            # Train the model
            w2r = Word2Rec(dataset, dry=dry, model_file=model_file, mode=sentence.Mode.ITEM)
            return w2r.model

    def compute_fasttext(self, ft_model_file):
        documents = [[process_title(pl['title']).strip() for pl in np.array(self.playlists)[self.clusters == i]]
                     for i in np.arange(self.num_clusters)]
        doc_file = 'models/documents.txt'
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
        _item_embs = list(map(lambda track_id: self.w2rmodel[str(track_id)], playlist['items']))
        return np.array(_item_embs).mean(axis=0)

    def compute_clusters(self, cluster_file, pl_embs, num_clusters=100):
        if os.path.isfile(cluster_file):
            return np.loadtxt(cluster_file)
        else:
            # from scipy.cluster.hierarchy import linkage, fcluster
            # Z = linkage(pl_embs, 'ward')
            # clusters = fcluster(Z, num_clusters, criterion='maxclust')

            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pl_embs)
            clusters = kmeans.predict(pl_embs)
            np.savetxt(cluster_file, clusters, fmt="%u")
            return clusters

    def get_vector_from_title(self, playlist):
        return self.ft_model.get_numpy_sentence_vector(process_title(playlist['title']).strip())

    def recommend(self, playlist, n=500, n_pl=300):
        if not playlist['title']:
            return self.fallback.recommend(playlist)

        this_vec = self.get_vector_from_title(playlist)
        seeds = list(map(lambda x: str(x), playlist['items']))

        # get more popular tracks among the 100 most similar playlists
        most_similar_vec = self.pl_vec.most_similar(positive=[this_vec], topn=n_pl)
        most_similar_pl = self.playlists[[int(v[0]) for v in most_similar_vec]]
        predictions_and_seeds = [pl['items'] for pl in most_similar_pl]
        predictions_and_seeds = [item for sublist in predictions_and_seeds for item in sublist]  # flatten
        predictions = [p for p in predictions_and_seeds if p not in seeds]
        predictions = sorted(set(predictions), key=predictions.count, reverse=True)[0:n]
        playlist['items'] = list(map(lambda x: int(x), predictions))


def process_title(word=''):
    # punctuation_regex = r"[()]?[.,!?;~:]+"
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
