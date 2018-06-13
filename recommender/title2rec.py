import os
import re
from pyfasttext import FastText

import emot
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans

from utils import sentence
from _recommender import AbstractRecommender
from baseline import Word2Rec, MostPopular


def index(l, f):
    return next((i for i in np.arange(len(l)) if f(l[i])), None)


class Title2Rec(AbstractRecommender):

    def __init__(self, dataset=False, dry=True, w2r_model_file=None, pl_model_file=None, ft_model_file=None,
                 ft_vec_file=None, cluster_file=None, num_clusters=100, fallback=MostPopular, rnn=False):
        super().__init__(dataset, dry=dry)
        print('Import playlists')

        if rnn:
            self.init_light(ft_model_file)
            return

        self.playlists = self.dataset.reader(self.train_playlists, self.train_items)
        self.playlists = np.array(list(filter(lambda p: p['title'] and len(p['items']) > 0, self.playlists)))

        self.fallback = fallback(dataset, dry=dry)

        if os.path.isfile(ft_model_file):
            self.ft_model = FastText(ft_model_file)
        else:
            print('***Full init started***')
            self.num_clusters = num_clusters
            print('- Import w2r models')
            self.w2r_model = self.get_w2r(dataset, dry, w2r_model_file)
            print('- Compute playlists embeddings')
            self.pl_embs = self.compute_pl_embs(pl_model_file)
            print('- Cluster the playlist')
            self.clusters = self.compute_clusters(cluster_file, self.pl_embs, num_clusters)
            print('- Fast_text on the clusters')
            self.ft_model = self.compute_fasttext(ft_model_file)
            print('***Full init end***')

        if not os.path.isfile(ft_vec_file):
            self.title_vecs = [self.get_title_vector_from_playlist(pl) for pl in self.playlists]
            with open(ft_vec_file, 'w') as file_handler:
                file_handler.write('%d %d\n' % (len(self.title_vecs), len(self.title_vecs[0])))
                for idx, vec in enumerate(self.title_vecs):
                    file_handler.write('%d %s\n' % (idx, ' '.join(vec.astype(np.str))))

        self.pl_vec = KeyedVectors.load_word2vec_format(ft_vec_file, binary=False)

        # nltk.download('stopwords')

    def init_light(self, ft_model_file):
        self.ft_model = FastText(ft_model_file)

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
        ft_model_file = ft_model_file.replace('.bin', '')

        documents = []
        for i in np.arange(self.num_clusters):
            involved_pl = np.array(self.playlists)[self.clusters == i]
            documents.append(process_title(pl['title']).strip() for pl in involved_pl)

        doc_file = 'models/documents.txt'
        np.savetxt(doc_file, [' '.join(d) for d in documents], fmt='%s')
        model = FastText()
        model.skipgram(input=doc_file, output=ft_model_file, epoch=100, lr=0.1)
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
        _item_embs = list(map(lambda track_id: self.w2r_model[str(track_id)], playlist['items']))
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

    def get_vector_from_title(self, title):
        return self.ft_model.get_numpy_sentence_vector(process_title(title).strip())

    def get_title_vector_from_playlist(self, playlist):
        return self.get_vector_from_title(playlist['title'])

    def recommend(self, playlist, n=500, n_pl=300):
        if not playlist['title']:
            return self.fallback.recommend(playlist)

        this_vec = self.get_title_vector_from_playlist(playlist)
        seeds = playlist['items']

        # get more popular tracks among the 100 most similar playlists
        most_similar_vec = self.pl_vec.most_similar(positive=[this_vec], topn=n_pl)
        most_similar_pl = self.playlists[[int(v[0]) for v in most_similar_vec]]
        weights = [v[1] for v in most_similar_vec]

        predictions_and_seeds = [pl['items'] for pl in most_similar_pl]
        playlist['items'] = count_and_weights(predictions_and_seeds, seeds, weights)[0:n]


def count_and_weights(list_of_listes, seeds, weights):
    _set = [item for sublist in list_of_listes for item in sublist]  # flatten
    _set = set([p for p in _set if p not in seeds])

    votes = {}
    for item in _set:
        votes[item] = 0
        for p, pl in enumerate(list_of_listes):
            w = weights[p]
            # w = 1
            if item in pl:
                votes[item] += 1 * w

    return sorted(_set, key=lambda x: votes[x], reverse=True)


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

    # #remove stopwords
    # stop_words = stopwords.words('english')
    # ' '.join([w for w in word.split(' ') if w not in stop_words])

    # remove spaces
    # word_no_spaces = word.replace(' ', '')

    # if(len(punctuation)>=1):
    #       print(punctuation)
    return ' '.join(emos + others) + ' ' + word


class WordPlusTitle2Rec(AbstractRecommender):

    def __init__(self, dataset, dry=True, w2r_model_file=None, pl_model_file=None, ft_model_file=None,
                 ft_vec_file=None, cluster_file=None):
        super().__init__(dataset, dry=dry)
        self.word2rec = Word2Rec(dataset, dry=dry, model_file=w2r_model_file, mode=sentence.Mode.ITEM)
        self.title2rec = Title2Rec(dataset, dry=dry, w2r_model_file=w2r_model_file, pl_model_file=pl_model_file,
                                   ft_model_file=ft_model_file, ft_vec_file=ft_vec_file, cluster_file=cluster_file)

    def recommend(self, playlist):
        seeds = len(playlist['items'])

        if seeds <= 1:
            # If we have only the title or just one seed
            self.title2rec.recommend(playlist)
        else:
            self.word2rec.recommend(playlist)
