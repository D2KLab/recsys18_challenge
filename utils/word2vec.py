from gensim.models import Word2Vec
import argparse
import logging
import dataset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser(description="Run word2vec")

parser.add_argument('--file', default=None, required=True)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--test', default=False, action='store_true')

args = parser.parse_args()

class MySentences(object):

    def __init__(self, file, dataset, sep=','):

        self.file = file
        self.sep = sep
        self.dataset = dataset
    
    # streams through the items_training.csv and yield sequences of items

    def __iter__(self):

        for playlist in self.dataset.reader('playlists_%s.csv' %self.file, 'items_%s.csv' %self.file):

            #  convert ids to strings
            sentence = list(map(lambda x : str(x), playlist['items']))

            yield sentence

if not args.test:  # train the model
    
    dataset = dataset.Dataset('../dataset')

    sentences = MySentences(args.file, dataset)

    model = Word2Vec(sentences, workers=args.workers, min_count=0)

    file_name = args.file.split('/')[-1]
    file_name = file_name.split('.')[0]

    model.save('../models/'+file_name+'.w2v')

else:  # test the model

    model = Word2Vec.load(args.file)

    word_vectors = model.wv

    del model  # save memory

    key = 'spotify:track:2PpruBYCo4H7WOBJ7Q2EwM'  # OutKast, Hey Ya

    print(word_vectors.most_similar(key))

    # Results:
    # OutKast, Ms. Jackson
    # OutKast, The Way You Move - Club Mix
    # OutKast, Roses
    # R. Kelly, Ignition - Remix
    # OutKast, So Fresh, So Clean
    # Usher, Yeah!
    # Beyoncé, Crazy In Love
    # Shakira, Hips Don't Lie
    # Gnarls Barkley, Crazy
    # Kanye West, Gold Digger
