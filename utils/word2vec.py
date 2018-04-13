from gensim.models import Word2Vec
import argparse
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser(description="Run word2vec")

parser.add_argument('--file', default=None, required=True)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--test', default=False, action='store_true')

args = parser.parse_args()

class MySentences(object):

    def __init__(self, file, sep=','):

        self.file = file
        self.sep = sep
    
    # streams through the items_training.csv and yield playlists

    def __iter__(self):

        playlist = []

        with open(self.file, "r") as f:

            for i, line in enumerate(f):

                line_split = line.replace('\n', '').split(',')

                curr_id = line_split[0]

                # only for the first line
                if i == 0:

                    prev_id = curr_id

                # check if there is a new playlist
                if curr_id != prev_id:

                    yield playlist

                    playlist = []

                playlist.append(line_split[-1])

                prev_id = curr_id

            # last playlist
            yield playlist


if not args.test:  # train the model

    sentences = MySentences(args.file)

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
