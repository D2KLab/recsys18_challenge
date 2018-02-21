from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser(description="Run word2vec")

parser.add_argument('--file')
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()


# memory-friendly iterator streaming through the text file

class MySentences(object):

    def __init__(self, file, sep=';'):

        self.file = file
        self.sep = sep
 
    def __iter__(self):
        
        for line in open(self.file):

            line = line.strip('\n').split(self.sep)

            yield line

sentences = MySentences(args.file)

model = Word2Vec(sentences, workers=args.workers, min_count=0)

file_name = args.file.split('/')[1]
file_name = file_name.split('.')[0]

model.save('models/'+file_name+'.wv')

