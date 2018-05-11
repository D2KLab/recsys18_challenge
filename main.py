from utils.dataset import Dataset
from recommender import baseline

dataset = Dataset('dataset')

most_popular = baseline.MostPopular(dataset, dry=True)
most_popular.run('most_popular_dry.csv')

random = baseline.Random(dataset, dry=True)
random.run('random_dry.csv')

unigram = baseline.Random(dataset, dry=True, weighted=True)
unigram.run('unigram_dry.csv')

word2rec = baseline.Word2Rec(dataset, dry=True, model_file='models/word2rec_dry.w2v')
word2rec.run('word2rec_dry.csv')
