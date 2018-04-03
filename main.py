from utils.dataset import Dataset
from recommender import baseline

dataset = Dataset('dataset')

most_popular = baseline.MostPopular(dataset)
most_popular.run('most_popular.csv')

random = baseline.Random(dataset)
random.run('random.csv')

unigram = baseline.Random(dataset, weighted=True)
unigram.run('unigram.csv')
