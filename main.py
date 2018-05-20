import argparse

from utils.dataset import Dataset
from recommender import baseline
from recommender import title2rec
from utils import sentence


def run():
    parser = argparse.ArgumentParser(description='RecSys 2018 Challenge')
    parser.add_argument('recommender', type=str, help='Recommandation algorithm')
    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset path')
    parser.add_argument('--no-dry', dest='dry', action='store_false', default=True, help='Real run')
    parser.add_argument('--w2r', type=str, default='models/w2r.bin', help='Word2Rec model')
    parser.add_argument('--pl', type=str, default='models/pl.bin', help='Playlist embeddings')
    parser.add_argument('--ft', type=str, default='models/ft.bin', help='FastText model')
    parser.add_argument('--ft_vec', type=str, default='models/ft_vec.bin', help='FastText vector')
    parser.add_argument('--cluster', type=str, default='models/cluster.bin', help='Cluster model')
    args = parser.parse_args()

    dataset = Dataset(args.dataset)

    if args.recommender == 'mp':
        baseline.MostPopular(dataset, dry=args.dry).run(args.output)
    elif args.recommender == 'random':
        baseline.Random(dataset, dry=args.dry).run(args.output)
    elif args.recommender == 'random_mp':
        baseline.Random(dataset, dry=args.dry, weighted=True).run(args.output)
    elif args.recommender == 'word2rec_item':
        baseline.Word2Rec(dataset, dry=args.dry, model_file=args.w2r_model, mode=sentence.Mode.ITEM).run(args.output)
    elif args.recommender == 'word2rec_album':
        baseline.Word2Rec(dataset, dry=args.dry, model_file=args.w2r_model, mode=sentence.Mode.ALBUM).run(args.output)
    elif args.recommender == 'word2rec_artist':
        baseline.Word2Rec(dataset, dry=args.dry, model_file=args.w2r_model, mode=sentence.Mode.ARTIST).run(args.output)
    elif args.recommender == 'title2rec':
        title2rec.Title2Rec(dataset, dry=args.dry, w2rmodel_file=args.w2r, pl_model_file=args.pl,
                            ft_model_file=args.ft, ft_vec_file=args.ft_vec,
                            cluster_file=args.cluster).run(args.output)
    else:
        print('Unknown recommender', args.recommender)


if __name__ == "__main__":
    run()
