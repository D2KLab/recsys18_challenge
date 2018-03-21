import argparse
import csv
import random

parser = argparse.ArgumentParser(description="Random Recommender")

parser.add_argument('--items_training', default='dataset/items_training.csv', required=False)
parser.add_argument('--playlists_test', default='dataset/playlists_validation.csv', required=False)
parser.add_argument('--items_test', default='dataset/items_validation.csv', required=False)
parser.add_argument('--items_submission', default=None, required=True)
parser.add_argument('--unigram', action='store_true')

args = parser.parse_args()

items_training_file = open(args.items_training, 'r', newline='', encoding='utf8')
playlists_test_file = open(args.playlists_test, 'r', newline='', encoding='utf8')
items_test_file = open(args.items_test, 'r', newline='', encoding='utf8')
items_submission_file = open(args.items_submission, 'w', newline='', encoding='utf8')

items_training_reader = csv.reader(items_training_file)
playlists_test_reader = csv.reader(playlists_test_file)
items_test_reader = csv.reader(items_test_file)
items_submission_writer = csv.writer(items_submission_file)

# Load tracks from the training set
tracks = []

for row in items_training_reader:
    track_uri = row[2]
    tracks.append(track_uri)

if args.unigram is False:
    tracks = list(set(tracks))

# Load the test playlists
playlists_test = []

for row in playlists_test_reader:
    playlists_test.append(row[0])

# Load the test items
items_test = {}

for row in items_test_reader:
    pid = row[0]
    track_uri = row[2]
    if pid in items_test:
        items_test[pid].append(track_uri)
    else:
        items_test[pid] = [track_uri]

# Create the submission
for pid in playlists_test:
    row_submission = [pid]

    while len(row_submission) < 501:
        # Random track_uri
        track_uri = random.choice(tracks)
        # Uniqueness of the submission
        if track_uri in row_submission:
            continue
        # Avoid selecting seed tracks
        try:
            if track_uri in items_test[pid]:
                continue
        except KeyError:
            pass
        row_submission.append(track_uri)

    items_submission_writer.writerow(row_submission)
