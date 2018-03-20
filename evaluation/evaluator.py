import argparse
import csv
import math
import numpy as np


def print_evaluation(metric):
    print('\t', 'Only title', metric[0:1000].mean())
    print('\t', 'Title and first one', metric[1000:2000].mean())
    print('\t', 'Title and first five', metric[2000:3000].mean())
    print('\t', 'No title and first five', metric[3000:4000].mean())
    print('\t', 'Title and first ten', metric[4000:5000].mean())
    print('\t', 'No title and first ten', metric[5000:6000].mean())
    print('\t', 'Title and first twenty-five', metric[6000:7000].mean())
    print('\t', 'Title and random twenty-five', metric[7000:8000].mean())
    print('\t', 'Title and first a hundred', metric[8000:9000].mean())
    print('\t', 'Title and random a hundred', metric[9000:10000].mean())


def track2artist(tracks_local):
    artists = []
    for track in tracks_local:
        artists.append(tracks_mpd[track])
    return artists


parser = argparse.ArgumentParser(description="Evaluator")

parser.add_argument('--tracks_mpd', default='dataset/tracks_mpd.csv', required=False)
parser.add_argument('--pid_validation', default='dataset/pid_validation.csv', required=False)
parser.add_argument('--items_validation', default='dataset/items_validation.csv', required=False)
parser.add_argument('--items_validation_hidden', default='dataset/items_validation_hidden.csv', required=False)
parser.add_argument('--items_submission', default=None, required=True)

args = parser.parse_args()

print('Evaluating the submission file ' + args.items_submission)

tracks_mpd_file = open(args.tracks_mpd, 'r', newline='', encoding='utf8')
pid_validation_file = open(args.pid_validation, 'r', newline='', encoding='utf8')
items_validation_file = open(args.items_validation, 'r', newline='', encoding='utf8')
items_validation_hidden_file = open(args.items_validation_hidden, 'r', newline='', encoding='utf8')
items_submission_file = open(args.items_submission, 'r', newline='', encoding='utf8')

tracks_mpd_reader = csv.reader(tracks_mpd_file)
pid_validation_reader = csv.reader(pid_validation_file)
items_validation_reader = csv.reader(items_validation_file)
items_validation_hidden_reader = csv.reader(items_validation_hidden_file)
items_submission_reader = csv.reader(items_submission_file)

# Read tracks
tracks_mpd = {}

for row in tracks_mpd_reader:
    track_uri = row[0]
    artist_uri = row[2]
    tracks_mpd[track_uri] = artist_uri

# Read PIDs
pids = []

for row in pid_validation_reader:
    pids.append(row[0])

# Read validation items
items_validation = {}

for row in items_validation_reader:
    pid = row[0]
    if pid in items_validation:
        items_validation[pid].append(row[2])
    else:
        items_validation[pid] = [row[2]]

# Read the submission
submission = {}

for row in items_submission_reader:
    # Skip empty lines
    if len(row) == 0:
        continue

    # Skip team info
    if row[0].startswith('team_info'):
        continue

    # Skip comments
    if row[0].startswith('#'):
        continue

    # Check the length of a row
    if len(row) != 501:
        print('The playlist ' + row[0] + ' has ' + str(len(row) - 1) + ' tracks instead of 500.')
        exit(1)

    pid = row[0]
    tracks = row[1:]

    # Check that the playlist is a required one
    if pid not in pids:
        print('The playlist ' + pid + ' is not in the validation set.')
        exit(1)

    # Check that the playlist was not already provided
    if pid in submission:
        print('The playlist ' + pid + ' is duplicated.')
        exit(1)

    # Check that the tracks are not duplicated
    if len(set(tracks)) != 500:
        print('The playlist ' + pid + ' contains duplicated tracks.')
        exit(1)

    # Check that the tracks are valid
    for track_uri in tracks:
        if track_uri not in tracks_mpd:
            print('The track uri ' + track_uri + ' from playlist ' + pid + ' is invalid.')
            exit(1)

    # Check that the tracks are not in the validation set
    for track_uri in tracks:
        try:
            if track_uri in items_validation[pid]:
                print('The track uri ' + track_uri + ' from playlist ' + pid + ' is in the validation set.')
                exit(1)
        except KeyError:
            # For playlists with no items
            continue

    submission[pid] = tracks

# Check that all the playlists are available
for pid in pids:
    if pid not in submission:
        print('The playlist ' + pid + ' is not included in the submission.')
        exit(1)

# Read hidden items
items_hidden = {}

for row in items_validation_hidden_reader:
    pid = row[0]
    if pid in items_hidden:
        items_hidden[pid].append(row[2])
    else:
        items_hidden[pid] = [row[2]]

# Precision per track
precision_tracks = np.full(len(pids), 0.0)

for i, pid in enumerate(pids):
    g_tracks = items_hidden[pid]
    g_tracks_size = len(g_tracks)
    r_tracks = submission[pid][:g_tracks_size]

    for track_uri in r_tracks:
        if track_uri in g_tracks:
            precision_tracks[i] += 1

    precision_tracks[i] /= g_tracks_size

print('\nPrecision per track', precision_tracks.mean())
print_evaluation(precision_tracks)

# Precision per artist
precision_artists = np.full(len(pids), 0.0)

for i, pid in enumerate(pids):
    g_artists = track2artist(items_hidden[pid])
    g_artists_size = len(g_artists)
    r_artists = track2artist(submission[pid][:g_artists_size])

    for artist_uri in r_artists:
        if artist_uri in g_artists:
            precision_artists[i] += 1

    precision_artists[i] /= g_artists_size

print('\nPrecision per artist', precision_artists.mean())
print_evaluation(precision_artists)

# NDCG per track
ndcg_tracks = np.full(len(pids), 0.0)

for i, pid in enumerate(pids):
    g_tracks = items_hidden[pid]
    r_tracks = submission[pid]

    # DCG
    for index, track_uri in enumerate(r_tracks):
        if track_uri in g_tracks:
            if index != 0:
                ndcg_tracks[i] += 1 / math.log2(index + 1)
            else:
                ndcg_tracks[i] += 1

    # IDCG
    idcg = 1.0
    for track_uri in r_tracks:
        index = 2
        if track_uri in g_tracks:
            idcg += 1 / math.log2(index)
            index += 1

    # NDCG
    ndcg_tracks[i] /= idcg

print('\nNDCG per track', ndcg_tracks.mean())
print_evaluation(ndcg_tracks)

# NDCG per artist
ndcg_artists = np.full(len(pids), 0.0)

for i, pid in enumerate(pids):
    g_artists = track2artist(items_hidden[pid])
    r_artists = track2artist(submission[pid])

    # DCG
    for index, artist_uri in enumerate(r_artists):
        if artist_uri in g_artists:
            if index != 0:
                ndcg_artists[i] += 1 / math.log2(index + 1)
            else:
                ndcg_artists[i] += 1

    # IDCG
    idcg = 1.0
    for artist_uri in r_artists:
        index = 2
        if artist_uri in g_artists:
            idcg += 1 / math.log2(index)
            index += 1

    # NDCG
    ndcg_artists[i] /= idcg

print('\nNDCG per artist', ndcg_artists.mean())
print_evaluation(ndcg_artists)
