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
    print('\t', 'Title and random a hundred', metric[9000:10000].mean(), '\n')


def track2artist(tracks_local):
    artists = []
    for track in tracks_local:
        artists.append(tracks[track])
    return artists


parser = argparse.ArgumentParser(description="Evaluator")

parser.add_argument('--tracks', default=None, required=True)
parser.add_argument('--playlists_test_pid', default=None, required=True)
parser.add_argument('--items_test_x', default=None, required=True)
parser.add_argument('--items_test_y', default=None, required=True)
parser.add_argument('--items_submission', default=None, required=True)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

print('Evaluating the submission file ' + args.items_submission + '\n')

tracks_file = open(args.tracks, 'r', newline='', encoding='utf8')
playlists_test_pid_file = open(args.playlists_test_pid, 'r', newline='', encoding='utf8')
items_test_x_file = open(args.items_test_x, 'r', newline='', encoding='utf8')
items_test_y_file = open(args.items_test_y, 'r', newline='', encoding='utf8')
items_submission_file = open(args.items_submission, 'r', newline='', encoding='utf8')

tracks_reader = csv.reader(tracks_file)
playlists_test_pid_reader = csv.reader(playlists_test_pid_file)
items_test_x_reader = csv.reader(items_test_x_file)
items_test_y_reader = csv.reader(items_test_y_file)
items_submission_reader = csv.reader(items_submission_file)

# Read tracks
tracks = {}

for row in tracks_reader:
    track_uri = row[0]
    artist_uri = row[2]
    tracks[track_uri] = artist_uri

# Read PIDs
playlists_pid = []

for row in playlists_test_pid_reader:
    playlists_pid.append(row[0])

# Read test items
items_test = {}

for row in items_test_x_reader:
    pid = row[0]
    if pid in items_test:
        items_test[pid].append(row[2])
    else:
        items_test[pid] = [row[2]]

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
        print('The playlist ' + row[0] + ' has ' + str(len(row) - 1) + ' items instead of 500.')
        exit(1)

    pid = row[0]
    items = row[1:]

    # Check that the playlist is a required one
    if pid not in playlists_pid:
        print('The playlist ' + pid + ' is not in the test set.')
        exit(1)

    # Check that the playlist was not already provided
    if pid in submission:
        print('The playlist ' + pid + ' is duplicated.')
        exit(1)

    # Check that the items are not duplicated
    if len(set(items)) != 500:
        print('The playlist ' + pid + ' contains duplicated items.')
        exit(1)

    # Check that the items are valid
    for track_uri in items:
        if track_uri not in tracks:
            print('The track uri ' + track_uri + ' from playlist ' + pid + ' is invalid.')
            exit(1)

    # Check that the tracks are not in the test set
    for track_uri in items:
        try:
            if track_uri in items_test[pid]:
                print('The track uri ' + track_uri + ' from playlist ' + pid + ' is in the test set.')
                exit(1)
        except KeyError:
            # For playlists with no items
            continue

    submission[pid] = items

# Check that all the playlists are available
for pid in playlists_pid:
    if pid not in submission:
        print('The playlist ' + pid + ' is not included in the submission.')
        exit(1)

# Read y items
items_y = {}

for row in items_test_y_reader:
    pid = row[0]
    if pid in items_y:
        items_y[pid].append(row[2])
    else:
        items_y[pid] = [row[2]]

# Precision per track
precision_tracks = np.full(len(playlists_pid), 0.0)

for i, pid in enumerate(playlists_pid):
    try:
        g_tracks = items_y[pid]
    except KeyError:
        continue
    g_tracks_size = len(g_tracks)
    r_tracks = submission[pid][:g_tracks_size]

    for track_uri in r_tracks:
        if track_uri in g_tracks:
            precision_tracks[i] += 1

    precision_tracks[i] /= g_tracks_size

print('Precision per track', precision_tracks.mean())
if args.verbose is True:
    print_evaluation(precision_tracks)

# Precision per artist
precision_artists = np.full(len(playlists_pid), 0.0)

for i, pid in enumerate(playlists_pid):
    try:
        g_artists = track2artist(items_y[pid])
    except KeyError:
        continue
    g_artists_size = len(g_artists)
    r_artists = track2artist(submission[pid][:g_artists_size])

    for artist_uri in r_artists:
        if artist_uri in g_artists:
            precision_artists[i] += 1

    precision_artists[i] /= g_artists_size

print('Precision per artist', precision_artists.mean())
if args.verbose is True:
    print_evaluation(precision_artists)

# NDCG per track
ndcg_tracks = np.full(len(playlists_pid), 0.0)

for i, pid in enumerate(playlists_pid):
    try:
        g_tracks = items_y[pid]
    except KeyError:
        continue
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

print('NDCG per track', ndcg_tracks.mean())
if args.verbose is True:
    print_evaluation(ndcg_tracks)

# NDCG per artist
ndcg_artists = np.full(len(playlists_pid), 0.0)

for i, pid in enumerate(playlists_pid):
    try:
        g_artists = track2artist(items_y[pid])
    except KeyError:
        continue
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

print('NDCG per artist', ndcg_artists.mean())
if args.verbose is True:
    print_evaluation(ndcg_artists)

# Clicks per track
clicks_tracks = np.full(len(playlists_pid), 51)

for i, pid in enumerate(playlists_pid):
    try:
        g_tracks = items_y[pid]
    except KeyError:
        continue
    r_tracks = submission[pid]

    for index, track_uri in enumerate(r_tracks):
        if track_uri in g_tracks:
            clicks_tracks[i] = math.floor(index / 10)
            break

print('Clicks per track', clicks_tracks.mean())
if args.verbose is True:
    print_evaluation(clicks_tracks)

# Clicks per artist
clicks_artists = np.full(len(playlists_pid), 51)

for i, pid in enumerate(playlists_pid):
    try:
        g_artists = track2artist(items_y[pid])
    except KeyError:
        continue
    r_artists = track2artist(submission[pid])

    for index, artist_uri in enumerate(r_artists):
        if artist_uri in g_artists:
            clicks_artists[i] = math.floor(index / 10)
            break

print('Clicks per artist', clicks_artists.mean())
if args.verbose is True:
    print_evaluation(clicks_artists)
