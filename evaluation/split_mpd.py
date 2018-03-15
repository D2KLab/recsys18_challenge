import argparse
import csv
import json
import random
from os import listdir
from os.path import join

random.seed(1)

parser = argparse.ArgumentParser(description="Split MPD")

parser.add_argument('--mpd_path', default=None, required=True)
parser.add_argument('--out_path', default=None, required=True)
parser.add_argument('--scale', type=int, required=True)

args = parser.parse_args()

mpd_path = args.mpd_path
out_path = args.out_path
scale = args.scale
validation_size = 10 * scale

tracks_count = {}
pid_slices = {}
slices = []

print("Counting the number of entries per track")
with open(join(out_path, 'tracks_mpd.csv'), 'w', newline='', encoding='utf8') as tracks_file:
    writer = csv.writer(tracks_file)
    for mpd_slice in listdir(mpd_path):
        with open(join(mpd_path, mpd_slice), encoding='utf8') as json_file:
            print("\tReading file " + mpd_slice)
            json_slice = json.load(json_file)
            slices.append(json_slice)

            for playlist in json_slice['playlists']:
                pid = playlist['pid']
                pid_slices[pid] = json_slice

                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    if track_uri in tracks_count:
                        tracks_count[track_uri] += 1
                    else:
                        # This is a new track
                        tracks_count[track_uri] = 1
                        writer.writerow([track['track_uri'], track['track_name'],
                                         track['artist_uri'], track['artist_name'],
                                         track['album_uri'], track['album_name'],
                                         track['duration_ms']])

# Saving the count of the entries
with open(join(out_path, 'tracks_mpd_count.csv'), 'w', newline='', encoding='utf8') as count_file:
    writer = csv.writer(count_file)
    for track_uri, count in tracks_count.items():
        writer.writerow([track_uri, count])

print("Selecting validation playlists randomly")
validation_playlists = []
candidate_pid_list = list(pid_slices)
random.shuffle(candidate_pid_list)

for candidate_pid in candidate_pid_list:
    candidate_playlist = None
    candidate_count = tracks_count.copy()

    # Check that pid is not already in the validation set
    if candidate_pid in validation_playlists:
        continue

    # Load the candidate playlist
    json_slice = pid_slices[candidate_pid]

    for playlist in json_slice['playlists']:
        if playlist['pid'] == candidate_pid:
            candidate_playlist = playlist
            break

    # Innocent until proven guilty
    good_candidate = True

    # Check that pid does not contain unique tracks
    for track in candidate_playlist['tracks']:
        track_uri = track['track_uri']

        if candidate_count[track_uri] > 1:
            candidate_count[track_uri] -= 1
        else:
            good_candidate = False
            break

    # Challenge category
    validation_index = len(validation_playlists)

    # Check the length of the playlist
    if validation_index < 1 * scale:
        # Only title
        if len(candidate_playlist['tracks']) < 1:
            good_candidate = False
    elif validation_index < 2 * scale:
        # Title and first one
        if len(candidate_playlist['tracks']) <= 1:
            good_candidate = False
    elif validation_index < 3 * scale:
        # Title and first five
        if len(candidate_playlist['tracks']) <= 5:
            good_candidate = False
    elif validation_index < 4 * scale:
        # No title and first five
        if len(candidate_playlist['tracks']) <= 5:
            good_candidate = False
    elif validation_index < 5 * scale:
        # Title and first ten
        if len(candidate_playlist['tracks']) <= 10:
            good_candidate = False
    elif validation_index < 6 * scale:
        # No title and first ten
        if len(candidate_playlist['tracks']) <= 10:
            good_candidate = False
    elif validation_index < 7 * scale:
        # Title and first twenty-five
        if len(candidate_playlist['tracks']) <= 25:
            good_candidate = False
    elif validation_index < 8 * scale:
        # Title and random twenty-five
        if len(candidate_playlist['tracks']) <= 25:
            good_candidate = False
    elif validation_index < 9 * scale:
        # Title and first a hundred
        if len(candidate_playlist['tracks']) <= 100:
            good_candidate = False
    else:
        # Title and random a hundred
        if len(candidate_playlist['tracks']) <= 100:
            good_candidate = False

    # Commit the changes
    if good_candidate is True:
        tracks_count = candidate_count
        validation_playlists.append(candidate_pid)
        print("\tValidation set size is", len(validation_playlists))

        # Check if we are done
        if len(validation_playlists) >= validation_size:
            break

# Saving the results
with open(join(out_path, 'pid_validation.csv'), 'w', newline='', encoding='utf8') as pid_validation_file:
    pid_validation_writer = csv.writer(pid_validation_file)
    for pid in validation_playlists:
        pid_validation_writer.writerow([pid])

playlists_training_file = open(join(out_path, 'playlists_training.csv'), 'w', newline='', encoding='utf8')
items_training_file = open(join(out_path, 'items_training.csv'), 'w', newline='', encoding='utf8')
playlists_validation_file = open(join(out_path, 'playlists_validation.csv'), 'w', newline='', encoding='utf8')
items_validation_file = open(join(out_path, 'items_validation.csv'), 'w', newline='', encoding='utf8')
items_validation_hidden_file = open(join(out_path, 'items_validation_hidden.csv'), 'w', newline='', encoding='utf8')

playlists_training_writer = csv.writer(playlists_training_file)
items_training_writer = csv.writer(items_training_file)
playlists_validation_writer = csv.writer(playlists_validation_file)
items_validation_writer = csv.writer(items_validation_file)
items_validation_hidden_writer = csv.writer(items_validation_hidden_file)

for json_slice in slices:
    for playlist in json_slice['playlists']:
        # Training playlist
        if playlist['pid'] not in validation_playlists:
            playlists_training_writer.writerow([playlist['pid'], playlist['name'],
                                                playlist['collaborative'], playlist['num_tracks'],
                                                playlist['num_artists'], playlist['num_albums'],
                                                playlist['num_followers'], playlist['num_edits'],
                                                playlist['modified_at'], playlist['duration_ms']])

            for track in playlist['tracks']:
                items_training_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

        # Validation playlist
        else:
            # Challenge category
            validation_index = validation_playlists.index(playlist['pid'])

            if validation_index < 1 * scale:
                # Only title
                playlist_name = playlist['name']
                tracks_provided = []
                tracks_hidden = playlist['tracks']
            elif validation_index < 2 * scale:
                # Title and first one
                playlist_name = playlist['name']
                tracks_provided = playlist['tracks'][:1]
                tracks_hidden = playlist['tracks'][1:]
            elif validation_index < 3 * scale:
                # Title and first five
                playlist_name = playlist['name']
                tracks_provided = playlist['tracks'][:5]
                tracks_hidden = playlist['tracks'][5:]
            elif validation_index < 4 * scale:
                # No title and first five
                playlist_name = None
                tracks_provided = playlist['tracks'][:5]
                tracks_hidden = playlist['tracks'][5:]
            elif validation_index < 5 * scale:
                # Title and first ten
                playlist_name = playlist['name']
                tracks_provided = playlist['tracks'][:10]
                tracks_hidden = playlist['tracks'][10:]
            elif validation_index < 6 * scale:
                # No title and first ten
                playlist_name = None
                tracks_provided = playlist['tracks'][:10]
                tracks_hidden = playlist['tracks'][10:]
            elif validation_index < 7 * scale:
                # Title and first twenty-five
                playlist_name = playlist['name']
                tracks_provided = playlist['tracks'][:25]
                tracks_hidden = playlist['tracks'][25:]
            elif validation_index < 8 * scale:
                # Title and random twenty-five
                playlist_name = playlist['name']
                random.shuffle(playlist['tracks'])
                tracks_provided = playlist['tracks'][:25]
                tracks_hidden = playlist['tracks'][25:]
            elif validation_index < 9 * scale:
                # Title and first a hundred
                playlist_name = playlist['name']
                tracks_provided = playlist['tracks'][:100]
                tracks_hidden = playlist['tracks'][100:]
            else:
                # Title and random a hundred
                playlist_name = playlist['name']
                random.shuffle(playlist['tracks'])
                tracks_provided = playlist['tracks'][:100]
                tracks_hidden = playlist['tracks'][100:]

            # Sort tracks by position
            tracks_provided = sorted(tracks_provided, key=lambda item: item['pos'])
            tracks_hidden = sorted(tracks_hidden, key=lambda item: item['pos'])

            playlists_validation_writer.writerow([playlist['pid'], playlist_name,
                                                  len(tracks_provided), len(tracks_hidden),
                                                  len(tracks_provided + tracks_hidden)])

            for track in tracks_provided:
                items_validation_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

            for track in tracks_hidden:
                items_validation_hidden_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])
