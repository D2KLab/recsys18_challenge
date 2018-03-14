from os import listdir
from os.path import join

import json
import random
import csv

validation_size = 10
mpd_path = 'data'

tracks_count = {}
pid_files = {}

# Count the number of tracks
with open('tracks_mpd.csv', 'w', newline='', encoding='utf8') as tracks_file:
    writer = csv.writer(tracks_file)
    for mpd_slice in listdir(mpd_path):
        with open(join(mpd_path, mpd_slice), encoding='utf8') as json_file:
            json_slice = json.load(json_file)

            for playlist in json_slice['playlists']:
                pid = playlist['pid']
                pid_files[pid] = join(mpd_path, mpd_slice)

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

#  Save the number of tracks
with open('tracks_mpd_count.csv', 'w', newline='', encoding='utf8') as count_file:
    writer = csv.writer(count_file)
    for track_uri, count in tracks_count.items():
        writer.writerow([track_uri, count])

# Select validation playlists randomly
validation_playlists = []
candidate_pid_list = list(pid_files)
random.shuffle(candidate_pid_list)

for candidate_pid in candidate_pid_list:
    candidate_playlist = None
    candidate_count = tracks_count.copy()

    # Check that pid is not already in the validation set
    if candidate_pid in validation_playlists:
        continue

    # Load the candidate playlist
    with open(pid_files[candidate_pid], encoding='utf8') as json_file:
        json_slice = json.load(json_file)

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
    if validation_index < 1:
        # Only title
        if len(candidate_playlist['tracks']) < 1:
            good_candidate = False
    elif validation_index < 2:
        # Title and first one
        if len(candidate_playlist['tracks']) <= 1:
            good_candidate = False
    elif validation_index < 3:
        # Title and first five
        if len(candidate_playlist['tracks']) <= 5:
            good_candidate = False
    elif validation_index < 4:
        # No title and first five
        if len(candidate_playlist['tracks']) <= 5:
            good_candidate = False
    elif validation_index < 5:
        # Title and first ten
        if len(candidate_playlist['tracks']) <= 10:
            good_candidate = False
    elif validation_index < 6:
        # No title and first ten
        if len(candidate_playlist['tracks']) <= 10:
            good_candidate = False
    elif validation_index < 7:
        # Title and first twenty-five
        if len(candidate_playlist['tracks']) <= 25:
            good_candidate = False
    elif validation_index < 8:
        # Title and random twenty-five
        if len(candidate_playlist['tracks']) <= 25:
            good_candidate = False
    elif validation_index < 9:
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

        # Check if we are done
        if len(validation_playlists) >= validation_size:
            break

print("Validation playlists:", len(validation_playlists))

playlist_training_file = open('playlists_training.csv', 'w', newline='', encoding='utf8')
tracks_training_file = open('tracks_training.csv', 'w', newline='', encoding='utf8')
playlists_validation_file = open('playlists_validation.csv', 'w', newline='', encoding='utf8')
tracks_validation_file = open('tracks_validation.csv', 'w', newline='', encoding='utf8')
tracks_validation_hidden_file = open('tracks_validation_hidden.csv', 'w', newline='', encoding='utf8')

playlist_training_writer = csv.writer(playlist_training_file)
tracks_training_writer = csv.writer(tracks_training_file)
playlists_validation_writer = csv.writer(playlists_validation_file)
tracks_validation_writer = csv.writer(tracks_validation_file)
tracks_validation_hidden_writer = csv.writer(tracks_validation_hidden_file)

for mpd_slice in listdir(mpd_path):
    with open(join(mpd_path, mpd_slice), encoding='utf8') as json_file:
        json_slice = json.load(json_file)

        for playlist in json_slice['playlists']:
            # Training playlist
            if playlist['pid'] not in validation_playlists:
                playlist_training_writer.writerow([playlist['pid'], playlist['name'],
                                                   playlist['collaborative'], playlist['num_tracks'],
                                                   playlist['num_artists'], playlist['num_albums'],
                                                   playlist['num_followers'], playlist['num_edits'],
                                                   playlist['modified_at'], playlist['duration_ms']])

                for track in playlist['tracks']:
                    tracks_training_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

            # Validation playlist
            else:
                # Challenge category
                validation_index = validation_playlists.index(playlist['pid'])

                if validation_index < 1:
                    # Only title
                    playlist_name = playlist['name']
                    tracks_provided = []
                    tracks_hidden = playlist['tracks']
                elif validation_index < 2:
                    # Title and first one
                    playlist_name = playlist['name']
                    tracks_provided = playlist['tracks'][:1]
                    tracks_hidden = playlist['tracks'][1:]
                elif validation_index < 3:
                    # Title and first five
                    playlist_name = playlist['name']
                    tracks_provided = playlist['tracks'][:5]
                    tracks_hidden = playlist['tracks'][5:]
                elif validation_index < 4:
                    # No title and first five
                    playlist_name = None
                    tracks_provided = playlist['tracks'][:5]
                    tracks_hidden = playlist['tracks'][5:]
                elif validation_index < 5:
                    # Title and first ten
                    playlist_name = playlist['name']
                    tracks_provided = playlist['tracks'][:10]
                    tracks_hidden = playlist['tracks'][10:]
                elif validation_index < 6:
                    # No title and first ten
                    playlist_name = None
                    tracks_provided = playlist['tracks'][:10]
                    tracks_hidden = playlist['tracks'][10:]
                elif validation_index < 7:
                    # Title and first twenty-five
                    playlist_name = playlist['name']
                    tracks_provided = playlist['tracks'][:25]
                    tracks_hidden = playlist['tracks'][25:]
                elif validation_index < 8:
                    # Title and random twenty-five
                    playlist_name = playlist['name']
                    random.shuffle(playlist['tracks'])
                    tracks_provided = playlist['tracks'][:25]
                    tracks_hidden = playlist['tracks'][25:]
                elif validation_index < 9:
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

                playlists_validation_writer.writerow([playlist['pid'], playlist_name,
                                                      len(tracks_provided), len(tracks_hidden),
                                                      len(tracks_provided + tracks_hidden)])

                for track in tracks_provided:
                    tracks_validation_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])

                for track in tracks_hidden:
                    tracks_validation_hidden_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])
