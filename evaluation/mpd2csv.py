import argparse
import csv
import json
from os import listdir
from os.path import join

parser = argparse.ArgumentParser(description="Convert MPD")

parser.add_argument('--path', default=None, required=True)
args = parser.parse_args()

mpd_path = args.path

playlists_file = open('playlists_mpd.csv', 'w', newline='', encoding='utf8')
items_file = open('items_mpd.csv', 'w', newline='', encoding='utf8')

playlists_writer = csv.writer(playlists_file)
items_writer = csv.writer(items_file)

for mpd_slice in listdir(mpd_path):
    with open(join(mpd_path, mpd_slice), encoding='utf8') as json_file:
        print("\tReading file " + mpd_slice)
        json_slice = json.load(json_file)

        for playlist in json_slice['playlists']:
            playlists_writer.writerow([playlist['pid'], playlist['name'],
                                       playlist['collaborative'], playlist['num_tracks'],
                                       playlist['num_artists'], playlist['num_albums'],
                                       playlist['num_followers'], playlist['num_edits'],
                                       playlist['modified_at'], playlist['duration_ms']])

            for track in playlist['tracks']:
                items_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])
