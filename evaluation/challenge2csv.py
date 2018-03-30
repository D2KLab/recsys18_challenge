import argparse
import csv
import json
from os import path

parser = argparse.ArgumentParser(description="Convert challenge")

parser.add_argument('--challenge_path', default=None, required=True)
parser.add_argument('--out_path', default=None, required=True)

args = parser.parse_args()

playlists_file = open(path.join(args.out_path, 'playlists_challenge.csv'), 'w', newline='', encoding='utf8')
items_file = open(path.join(args.out_path, 'items_challenge.csv'), 'w', newline='', encoding='utf8')

playlists_writer = csv.writer(playlists_file)
items_writer = csv.writer(items_file)

with open(args.challenge_path, encoding='utf8') as json_file:
    json_slice = json.load(json_file)

    for playlist in json_slice['playlists']:
        try:
            playlist_name = playlist['name']
        except KeyError:
            playlist_name = None

        playlists_writer.writerow([playlist['pid'], playlist_name,
                                   playlist['num_samples'], playlist['num_holdouts'],
                                   playlist['num_tracks']])

        for track in playlist['tracks']:
            items_writer.writerow([playlist['pid'], track['pos'], track['track_uri']])
