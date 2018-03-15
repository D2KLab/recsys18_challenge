import argparse
import csv
import json

parser = argparse.ArgumentParser(description="Convert challenge")

parser.add_argument('--json', default=None, required=True)
args = parser.parse_args()

playlists_file = open('playlists_challenge.csv', 'w', newline='', encoding='utf8')
items_file = open('items_challenge.csv', 'w', newline='', encoding='utf8')

playlists_writer = csv.writer(playlists_file)
items_writer = csv.writer(items_file)

with open(args.json, encoding='utf8') as json_file:
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
