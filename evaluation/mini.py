import argparse
import csv
import random
from os import path

random.seed(1)

parser = argparse.ArgumentParser(description="Generate mini dataset.")

parser.add_argument('--path', default='dataset', required=False)
parser.add_argument('--size', type=int, default=100, required=False)
parser.add_argument('--out', default=None, required=True)

args = parser.parse_args()

items = {}
tracks = {}
tracks_ids = set()
playlists = {}
playlists_pid = []

print("Reading the tracks")
with open(path.join(args.path, 'tracks.csv'), 'r', newline='', encoding='utf8') as tracks_file:
    tracks_reader = csv.reader(tracks_file)

    for track in tracks_reader:
        pid = track[0]
        tracks[pid] = track

print("Reading the playlists")
with open(path.join(args.path, 'playlists.csv'), 'r', newline='', encoding='utf8') as playlists_file:
    playlists_reader = csv.reader(playlists_file)

    for playlist in playlists_reader:
        pid = playlist[0]
        playlists[pid] = playlist
        playlists_pid.append(pid)

print("Reading the items")
with open(path.join(args.path, 'items.csv'), 'r', newline='', encoding='utf8') as items_file:
    items_reader = csv.reader(items_file)

    for item in items_reader:
        pid = item[0]
        track_uri = item[2]

        if pid in items:
            items[pid].append(item)
        else:
            items[pid] = [item]

candidate_pid_list = list(playlists_pid)
random.shuffle(candidate_pid_list)

print("Writing the training playlists")
counter = 0

output_playlists_file = open(path.join(args.out, 'playlists_training.csv'), 'w', newline='', encoding='utf8')
output_items_file = open(path.join(args.out, 'items_training.csv'), 'w', newline='', encoding='utf8')

output_playlists_writer = csv.writer(output_playlists_file)
output_items_writer = csv.writer(output_items_file)

while counter < args.size:
    candidate_pid = candidate_pid_list.pop(0)
    counter += 1

    output_playlists_writer.writerow(playlists[candidate_pid])

    for item in items[candidate_pid]:
        output_items_writer.writerow(item)
        tracks_ids.add(item[2])

print("Writing the validation playlists")
validation_size = int(args.size * 0.01)
counter = 0

output_playlists_file = open(path.join(args.out, 'playlists_validation.csv'), 'w', newline='', encoding='utf8')
output_items_file = open(path.join(args.out, 'items_validation.csv'), 'w', newline='', encoding='utf8')

output_playlists_writer = csv.writer(output_playlists_file)
output_items_writer = csv.writer(output_items_file)

while counter < validation_size:
    candidate_pid = candidate_pid_list.pop(0)

    for item in items[candidate_pid]:
        if item[2] not in tracks_ids:
            break
    else:
        counter += 1

        output_playlists_writer.writerow(playlists[candidate_pid])

        for item in items[candidate_pid]:
            output_items_writer.writerow(item)

print("Writing the test playlists")
test_size = int(args.size * 0.01)
counter = 0

output_playlists_file = open(path.join(args.out, 'playlists_test.csv'), 'w', newline='', encoding='utf8')
output_playlists_pid_file = open(path.join(args.out, 'playlists_test_pid.csv'), 'w', newline='', encoding='utf8')
output_items_x_file = open(path.join(args.out, 'items_test_x.csv'), 'w', newline='', encoding='utf8')
output_items_y_file = open(path.join(args.out, 'items_test_y.csv'), 'w', newline='', encoding='utf8')

output_playlists_writer = csv.writer(output_playlists_file)
output_playlists_pid_writer = csv.writer(output_playlists_pid_file)
output_items_x_writer = csv.writer(output_items_x_file)
output_items_y_writer = csv.writer(output_items_y_file)

while counter < validation_size:
    candidate_pid = candidate_pid_list.pop(0)

    for item in items[candidate_pid]:
        if item[2] not in tracks_ids:
            break
    else:
        counter += 1

        output_playlists_writer.writerow(playlists[candidate_pid])
        output_playlists_pid_writer.writerow([candidate_pid])

        for i, item in enumerate(items[candidate_pid]):
            if i < 5:
                output_items_x_writer.writerow(item)
            else:
                output_items_y_writer.writerow(item)

print("Writing the tracks file")
output_tracks_file = open(path.join(args.out, 'tracks.csv'), 'w', newline='', encoding='utf8')

output_tracks_writer = csv.writer(output_tracks_file)

for track_id in tracks_ids:
    output_tracks_writer.writerow(tracks[track_id])
