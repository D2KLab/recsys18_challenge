import argparse
import csv
import random
from os import path

random.seed(1)

parser = argparse.ArgumentParser(description="Split MPD")

parser.add_argument('--path', default=None, required=True)
parser.add_argument('--input_playlists', default=None, required=True)
parser.add_argument('--input_items', default=None, required=True)
parser.add_argument('--output_playlists', default=None, required=True)
parser.add_argument('--output_items', default=None, required=True)
parser.add_argument('--output_playlists_split', default=None, required=True)
parser.add_argument('--output_playlists_split_pid', default=None, required=True)
parser.add_argument('--output_items_split', default=None, required=True)
parser.add_argument('--output_items_split_x', default=None, required=True)
parser.add_argument('--output_items_split_y', default=None, required=True)
parser.add_argument('--scale', type=int, required=True)

args = parser.parse_args()

items = {}
tracks = {}
playlists = {}
playlists_pid = []

print("Reading the playlists")
with open(path.join(args.path, args.input_playlists), 'r', newline='', encoding='utf8') as playlists_file:
    playlists_reader = csv.reader(playlists_file)

    for playlist in playlists_reader:
        pid = playlist[0]
        playlists[pid] = playlist
        playlists_pid.append(pid)

print("Reading the items")
with open(path.join(args.path, args.input_items), 'r', newline='', encoding='utf8') as items_file:
    items_reader = csv.reader(items_file)

    for item in items_reader:
        pid = item[0]
        track_uri = item[2]

        if track_uri in tracks:
            tracks[track_uri] += 1
        else:
            tracks[track_uri] = 1

        if pid in items:
            items[pid].append(item)
        else:
            items[pid] = [item]

print("Selecting split playlists randomly")
split_playlists = []
candidate_pid_list = list(playlists_pid)
random.shuffle(candidate_pid_list)

for candidate_pid in candidate_pid_list:
    candidate_tracks = tracks.copy()

    # Check that pid is not already in the split set
    if candidate_pid in split_playlists:
        continue

    # Load the candidate items
    candidate_items = items[candidate_pid]

    # Innocent until proven guilty
    good_candidate = True

    # Check that pid does not contain unique tracks
    for item in candidate_items:
        track_uri = item[2]
        if candidate_tracks[track_uri] > 1:
            candidate_tracks[track_uri] -= 1
        else:
            good_candidate = False
            break

    # Challenge category
    validation_index = len(split_playlists)

    # Check the length of the playlist
    if validation_index < 1 * args.scale:
        # Only title
        if len(candidate_items) < 1:
            good_candidate = False
    elif validation_index < 2 * args.scale:
        # Title and first one
        if len(candidate_items) <= 1:
            good_candidate = False
    elif validation_index < 3 * args.scale:
        # Title and first five
        if len(candidate_items) <= 5:
            good_candidate = False
    elif validation_index < 4 * args.scale:
        # No title and first five
        if len(candidate_items) <= 5:
            good_candidate = False
    elif validation_index < 5 * args.scale:
        # Title and first ten
        if len(candidate_items) <= 10:
            good_candidate = False
    elif validation_index < 6 * args.scale:
        # No title and first ten
        if len(candidate_items) <= 10:
            good_candidate = False
    elif validation_index < 7 * args.scale:
        # Title and first twenty-five
        if len(candidate_items) <= 25:
            good_candidate = False
    elif validation_index < 8 * args.scale:
        # Title and random twenty-five
        if len(candidate_items) <= 25:
            good_candidate = False
    elif validation_index < 9 * args.scale:
        # Title and first a hundred
        if len(candidate_items) <= 100:
            good_candidate = False
    else:
        # Title and random a hundred
        if len(candidate_items) <= 100:
            good_candidate = False

    # Commit the changes
    if good_candidate is True:
        tracks = candidate_tracks
        split_playlists.append(candidate_pid)
        print("\tSplit set size is", len(split_playlists))

        # Check if we are done
        if len(split_playlists) >= args.scale * 10:
            break

# Saving the results
with open(path.join(args.path, args.output_playlists_split_pid), 'w', newline='', encoding='utf8') as pid_file:
    pid_writer = csv.writer(pid_file)
    for pid in split_playlists:
        pid_writer.writerow([pid])

output_playlists_file = open(path.join(args.path, args.output_playlists), 'w', newline='', encoding='utf8')
output_items_file = open(path.join(args.path, args.output_items), 'w', newline='', encoding='utf8')
output_playlists_split_file = open(path.join(args.path, args.output_playlists_split), 'w', newline='', encoding='utf8')
output_items_split_file = open(path.join(args.path, args.output_items_split), 'w', newline='', encoding='utf8')
output_items_split_x_file = open(path.join(args.path, args.output_items_split_x), 'w', newline='', encoding='utf8')
output_items_split_y_file = open(path.join(args.path, args.output_items_split_y), 'w', newline='', encoding='utf8')

output_playlists_writer = csv.writer(output_playlists_file)
output_items_writer = csv.writer(output_items_file)
output_playlists_split_writer = csv.writer(output_playlists_split_file)
output_items_split_writer = csv.writer(output_items_split_file)
output_items_split_x_writer = csv.writer(output_items_split_x_file)
output_items_split_y_writer = csv.writer(output_items_split_y_file)

for playlist in playlists.values():
    pid = playlist[0]

    # Original playlist
    if pid not in split_playlists:
        output_playlists_writer.writerow(playlist)

        for item in items[pid]:
            output_items_writer.writerow(item)

    # Split playlist
    else:
        # Challenge category
        validation_index = split_playlists.index(pid)

        if validation_index < 1 * args.scale:
            # Only title
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            items_x = []
            items_y = items[pid][:]
        elif validation_index < 2 * args.scale:
            # Title and first one
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            items_x = items[pid][:1]
            items_y = items[pid][1:]
        elif validation_index < 3 * args.scale:
            # Title and first five
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            items_x = items[pid][:5]
            items_y = items[pid][5:]
        elif validation_index < 4 * args.scale:
            # No title and first five
            playlist_name = None
            items_xy = items[pid][:]
            items_x = items[pid][:5]
            items_y = items[pid][5:]
        elif validation_index < 5 * args.scale:
            # Title and first ten
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            items_x = items[pid][:10]
            items_y = items[pid][10:]
        elif validation_index < 6 * args.scale:
            # No title and first ten
            playlist_name = None
            items_xy = items[pid][:]
            items_x = items[pid][:10]
            items_y = items[pid][10:]
        elif validation_index < 7 * args.scale:
            # Title and first twenty-five
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            items_x = items[pid][:25]
            items_y = items[pid][25:]
        elif validation_index < 8 * args.scale:
            # Title and random twenty-five
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            random.shuffle(items[pid])
            items_x = items[pid][:25]
            items_y = items[pid][25:]
        elif validation_index < 9 * args.scale:
            # Title and first a hundred
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            items_x = items[pid][:100]
            items_y = items[pid][100:]
        else:
            # Title and random a hundred
            playlist_name = playlist[1]
            items_xy = items[pid][:]
            random.shuffle(items[pid])
            items_x = items[pid][:100]
            items_y = items[pid][100:]

        # Sort tracks by position
        items_xy = sorted(items_xy, key=lambda row: row[1])
        items_x = sorted(items_x, key=lambda row: row[1])
        items_y = sorted(items_y, key=lambda row: row[1])

        output_playlists_split_writer.writerow([pid, playlist_name,
                                                len(items_x), len(items_y),
                                                len(items_x + items_y)])

        for item in items_xy:
            output_items_split_writer.writerow(item)

        for item in items_x:
            output_items_split_x_writer.writerow(item)

        for item in items_y:
            output_items_split_y_writer.writerow(item)
