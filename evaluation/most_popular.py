import argparse
import csv

parser = argparse.ArgumentParser(description="Most Popular")

parser.add_argument('--items_training', default=None, required=True)
parser.add_argument('--playlists_test', default=None, required=True)
parser.add_argument('--items_test', default=None, required=True)
parser.add_argument('--items_submission', default=None, required=True)

args = parser.parse_args()

items_training_file = open(args.items_training, 'r', newline='', encoding='utf8')
playlists_test_file = open(args.playlists_test, 'r', newline='', encoding='utf8')
items_test_file = open(args.items_test, 'r', newline='', encoding='utf8')
items_submission_file = open(args.items_submission, 'w', newline='', encoding='utf8')

items_training_reader = csv.reader(items_training_file)
playlists_test_reader = csv.reader(playlists_test_file)
items_test_reader = csv.reader(items_test_file)
items_submission_writer = csv.writer(items_submission_file)

# Count the items
items_count = {}

for row in items_training_reader:
    track_uri = row[2]
    if track_uri in items_count:
        items_count[track_uri] += 1
    else:
        items_count[track_uri] = 1

items_sorted = sorted(items_count, key=items_count.get, reverse=True)

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
    if pid in items_test:
        # Set difference
        count = 0
        for track_uri in items_sorted:
            if track_uri not in items_test[pid]:
                row_submission.append(track_uri)
                count += 1
            if count >= 500:
                break
    else:
        row_submission += items_sorted[:500]

    items_submission_writer.writerow(row_submission)
