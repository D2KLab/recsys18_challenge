import csv
from os import path


class Dataset:

    tracks_uri2id = {}
    tracks_id2uri = {}

    def __init__(self, root_path):
        """
        :param root_path: The directory that contains the dataset.
        """
        self.root_path = root_path

        with open(path.join(root_path, 'tracks.csv'), 'r', newline='', encoding='utf8') as tracks_file:
            tracks_reader = csv.reader(tracks_file)
            track_id = 0

            for track in tracks_reader:
                track_id += 1
                track_uri = track[0]
                self.tracks_uri2id[track_uri] = track_id
                self.tracks_id2uri[track_id] = track_uri

    def reader(self, playlists_path, items_path):
        """
        :param playlists_path: The file that contains the playlists.
        :param items_path: The file that contains the items.
        """
        playlists = {}

        with open(path.join(self.root_path, playlists_path), 'r', newline='', encoding='utf8') as playlists_file:
            playlists_reader = csv.reader(playlists_file)
            for playlist in playlists_reader:
                playlist_id = playlist[0]
                playlists[playlist_id] = playlist

        playlists_with_items = set()

        with open(path.join(self.root_path, items_path), 'r', newline='', encoding='utf8') as items_file:
            items_reader = csv.reader(items_file)
            current_playlist = None

            for item in items_reader:
                playlists_with_items.add(item[0])

                # Only the first time
                if current_playlist is None:
                    track_id = self.tracks_uri2id[item[2]]
                    current_playlist = {'pid': item[0],
                                        'title': playlists[item[0]][1],
                                        'items': [track_id]}
                    continue

                if current_playlist['pid'] == item[0]:
                    track_id = self.tracks_uri2id[item[2]]
                    current_playlist['items'].append(track_id)
                else:
                    previous_playlist = current_playlist
                    track_id = self.tracks_uri2id[item[2]]
                    current_playlist = {'pid': item[0],
                                        'title': playlists[item[0]][1],
                                        'items': [track_id]}
                    yield previous_playlist

            # Only the last time
            yield current_playlist

        for playlist_id in playlists:
            if playlist_id not in playlists_with_items:
                current_playlist = {'pid': playlist_id,
                                    'title': playlists[playlist_id][1],
                                    'items': []}
                yield current_playlist

    def writer(self, submission_path, main=True):
        """
        :param submission_path: The file in which save the submission.
        :param main: If the track is main or creative.
        """
        class DatasetWriter:

            def __init__(self, tracks_id2uri, file_path):
                self.tracks_id2uri = tracks_id2uri
                self.file = open(file_path, 'w', newline='', encoding='utf8')
                self.writer = csv.writer(self.file)

                if main is True:
                    row = ['team_info', 'D2KLab', 'main', 'diego.monti@polito.it']
                else:
                    row = ['team_info', 'D2KLab', 'creative', 'diego.monti@polito.it']
                self.writer.writerow(row)

            def __del__(self):
                self.file.close()

            def write(self, playlist):
                """
                :param playlist: The first item of the list is the playlist id, the others are item ids.
                """
                row = [playlist['pid']]

                for track_id in playlist['items']:
                    track_uri = self.tracks_id2uri[track_id]
                    row.append(track_uri)

                self.writer.writerow(row)

        return DatasetWriter(self.tracks_id2uri, submission_path)
