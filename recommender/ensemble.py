from collections import Counter, defaultdict
from itertools import combinations
import os

dry = True

if dry:

    path = 'submissions/dry/'

    file_names = [file_name for file_name in os.listdir(path)]

else:

    path = 'submissions/challenge/'

    file_names = [file_name for file_name in os.listdir(path)]

n = len(file_names)

with open('combination_index.txt', 'w') as combination_index_file:

    index = 1

    # groups of 2 up to n-1

    for c in range(2, n):

        # all possible combinations of size c

        for combination in combinations(file_names, c):

            challenge_type = 'main'

            print(index)

            combination_index_file.write('%d,' % index)

            playlist_song_score = Counter()  # (playlist_id, song) : total_score

            for file_name in combination:

                if 'fuzzy' in file_name or 'emotion' in file_name:

                    challenge_type = 'creative'

                with open(path+file_name) as file:

                    for i, line in enumerate(file):

                        if i > 0:  # skip header

                            line_split = line.strip('\n').split(',')

                            playlist_id = line_split[0]

                            tracks = line_split[1:]

                            assert len(tracks) == 500

                            for j, song in enumerate(tracks):

                                score = 500 - j  # higher score to first ranked

                                playlist_song_score[(playlist_id, song)] += score

                combination_index_file.write('%s,' % file_name)

            combination_index_file.write('\n')

            output_dict = defaultdict(dict)  # dict of dict

            for (playlist_id, song), score in playlist_song_score.items():

                output_dict[playlist_id][song] = score

            with open('ensemble_%d.csv' % index, 'w') as output_write:

                output_write.write('team_info,D2KLab,%s,diego.monti@polito.it\n' % challenge_type)

                for playlist_id in output_dict.keys():

                    tracks = output_dict[playlist_id]

                    sorted_tracks = sorted(tracks, key=lambda x: tracks[x], reverse=True)[0:500]

                    output_write.write('%s,' % playlist_id)

                    for track in sorted_tracks[:-1]:

                        output_write.write('%s,' % track)

                    output_write.write('%s\n' % sorted_tracks[-1])

            index += 1

















