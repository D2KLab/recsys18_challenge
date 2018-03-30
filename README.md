# RecSys Challenge 2018
Scripts and data for the RecSys Challenge 2018.

## Dataset
[This directory](https://istitutoboella-my.sharepoint.com/:f:/g/personal/giuseppe_rizzo_ismb_it/Eh3XEurFRMFFmNXTbMpBX4YBTfRrmwVnxsLPagpwcUBscA?e=kzvYH2) contains a CSV version of the dataset released by Spotify for the RecSys Challenge 2018. We have divided the original MPD dataset in training, validation and test sets. The validation and test sets mirror the characteristics of the [challenge set](https://recsys-challenge.spotify.com/challenge_readme) provided by Spotify.

| File                              | Description                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| tracks.csv                        | The tracks of the MPD as provided by Spotify                     |
| playlists.csv                     | The playlists of the MPD as provided by Spotify                  |
| items.csv                         | The items of the MPD as provided by Spotify                      |
| playlists_challenge.csv           | The playlists of the challenge set as provided by Spotify        |
| items_challenge.csv               | The items of the challenge set as provided by Spotify            |

| File                              | Description                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| playlists_test.csv                | The playlists of the test set                                    |
| playlists_test_pid.csv            | The ids of the playlists that are included in the test set       |
| items_test.csv                    | The seed and hidden items of the test set                        |
| items_test_x.csv                  | The seed items of the test set                                   |
| items_test_y.csv                  | The hidden items of the test set                                 |

| File                              | Description                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| playlists_training_validation.csv | The playlists of the training and validation set                 |
| items_training_validation.csv     | The items of the training and validation set                     |

| File                              | Description                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| playlists_pid.csv                 | The playlists of the validation set                              |
| playlists_validation_pid.csv      | The ids of the playlists that are included in the validation set |
| items_validation.csv              | The seed and hidden items of the validation set                  |
| items_validation_x.csv            | The seed items of the validation set                             |
| items_validation_y.csv            | The hidden items of the validation set                           |

| File                              | Description                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| playlists_training.csv            | The playlists of the training set                                |
| items_training.csv                | The items of the training set                                    |

### items.csv

* pid
* pos
* track_uri

### playlists_training.csv

* pid
* name
* collaborative
* num_tracks
* num_artists
* num_albums
* num_followers
* num_edits
* modified_at
* duration_ms

### playlists_challenge.csv

* pid
* name
* num_samples
* num_holdouts
* num_tracks

###  tracks.csv

* track_uri
* track_name
* artist_uri
* artist_name
* album_uri
* album_name
* duration_ms
