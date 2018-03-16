# RecSys Challenge 2018
Scripts and data for the RecSys Challenge 2018.

## Dataset
[This directory](https://istitutoboella-my.sharepoint.com/:f:/g/personal/giuseppe_rizzo_ismb_it/Eh3XEurFRMFFmNXTbMpBX4YBTfRrmwVnxsLPagpwcUBscA?e=kzvYH2) contains a CSV version of the dataset released by Spotify for the RecSys Challenge 2018. We have divided the original MPD dataset in a training and a validation set. The validation set mirrors the characteristics of the [challenge set](https://recsys-challenge.spotify.com/challenge_readme) provided by Spotify.

| File                        | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| items_challenge.csv         | The items of the test set challenge as provided by Spotify       |
| items_mpd.csv               | The items of the training set MPD as provided by Spotify         |
| items_training.csv          | The items of a subset of MPD created for training purposes       |
| items_validation.csv        | The items of a subset of MPD created for validation purposes     |
| items_validation_hidden.csv | The items that should be predicted by the recommender            |
| pid_validation.csv          | The ids of the playlists that are included in the validation set |
| playlists_challenge.csv     | The playlists of the test set challenge as provided by Spotify   |
| playlists_mpd.csv           | The playlists of the training set MPD as provided by Spotify     |
| playlists_training.csv      | The playlists of a subset of MPD created for training purposes   |
| playlists_validation.csv    | The playlists of a subset of MPD created for validation purposes |
| tracks_mpd.csv              | The tracks of the training set MPD as provided by Spotify        |
| tracks_mpd_count.csv        | The tracks and the number of items per track in the MPD          |

### items_*.csv

* pid
* pos
* track_uri

### playlists_(mpd|training).csv

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

### playlists_(challenge|validation).csv

* pid
* name
* num_samples
* num_holdouts
* num_tracks

###  tracks_mdp.csv

* track_uri
* track_name
* artist_uri
* artist_name
* album_uri
* album_name
* duration_ms
