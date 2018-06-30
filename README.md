# RecSys Challenge 2018
Scripts for the RecSys Challenge 2018 from the D2KLab team.

## Install Dependencies

    pip install -r requirements.txt

## Dataset
We converted the original JSON files in an equivalent CSV version.

```
python evaluation/mpd2csv.py --mpd_path /path/to/mpd --out_path dataset
python evaluation/challenge2csv.py --challenge_path /path/to/challenge.json --out_path dataset
```

We have divided the MPD dataset in training, validation and test sets. The validation and test sets mirror the characteristics of the official challenge set.

```
python evaluation/split.py --path dataset --input_playlists playlists.csv --input_items items.csv --output_playlists playlists_training_validation.csv --output_items items_training_validation.csv --output_playlists_split playlists_test.csv --output_playlists_split_pid playlists_test_pid.csv --output_items_split items_test.csv --output_items_split_x items_test_x.csv --output_items_split_y items_test_y.csv --scale 1000
python evaluation/split.py --path dataset --input_playlists playlists_training_validation.csv --input_items items_training_validation.csv --output_playlists playlists_training.csv --output_items items_training.csv --output_playlists_split playlists_validation.csv --output_playlists_split_pid playlists_validation_pid.csv --output_items_split items_validation.csv --output_items_split_x items_validation_x.csv --output_items_split_y items_validation_y.csv --scale 1000
```


## Embeddings

We rely on 2 set of embeddings that we use as input of a Neural Network.

### Word2Rec

Embeddings representing the tracks/albums/artists based on their co-occurrence in playlists.

```
python main.py word2rec_item word2rec.csv --w2r models/embs/1M/word2rec_dry.w2v --dataset dataset
python main.py word2rec_album word2rec_album.csv --w2r models/embs/1M/word2rec_dry_albums.w2v --dataset dataset
python main.py word2rec_artist word2rec_artist.csv --w2r models/embs/1M/word2rec_dry_artists.w2v --dataset dataset
```

### Title2Rec

FastText embeddings representing the playlists' titles, computed on cluster of playlists based on the word2rec embeddings of their tracks.

```
python main.py title2rec title2rec.csv
python main.py title2rec_embs models/fast_text/title2rec.npy
```

### Creative Track features

`mpd_uri_topics` and `spotify_uri_features.pickle` are pickle files containing features extracted from song lyrics such as the dominant topics, the emotions, the style and so on.

For use these features (only for _Creative Track_):
- download the 2 files in this [folder](https://drive.google.com/drive/folders/1rrNwp1LIuXXyIr0P1xgT7DfyVqmuqGIz?usp=sharing);
- unzip them in `<projectpath>\models\lyrics`.

Then, they can be used in the RNN by using the argument `--lyrics`.

## RNN

### Training

```
python recommender/mpd_rnn.py --data_path=dataset --model=optimal --save_path=/path/to/model/optimal --embs=models/embs/1M --title_embs=models/fast_text/title2rec.npy
```

### Generation

```
python recommender/mpd_rnn.py --data_path=dataset --model=optimal --save_path=/path/to/model --embs=models/embs/1M --title_embs=models/fast_text/title2rec.npy --smooth=linear --sample_file=solution.csv --is_dry=False
```

Add `--lyrics` in order to include the Creative Track features.

## Ensemble

Ensemble allows to combine predictions of a set of RNN configurations to increase the accuracy. To use it, you need to put the submission files inside a folder `submissions/dry` and it will try all possible combinations of the submissions files and save them into files.

```
python recommender/ensemble.py
```
