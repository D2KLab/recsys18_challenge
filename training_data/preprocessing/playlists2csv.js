/********************************
 * Playlists to CSV
 *********************************/

const fs = require('fs-extra');
const path = require('path');
const async = require('async');
const Track = require('./Track');
const Playlist = require('./Playlist');

const input_dir = '/Users/pasquale/Desktop/mpd.v1/data';
const output_dir = '../data';
fs.ensureDirSync(output_dir);
fs.emptyDirSync(output_dir);

fs.readdir(input_dir)
  .then(files => {
    async.map(files, (file) => {
      let playlists = parseFile(file);

      writePlaylistsCsv(playlists);
      writeTracksCsv(playlists);
      writeSequentialCsv(playlists);
    }, () => console.log('done'));
  });

function parseFile(file) {
  console.log(file);
  var {
    playlists
  } = fs.readJsonSync(path.join(input_dir, file));

  return playlists.map(pl => new Playlist(pl));
}

function writePlaylistsCsv(playlists) {
  var pl_file = path.join(output_dir, 'playlists.csv');
  // console.log('writing', pl_file);
  var header = Playlist.allProperties.join(',') + '\n';
  var pl_string = playlists
    .map(p => p.toCSV())
    .join('\n');

  writeToFile(pl_file, pl_string, header);
}

function writeTracksCsv(playlists) {
  var tracks = flatten(playlists.map(p => p.tracks));

  var tr_file = path.join(output_dir, 'tracks.csv');
  // console.log('writing', tr_file);
  var header = Track.allProperties.join(',') + '\n';
  var tr_string = tracks
    .map(t => t.toCSV())
    .join('\n');

  writeToFile(tr_file, tr_string, header);
}

function writeSequentialCsv(playlists) {
  for (let what of ['track_uri', 'album_uri', 'artist_uri']) {
    let file = path.join(output_dir, `${what}_seq.csv`);
    // console.log('writing', file);
    var content = playlists.map(p =>
      p.getTracksProp(what).join(','));
    writeToFile(file, content.join('\n'));
  }
}

function writeToFile(file, content, header) {
  var writeHeader = !fs.existsSync(file) && header;

  var stream = fs.openSync(file, 'w');
  if (writeHeader)
    fs.writeSync(stream, header, 'utf-8');

  fs.writeSync(stream, content, 'utf-8');
}

const flatten = list => list.reduce(
  (a, b) => a.concat(Array.isArray(b) ? flatten(b) : b), []
);
