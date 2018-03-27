const Track = require('./Track');

var properties = [];

function addProperty(p) {
  if (!properties.includes(p))
    properties.push(p);
}

class Playlist {
  constructor(playlist_obj) {
    for (let k of Object.keys(playlist_obj)) {
      addProperty(k);
      this[k] = playlist_obj[k];
    }
  }

  toCSV() {
    return properties
      .filter(noTracks)
      .map(p => this.hasOwnProperty(p) ? this[p] : '')
      .map(t => {
        if(typeof t != 'string') return t;
        return t.replace(/&gt;/g, '>')
        .replace(/&lt;/g, '<')
        .replace(/;/g, ' ');
      })
      .join(';')
      .replace(/"/g, '""');
  }

  get tracks() {
    return this._tracks;
  }

  set tracks(track_list) {
    this._tracks = track_list
      .map((t, i) => new Track(t, i, this.pid));
  }

  getTracksProp(prop) {
    return this._tracks.map(t => t[prop]);
  }

  static get allProperties() {
    return properties.filter(noTracks);
  }
}

function noTracks(p) {
  return !['tracks', '_tracks'].includes(p);
}


module.exports = Playlist;
