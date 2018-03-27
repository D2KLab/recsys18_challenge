var properties = [];

function addProperty(p) {
  if (!properties.includes(p))
    properties.push(p);
}

class Track {
  constructor(track_obj, position, pid) {
    for (let k of Object.keys(track_obj)) {
      addProperty(k);
      this[k] = track_obj[k];
    }
    addProperty('pos');
    this.pos = position;
    addProperty('pid');
    this.pid = pid;
  }
  toCSV() {
    return properties
      .map(p => this.hasOwnProperty(p) ? this[p] : '')
      .map(t => {
        if (typeof t != 'string') return t;
        return t.replace(/&gt;/g, '>')
          .replace(/&lt;/g, '<')
          .replace(/;/g, ' ');
      })
      .join(';')
      .replace(/"/g, '""');
  }

  static get allProperties() {
    return properties;
  }
}


module.exports = Track;
