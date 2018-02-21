const fs = require('fs-extra');
const path = require('path');
const sparqlTransformer = require('sparql-transformer');
const wdk = require('wikidata-sdk');
const promiseRequest = require('request-promise');
const json2csv = require('json2csv').parse;

var outDir = 'spotify2musicbrainz/out';
fs.ensureDirSync(outDir);

var options = {
  debug: true,
  sparqlFunction: wikidataQueryPerform
};


runQuery('track', {
  proto: {
    id: "?id",
    spid: "$wdt:P2207$required",
    mbid: "$wdt:P435"
  }
});
runQuery('album', {
  proto: {
    id: "?id",
    spid: "$wdt:P2205$required",
    mbid: "$wdt:P436"
  }
});
runQuery('artist', {
  proto: {
    id: "?id",
    spid: "$wdt:P1902$required",
    mbid: "$wdt:P434"
  }
});

function runQuery(name, query) {
  console.log("running for", name);
  sparqlTransformer(query, options)
    .then(refold)
    .then(res => json2csv(res, {
      quote: ''
    }))
    .then(csv => fs.writeFile(path.join(outDir, name + '.csv'), csv))
    .catch(err => console.error(err));
}

function wikidataQueryPerform(query) {
  var uri = wdk.sparqlQuery(query);
  return promiseRequest({
    uri,
    json: true
  });
}

function asArray(input) {
  if (Array.isArray(input)) return input;
  return [input];
}

function refold(input) {
  let refolded = [];
  for (let w of input)
    for (let sp of asArray(w.spid))
      refolded.push({
        spid: sp,
        mbid: asArray(w.mbid)[0],
        wdid: w.id
      });
  return refolded;
}
