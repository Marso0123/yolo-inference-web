const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('static/app.js', 'utf8');
const code = source.slice(source.indexOf('function recordingSaved('), source.indexOf('function showExport('));
const storage = new Map();
function page(storageEnabled = true) {
  const downloads = [], notes = [], displayed = [];
  const context = vm.createContext({
    handledRecordings: new Set(),
    localStorage: {
      getItem(k) { if (!storageEnabled) throw Error('unavailable'); return storage.get(k); },
      setItem(k, v) { if (!storageEnabled) throw Error('unavailable'); storage.set(k, v); },
    },
    document: {body: {append() {}}},
    showDownload(item) { displayed.push(item.id); },
    downloadLink(item) { return {click() {downloads.push(item.id);}, remove() {}}; },
    notify(message) { notes.push(message); },
  });
  vm.runInContext(code, context);
  return {saved: context.recordingSaved, downloads, notes, displayed};
}
const p = page();
p.saved({id:'historical'}, false);
p.saved({id:'historical'});
assert.deepEqual(p.downloads, []);
p.saved({id:'eof'});
p.saved({id:'eof'}); // Poll and manual HTTP response both report the same completion.
assert.deepEqual(p.downloads, ['eof']);
assert.equal(p.notes.length, 1);
const secondTab = page();
secondTab.saved({id:'eof'});
assert.deepEqual(secondTab.downloads, []);
const refreshed = page();
refreshed.saved({id:'eof'}, false);
refreshed.saved({id:'next'});
assert.deepEqual(refreshed.downloads, ['next']);
const blocked = page(false);
blocked.saved({id:'storage-disabled'});
blocked.saved({id:'storage-disabled'});
assert.deepEqual(blocked.downloads, ['storage-disabled']);
assert.deepEqual(blocked.displayed, ['storage-disabled']);
console.log('PASS: automatic completion, manual/poll deduplication, history suppression, cross-tab suppression, storage fallback');
