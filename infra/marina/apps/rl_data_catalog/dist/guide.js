/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// Fills the static ingestion guide with counts from the catalog snapshot Atlas serves.
const number = new Intl.NumberFormat('en-US');
const leafTitles = {
  inventory: 'Inventory only', grandfathered: 'Grandfathered blends', unreviewed: 'Runnable, oversight not declared',
  excluded: 'Excluded', deprecated: 'Deprecated', sample: 'Admitted, oversight SAMPLE', full: 'Admitted, oversight FULL',
};
const state = {sources: [], leaves: {}};

// The grandfather list is exactly the agentic MarinSkyRL blends: the format gate
// rejects every other agentic source with MarinSkyRL's internal format.
function grandfathered(row) {
  return row.origin === 'MarinSkyRL' && (row.tags || []).includes('agentic');
}
function leafOf(row) {
  if (row.status === 'Deprecated') return 'deprecated';
  if (row.status === 'Excluded') return 'excluded';
  if (row.status === 'Inventory') return 'inventory';
  if (grandfathered(row)) return 'grandfathered';
  if (row.oversight === 'full') return 'full';
  if (row.oversight === 'sample') return 'sample';
  return 'unreviewed';
}
function tally(rows, key) {
  const counts = {};
  for (const row of rows) { const item = key(row) ?? 'none'; counts[item] = (counts[item] || 0) + 1; }
  return counts;
}
function sources(count) { return `${number.format(count)} source${count === 1 ? '' : 's'}`; }
function fill(selector, text) { for (const item of document.querySelectorAll(selector)) item.textContent = text; }
function summary(counts, labels) {
  return labels.filter(([key]) => counts[key]).map(([key, label]) => `${label} ${number.format(counts[key])}`).join(' · ') || 'none recorded';
}
function daytonaSummary(rows) {
  const counts = tally(rows, row => row.daytona_safety || 'none');
  return `Daytona: ${summary(counts, [['safe', 'safe'], ['unsafe', 'unsafe'], ['unassessed', 'unassessed'], ['none', 'not recorded']])}`;
}

function render(data) {
  state.sources = data.sources;
  state.leaves = {};
  for (const row of data.sources) (state.leaves[leafOf(row)] ||= []).push(row);
  for (const key of Object.keys(leafTitles)) {
    const rows = state.leaves[key] || [];
    let text = `${sources(rows.length)} in this snapshot`;
    if (key === 'unreviewed') {
      const agentic = rows.filter(row => (row.tags || []).includes('agentic')).length;
      text = `${number.format(agentic)} agentic need QA; ${number.format(rows.length - agentic)} non-agentic rest on campaign records`;
    }
    fill(`[data-count="${key}"]`, text);
    fill(`[data-count="${key}-n"]`, number.format(rows.length));
    fill(`[data-daytona="${key}"]`, daytonaSummary(rows));
  }
  const modes = tally(data.sources, row => row.curation?.mode);
  fill('[data-fill="modes"]', `Last pass in this snapshot: ${summary(modes, [['full', 'FULL'], ['sample', 'SAMPLE'], ['quick', 'QUICK'], ['none', 'none']])}`);
  const oracle = tally(data.sources, row => row.oracle_validation);
  fill('[data-fill="oracle"]', `Oracle: ${summary(oracle, [['executed', 'executed'], ['verifier_executed', 'verifier executed'], ['skipped', 'skipped'], ['none', 'not recorded']])}`);
  const limit = data.daytona_snapshot_limit;
  if (Number.isInteger(limit)) {
    fill('[data-fill="daytona-rule"]', `over ${limit} snapshots or a refusal is unsafe, still ingested`);
    fill('[data-fill="daytona-text"]', `A source is unsafe when its tasks need more than ${limit} unique snapshots, or when Daytona refuses any task's declared execution requirements.`);
    fill('[data-fill="daytona-safe"]', `The source's tasks need at most ${limit} unique environments, and Daytona accepts every task's execution requirements.`);
    fill('[data-fill="daytona-unsafe"]', `A run would need more than ${limit} snapshots or something Daytona refuses. The source stays ingested; run it on gVisor or the local backend, or rebuild it faithfully.`);
  }
  const predating = data.sources.filter(row => !('oversight' in row)).length;
  const loaded = new Date(data.synced_at).toLocaleString();
  $('message').textContent = `${sources(data.sources.length)} in catalog snapshot ${data.snapshot_id.slice(0, 8)}, loaded ${loaded}.` +
    (predating ? ` ${sources(predating)} predate the oversight field and count as oversight not declared.` : '');
  $('snapshot-info').textContent = `Counts come from catalog snapshot ${data.snapshot_id}, created ${new Date(data.created_at).toLocaleString()} and loaded ${loaded}. They change when an operator loads a newer snapshot.`;
}

function showLeaf(key) {
  for (const item of document.querySelectorAll('.pill[data-leaf]')) item.classList.toggle('selected', item.dataset.leaf === key);
  if (!key || !leafTitles[key]) { $('leaf-list').hidden = true; return; }
  const rows = [...(state.leaves[key] || [])].sort((a, b) => a.display_name.localeCompare(b.display_name));
  $('leaf-list-title').textContent = `${leafTitles[key]}: ${sources(rows.length)}`;
  $('leaf-rows').replaceChildren(...rows.map(row => {
    const tr = node('tr'), name = node('td'), link = node('a', row.display_name);
    link.href = `./#${new URLSearchParams({search: row.name})}`;
    name.append(link, node('small', row.id));
    tr.append(name, node('td', row.origin), node('td', row.oversight || 'not declared'),
      node('td', row.daytona_safety || 'not recorded'), node('td', row.oracle_validation || 'not recorded'));
    return tr;
  }));
  $('leaf-list').hidden = false;
}
function selectLeaf(key) {
  history.replaceState(null, '', key ? `#${new URLSearchParams({leaf: key})}` : location.pathname);
  showLeaf(key);
  if (key) $('leaf-list').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

for (const item of document.querySelectorAll('[data-leaf]')) {
  item.addEventListener('click', () => selectLeaf(item.dataset.leaf));
  if (item.getAttribute('role') === 'button') {
    item.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectLeaf(item.dataset.leaf); }
    });
  }
}
$('leaf-list-close').addEventListener('click', () => selectLeaf(null));

json('api/sources').then(data => {
  render(data);
  showLeaf(new URLSearchParams(location.hash.slice(1)).get('leaf'));
}).catch(error => {
  $('message').textContent = `Live counts are unavailable: ${error.message}. The guide text below does not depend on them.`;
  $('message').classList.add('error');
  fill('[data-count]', 'Counts unavailable');
});
