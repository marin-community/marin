/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// Fills the static ingestion guide with counts from the catalog snapshot Atlas serves.
const number = new Intl.NumberFormat('en-US');
const leafTitles = {
  grandfathered: 'Grandfathered blends', unreviewed: 'QUICK, converted without review',
  excluded: 'Excluded', deprecated: 'Deprecated', sample: 'Admitted, oversight SAMPLE', full: 'Admitted, oversight FULL',
};
const state = {sources: [], leaves: {}, loaded: false};

// The grandfather list is exactly the agentic MarinSkyRL blends: the format gate
// rejects every other agentic source with MarinSkyRL's internal format.
function grandfathered(row) {
  return Boolean(row.daytona_exemption);
}
function leafOf(row) {
  if (row.status === 'Deprecated') return 'deprecated';
  if (row.status === 'Excluded') return 'excluded';
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
function daytonaState(row) {
  return row.daytona?.state || row.daytona_safety || 'none';
}
function daytonaSummary(rows) {
  const counts = tally(rows, daytonaState);
  return `Daytona check: ${summary(counts, [
    ['safe', 'fits budget'], ['unsafe', 'exceeds budget'], ['unassessed', 'unmeasured'],
    ['unmeasured', 'unmeasured'], ['na', 'no sandbox'], ['none', 'not recorded'],
  ])}`;
}
function sumOf(rows, value) {
  return rows.reduce((total, row) => total + (value(row) ?? 0), 0);
}
// Coverage denominators for the admitted leaves: retained tasks from the curation
// summary, reviewed tasks from the oversight evidence. A field that was never
// recorded stays out of the text instead of counting as zero.
function coverageText(rows) {
  if (!rows.length) return 'Coverage: no sources in this snapshot';
  const retainedRows = rows.filter(row => row.curation?.admission_counts?.admitted != null || row.oversight_evidence?.admitted != null);
  const reviewedRows = rows.filter(row => row.oversight_evidence?.reviewed != null);
  if (!retainedRows.length && !reviewedRows.length) return 'Coverage not recorded in this snapshot';
  const parts = [];
  if (retainedRows.length) {
    parts.push(`${number.format(sumOf(retainedRows, row => row.curation?.admission_counts?.admitted ?? row.oversight_evidence?.admitted))} retained tasks`);
  }
  if (reviewedRows.length) {
    parts.push(`review covered ${number.format(sumOf(reviewedRows, row => row.oversight_evidence?.reviewed))}`);
  }
  return `Coverage: ${parts.join(', ')}`;
}

function render(data) {
  state.sources = data.sources;
  state.leaves = {};
  state.loaded = true;
  for (const row of data.sources) (state.leaves[leafOf(row)] ||= []).push(row);
  for (const key of Object.keys(leafTitles)) {
    const rows = state.leaves[key] || [];
    let text = `${sources(rows.length)} in this snapshot`;
    fill(`[data-count="${key}"]`, text);
    fill(`[data-count="${key}-n"]`, number.format(rows.length));
    fill(`[data-daytona="${key}"]`, daytonaSummary(rows));
    fill(`[data-coverage="${key}"]`, coverageText(rows));
  }
  const modes = tally(data.sources, row => row.curation?.mode);
  fill('[data-fill="modes"]', `Last pass in this snapshot: ${summary(modes, [['full', 'FULL'], ['sample', 'SAMPLE'], ['quick', 'QUICK'], ['none', 'none']])}`);
  const oracle = tally(data.sources, row => row.oracle_validation);
  fill('[data-fill="oracle"]', `Oracle: ${summary(oracle, [['executed', 'executed'], ['verifier_executed', 'verifier executed'], ['skipped', 'skipped'], ['none', 'not recorded']])}`);
  const limit = data.daytona_snapshot_limit;
  if (Number.isInteger(limit)) {
    fill('[data-fill="daytona-rule"]', `beyond the snapshot cap or refused: still ingested`);
    fill('[data-fill="daytona-text"]', `The check maps each task's declared actor environment to snapshot identities through the production naming path and applies the recorded capability rules. A source exceeds the budget when its tasks need more than ${limit} unique snapshots, or when the rules refuse a task's declared requirements.`);
    fill('[data-fill="daytona-safe"]', `The source's declared environments map to at most ${limit} unique snapshot identities and pass the recorded capability rules. Static measurement only: no provisioning, quota headroom or rollout was tested.`);
    fill('[data-fill="daytona-unsafe"]', `A run would need more than ${limit} snapshots, or asks for something the capability rules refuse. The source stays ingested; gVisor or a rebuilt, digest-pinned image are options to validate, not guaranteed remedies.`);
  }
  const predating = data.sources.filter(row => !('oversight' in row)).length;
  const loaded = new Date(data.synced_at).toLocaleString();
  $('message').textContent = `${sources(data.sources.length)} in catalog snapshot ${data.snapshot_id.slice(0, 8)}, loaded ${loaded}.` +
    (predating ? ` ${sources(predating)} predate the oversight field and count as oversight not declared.` : '');
  $('snapshot-info').textContent = `Counts come from catalog snapshot ${data.snapshot_id}, created ${new Date(data.created_at).toLocaleString()} and loaded ${loaded}. They change when an operator loads a newer snapshot. The rules on this page are the current policy; an older snapshot can lack fields the policy now records.`;
}

function showLeaf(key) {
  if (!state.loaded) {
    $('leaf-list-title').textContent = 'Source list unavailable: the catalog did not load';
    $('leaf-rows').replaceChildren();
    $('leaf-list').hidden = false;
    return;
  }
  for (const item of document.querySelectorAll('.pill[data-leaf]')) item.classList.toggle('selected', item.dataset.leaf === key);
  if (!key || !leafTitles[key]) { $('leaf-list').hidden = true; return; }
  const rows = [...(state.leaves[key] || [])].sort((a, b) => a.display_name.localeCompare(b.display_name));
  $('leaf-list-title').textContent = `${leafTitles[key]}: ${sources(rows.length)}`;
  $('leaf-rows').replaceChildren(...rows.map(row => {
    const tr = node('tr'), name = node('td'), link = node('a', row.display_name);
    link.href = `./#${new URLSearchParams({search: row.name})}`;
    name.append(link, node('small', row.id));
    tr.append(name, node('td', row.origin), node('td', row.oversight || 'not declared'),
      node('td', row.daytona?.state || row.daytona_safety || 'not recorded'), node('td', row.oracle_validation || 'not recorded'));
    return tr;
  }));
  $('leaf-list').hidden = false;
}
function selectLeaf(key) {
  if (!state.loaded) { showLeaf(null); return; }
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
  // Finish every placeholder so no field keeps saying "loading" beside a finished
  // error state, and no control can fabricate a zero from an empty catalog.
  $('message').textContent = `Catalog unavailable: ${error.message}. Counts, source lists, coverage and the configured Daytona limit cannot be shown; the text below describes the current policy, not verified properties of a loaded snapshot.`;
  $('message').classList.add('error');
  fill('[data-count]', 'Counts unavailable');
  fill('[data-count-n]', '—');
  fill('[data-daytona]', 'Daytona check: unavailable');
  fill('[data-coverage]', 'Coverage: unavailable');
  fill('[data-fill="modes"]', 'Modes in this snapshot: unavailable');
  fill('[data-fill="oracle"]', 'Oracle: unavailable');
  for (const pill of document.querySelectorAll('.pill[data-leaf]')) pill.setAttribute('aria-disabled', 'true');
  for (const button of document.querySelectorAll('button.count')) button.disabled = true;
});
