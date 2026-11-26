/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */

// The source-status page shows data only: statuses, stages, reason labels and
// evidence links, plus the typed Use and Method columns. Every label
// and explanation renders from the canonical taxonomy (/api/taxonomy); reason
// explanations render only when a label is clicked open; policy prose lives in
// the guide.
const state = {
  rows: [],
  taxonomy: {},
  search: '',
  status: 'all',
  reason: 'all',
  stage: 'all',
  use: 'all',
  method: 'all',
};

// Chip filter groups: filter key -> the taxonomy table and the row field it reads.
const CHIP_FILTERS = [
  ['stage', 'stage', 'stage'],
  ['use', 'use', 'use'],
  ['method', 'method', 'method_tags'],
];

function entry(kind, value) {
  return (state.taxonomy[kind] || {})[value] || null;
}

function label(kind, value) {
  if (value === undefined || value === null) return '—';
  const item = entry(kind, value);
  // Typed values always come from the taxonomy; a miss is drift, not a label.
  if (!item) throw new Error(`Untyped ${kind} value "${value}"`);
  return item.label;
}

function chipRow(value, kind) {
  const holder = node('div', undefined, 'status-chips');
  const details = node('details');
  const summary = node('summary', label(kind, value), 'chip');
  summary.title = 'Open this for what it means and what advances it';
  const body = node('div', undefined, 'status-explanation');
  const item = entry(kind, value) || {};
  body.append(node('p', item.meaning || '', 'meaning'));
  if (item.clears) body.append(node('p', `Advanced by: ${item.clears}`));
  details.append(summary, body);
  holder.append(details);
  return holder;
}

// A Deprecated row's typed replacement link, with the replacement's own status
// beside it: the replacement must exist in the Atlas in any status, and a miss
// is served-catalog drift, never a silent link.
function replacementLink(item) {
  const replacement = state.rows.find(row => row.id === item.replacement_source_id);
  const holder = node('div', undefined, 'status-chips');
  const anchor = node('a', `${item.replacement_source_id} ›`, 'chip');
  anchor.href = `source.html#${new URLSearchParams({source_id: item.replacement_source_id})}`;
  anchor.title = 'Open the replacement source';
  holder.append(anchor);
  const status = node('span', replacement ? label('status', replacement.status) : 'Unknown status', 'chip');
  status.title = replacement
    ? `The replacement's catalog status: ${entry('status', replacement.status)?.meaning || ''}`
    : `The linked replacement ${item.replacement_source_id} is not in this catalog; the audit refuses this drift.`;
  holder.append(status);
  return holder;
}

function reasonCell(claim) {
  // A plain div: the reasons cell is already a td, and nesting a td inside it makes the
  // auto table layout explode (1724px reason column, 50px source column).
  const holder = node('div', undefined, 'status-reasons');
  const details = node('details');
  const summary = node('summary', label('reason', claim.reason), 'chip');
  summary.title = 'Open this reason for what it means and what clears it';
  const body = node('div', undefined, 'status-explanation');
  const item = entry('reason', claim.reason) || {};
  body.append(
    node('p', item.meaning || '', 'meaning'),
    node('p', item.clears ? `Cleared by: ${item.clears}` : '')
  );
  if (claim.evidence) {
    const evidence = node('p');
    evidence.append(node('span', `${claim.evidence.kind}: ${claim.evidence.message}`));
    body.append(evidence);
  }
  details.append(summary, body);
  holder.append(details);
  return holder;
}

function evidenceCell(claims) {
  const cell = node('td', undefined, 'status-evidence');
  const links = new Map();
  for (const claim of claims || []) {
    const uri = claim.evidence && claim.evidence.uri;
    if (uri && !links.has(uri)) links.set(uri, claim.reason);
  }
  if (!links.size) {
    cell.append(node('span', '—'));
    return cell;
  }
  for (const [uri, reason] of links) {
    const link = node('a', label('reason', reason));
    link.href = uri;
    link.rel = 'noopener';
    cell.append(link);
  }
  return cell;
}

function columnCell(value, kind) {
  const cell = node('td');
  if (value === undefined || value === null || (Array.isArray(value) && !value.length)) {
    cell.append(node('span', '—'));
    return cell;
  }
  const values = Array.isArray(value) ? value : [value];
  const holder = node('div', undefined, 'status-reasons');
  for (const item of values) {
    const chip = node('span', label(kind, item), 'chip');
    chip.title = entry(kind, item)?.meaning || '';
    holder.append(chip);
  }
  cell.append(holder);
  return cell;
}

// Every typed value a row renders must be an enum member served by /api/taxonomy.
// An out-of-enum value is served-catalog drift: fail loudly instead of rendering
// the raw string as a label.
function requireTypedRows(rows) {
  const fail = (row, field, value) => {
    throw new Error(`Untyped ${field} value "${value}" on ${row.id}; the served catalog drifted from the taxonomy.`);
  };
  for (const row of rows) {
    if (!entry('status', row.status)) fail(row, 'status', row.status);
    if (row.stage && !entry('stage', row.stage)) fail(row, 'stage', row.stage);
    if (row.replacement_source_id && !rows.some(other => other.id === row.replacement_source_id)) {
      fail(row, 'replacement_source_id', row.replacement_source_id);
    }
    if (row.use !== undefined && row.use !== null && !entry('use', row.use)) fail(row, 'use', row.use);
    for (const tag of row.method_tags || []) if (!entry('method', tag)) fail(row, 'method', tag);
    for (const claim of row.reasons || []) if (!entry('reason', claim.reason)) fail(row, 'reason', claim.reason);
  }
}

function lastChange(row) {
  if (row.receipt) return row.receipt;
  return row.generated_at || '—';
}

function row(item) {
  const tr = node('tr');
  tr.dataset.sourceId = item.id;
  const source = node('td');
  source.append(node('strong', item.display_name || item.id));
  if (item.id !== item.display_name) {
    source.append(node('div', item.id, 'sr-only'));
  }
  tr.append(source);
  const status = node('td');
  const statusChip = node('span', label('status', item.status), 'chip');
  statusChip.title = entry('status', item.status)?.meaning || '';
  status.append(statusChip);
  if (item.consumer_findings?.length) {
    const flag = node('span', 'Consumer advisory', 'chip');
    flag.title = item.consumer_findings.join(', ');
    status.append(flag);
  }
  if (item.stage) status.append(chipRow(item.stage, 'stage'));
  if (item.replacement_source_id) status.append(replacementLink(item));
  tr.append(status, columnCell(item.use, 'use'), columnCell(item.method_tags, 'method'));
  const claims = item.reasons || [];
  const reasons = node('td');
  if (claims.length) {
    for (const claim of claims) reasons.append(reasonCell(claim));
  } else {
    reasons.append(node('span', '—'));
  }
  tr.append(reasons, evidenceCell(claims), node('td', lastChange(item)));
  return tr;
}

function matches(item) {
  if (state.status !== 'all' && item.status !== state.status) return false;
  if (state.reason !== 'all' && !(item.reasons || []).some(claim => claim.reason === state.reason)) return false;
  for (const [key, , field] of CHIP_FILTERS) {
    if (state[key] === 'all') continue;
    const value = item[field];
    const values = Array.isArray(value) ? value : [value];
    if (!values.includes(state[key])) return false;
  }
  if (state.search) {
    const haystack = `${item.id} ${item.display_name || ''}`.toLowerCase();
    if (!haystack.includes(state.search)) return false;
  }
  return true;
}

function render() {
  const rows = state.rows.filter(matches);
  const body = $('rows');
  body.replaceChildren(...rows.map(row));
  $('empty').hidden = rows.length > 0;
  $('visible-count').textContent = `${rows.length} of ${state.rows.length} sources`;
  for (const [key] of CHIP_FILTERS) renderChips(key);
}

// "all" participates as an explicit chip so every group shows its full value set.
function renderChips(key) {
  const [, kind, field] = CHIP_FILTERS.find(([name]) => name === key);
  const holder = $(`${key}-filters`);
  if (!holder) return;
  const values = [...new Set(state.rows.flatMap(item => {
    const value = item[field];
    return Array.isArray(value) ? value : value === undefined || value === null ? [] : [value];
  }))].filter(value => entry(kind, value));
  const selected = state[key];
  holder.replaceChildren();
  for (const value of ['all', ...values.sort()]) {
    const item = value === 'all' ? null : value;
    const count = item === null
      ? state.rows.length
      : state.rows.filter(row => {
          const raw = row[field];
          const values = Array.isArray(raw) ? raw : raw === undefined || raw === null ? [] : [raw];
          return values.includes(item);
        }).length;
    const button = node('button', `${item === null ? 'All' : label(kind, item)} ${count}`, 'chip');
    button.type = 'button';
    button.setAttribute('aria-pressed', String(selected === value));
    button.addEventListener('click', () => {
      state[key] = state[key] === value ? 'all' : value;
      render();
    });
    holder.append(button);
  }
}

function counts() {
  const holder = $('status-counts');
  holder.replaceChildren();
  for (const status of Object.keys(state.taxonomy.status || {})) {
    const count = state.rows.filter(item => item.status === status).length;
    const button = node('button', `${label('status', status)} ${count}`, 'chip');
    button.type = 'button';
    button.setAttribute('aria-pressed', String(state.status === status));
    button.addEventListener('click', () => {
      $('status-filter').value = state.status = state.status === status ? 'all' : status;
      render();
    });
    holder.append(button);
  }
}

function needsRuling(rows) {
  const flagged = rows.filter(item => (item.reasons || []).some(claim => claim.reason === 'unspecified-legacy'));
  if (!flagged.length) return;
  const section = $('needs-ruling');
  section.hidden = false;
  const list = $('needs-ruling-list');
  for (const item of flagged) {
    const entry = node('li');
    entry.append(node('strong', item.display_name || item.id), node('span', ` — ${item.id}`));
    list.append(entry);
  }
}

// The advisory re-seal queue (ruling 2026-11-26): train publications whose
// stored graders predate main's recipe fixes, with the main commits and PRs
// that touch the changed files. Never gates anything; the queue drives re-seal.
function graderDrift(extras) {
  const rows = extras.grader_drift || [];
  if (!rows.length) return;
  const section = $('grader-drift');
  section.hidden = false;
  const list = $('grader-drift-list');
  for (const item of rows) {
    const entry = node('li');
    const fixes = (item.fix_commits || [])
      .map(commit => commit.pr ? `#${commit.pr}` : commit.sha.slice(0, 7))
      .join(', ');
    entry.append(node('strong', item.display_name || item.id), node('span', ` — ${item.id}`));
    if (fixes) entry.append(node('span', ` · main fixes: ${fixes}`));
    list.append(entry);
  }
}

async function load() {
  const [catalog, extras, taxonomy] = await Promise.all([
    json('api/source-statuses'),
    json('measurements/source-status-extras.json'),
    loadTaxonomy(),
  ]);
  state.taxonomy = taxonomy;
  const reasonFilter = $('reason-filter');
  reasonFilter.replaceChildren(new Option('All reasons', 'all'));
  for (const [value, item] of Object.entries(taxonomy.reason || {})) {
    reasonFilter.append(new Option(item.label, value));
  }
  const statusFilter = $('status-filter');
  statusFilter.replaceChildren(new Option('All statuses', 'all'));
  for (const [value, item] of Object.entries(taxonomy.status || {})) {
    statusFilter.append(new Option(item.label, value));
  }
  const stages = extras.stages || {};
  const main = (catalog.sources || []).map(declaration => ({
    id: declaration.id,
    display_name: declaration.display_name,
    status: declaration.status,
    stage: stages[declaration.id] || declaration.stage || null,
    use: declaration.intended_use,
    method_tags: declaration.method_tags || [],
    reasons: declaration.exclusion_claims || [],
    notes: declaration.notes,
    consumer_findings: declaration.consumer_findings,
    receipt: catalog.snapshot_id ? `catalog snapshot ${catalog.snapshot_id}` : null,
    generated_at: declaration.updated_at,
  }));
  const queued = [...(extras.pending || []), ...(extras.metadata_exclusions || [])];
  // Every source ID appears exactly once: the served catalog row wins, and an
  // extras row for the same ID only donates its stage.
  const byId = new Map(main.map(item => [item.id, item]));
  for (const item of queued) {
    const known = byId.get(item.id);
    if (known) {
      known.stage = known.stage || item.stage || null;
      continue;
    }
    byId.set(item.id, item);
  }
  state.rows = [...byId.values()];
  requireTypedRows(state.rows);
  const generated = Math.max(
    0,
    ...state.rows.map(item => (item.generated_at ? Date.parse(item.generated_at) : 0))
  );
  $('metric-date').textContent = generated ? new Date(generated).toISOString().slice(0, 10) : '—';
  if (catalog.snapshot_id) {
    $('snapshot-info').textContent = `Catalog snapshot ${catalog.snapshot_id} · loaded ${catalog.synced_at || ''}`;
  }
  counts();
  render();
  needsRuling(state.rows);
  graderDrift(extras);
  $('message').textContent = '';
}
window.addEventListener('DOMContentLoaded', () => {
  $('status-table').addEventListener('focusin', event => {
    // Native table scrolling can leave a reason or evidence link mostly outside
    // the narrow scroller. Run after native focus scrolling and keep the target
    // centered without moving the reader to another row.
    const target = event.target;
    if (target instanceof HTMLElement) {
      requestAnimationFrame(() => target.scrollIntoView({ block: 'nearest', inline: 'center' }));
    }
  });
  load().catch(error => {
    $('message').textContent = error.message;
    $('message').classList.add('error');
  });
  $('search').addEventListener('input', event => {
    state.search = event.target.value.trim().toLowerCase();
    render();
  });
  $('status-filter').addEventListener('change', event => {
    state.status = event.target.value;
    render();
  });
  $('reason-filter').addEventListener('change', event => {
    state.reason = event.target.value;
    render();
  });
  $('clear-filters').addEventListener('click', () => {
    state.search = '';
    state.status = 'all';
    state.reason = 'all';
    for (const [key] of CHIP_FILTERS) state[key] = 'all';
    $('search').value = '';
    $('status-filter').value = 'all';
    $('reason-filter').value = 'all';
    render();
  });
});
