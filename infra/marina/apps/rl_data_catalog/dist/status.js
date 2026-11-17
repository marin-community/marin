/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */

// The source-status page shows data only: statuses, reason labels and evidence
// links. Reason explanations render only when a label is clicked open; policy
// prose lives in the guide.
const STATUS_COUNTS = ['Pending', 'Available', 'Excluded', 'Deprecated'];

const state = { rows: [], explanations: {}, search: '', status: 'all', reason: 'all' };

function label(reason) {
  const entry = state.explanations[reason];
  return entry ? entry.label : reason;
}

function reasonCell(claim) {
  // A plain div: the reasons cell is already a td, and nesting a td inside it makes the
  // auto table layout explode (1724px reason column, 50px source column).
  const holder = node('div', undefined, 'status-reasons');
  const details = node('details');
  const summary = node('summary', label(claim.reason), 'chip');
  summary.title = 'Open this reason for what it means and what clears it';
  const body = node('div', undefined, 'status-explanation');
  const entry = state.explanations[claim.reason] || {};
  body.append(
    node('p', entry.meaning || '', 'meaning'),
    node('p', entry.clears ? `Cleared by: ${entry.clears}` : '')
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
    const link = node('a', label(reason));
    link.href = uri;
    link.rel = 'noopener';
    cell.append(link);
  }
  return cell;
}

function lastChange(row) {
  if (row.receipt) return row.receipt;
  return row.generated_at || '—';
}

function row(item) {
  const tr = node('tr');
  tr.dataset.sourceId = item.id;
  const source = node('td');
  source.append(node('strong', item.display_name || item.name || item.id));
  if (item.id !== (item.display_name || item.name)) {
    source.append(node('div', item.id, 'sr-only'));
  }
  tr.append(source);
  const status = node('td');
  status.append(node('span', item.status, 'chip'));
  if (item.stage) status.append(node('div', item.stage, 'status-stage'));
  tr.append(status);
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
  if (state.search) {
    const haystack = `${item.id} ${item.display_name || ''} ${item.name || ''}`.toLowerCase();
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
}

function counts() {
  const holder = $('status-counts');
  holder.replaceChildren();
  for (const status of STATUS_COUNTS) {
    const count = state.rows.filter(item => item.status === status).length;
    const button = node('button', `${status} ${count}`, 'chip');
    button.type = 'button';
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

async function load() {
  const [catalog, extras] = await Promise.all([
    json('api/source-statuses'),
    json('measurements/source-status-extras.json'),
  ]);
  state.explanations = extras.reason_explanations || {};
  const reasonFilter = $('reason-filter');
  for (const [value, entry] of Object.entries(state.explanations)) {
    reasonFilter.append(new Option(entry.label, value));
  }
  const stages = extras.stages || {};
  const main = (catalog.sources || []).map(declaration => ({
    id: declaration.id,
    display_name: declaration.display_name,
    name: declaration.name,
    status: declaration.status,
    stage: stages[declaration.id] || declaration.stage || null,
    reasons: declaration.exclusion_claims || [],
    notes: declaration.notes,
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
    $('search').value = '';
    $('status-filter').value = 'all';
    $('reason-filter').value = 'all';
    render();
  });
});
