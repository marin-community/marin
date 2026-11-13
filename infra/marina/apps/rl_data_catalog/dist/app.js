/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat('en-US');
const resultLabels = {GOOD: 'Good', SOME_ISSUES: 'Some issues', BAD: 'Bad', UNKNOWN: 'Unknown', none: 'Not assessed'};
const resultClasses = {GOOD: 'good', SOME_ISSUES: 'some_issues', BAD: 'bad', UNKNOWN: 'unknown', none: 'unassessed'};
const useLabels = {train: 'Training', eval: 'Evaluation', unknown: 'Unknown'};
const oracleValidationLabels = {executed: 'Oracle executed', verifier_executed: 'Verifier executed', skipped: 'Skipped'};
const columns = [
  ['display_name', 'Source'], ['examples', 'Examples'], ['use', 'Use'], ['quality', 'Quality'], ['result', 'Result'],
  ['contamination', 'Contamination'], ['daytona', 'Daytona'], ['area', 'Top curriculum area'], ['updated', 'Last updated'], ['evidence', 'Evidence'],
];
const qualityOrder = {QUICK: 0, SAMPLE: 1, FULL: 2};
const state = {sources: [], sort: 'examples', direction: -1, origin: 'all', syncedAt: null};
const day = value => value ? new Date(value).toLocaleDateString('en-US', {month: 'short', day: '2-digit', year: 'numeric', timeZone: 'UTC'}) : '—';

const originLabels = {tasktrove: 'TaskTrove', evalchemy: 'Evalchemy', harbor_hub: 'Harbor Hub', nemotron: 'Nemotron', 'Task Trove': 'TaskTrove'};
function originOf(row) { return originLabels[row.origin] || row.origin; }
function resultKey(row) { return row.result || 'none'; }
function value(row, key) {
  if (key === 'examples') return row.within_source_dedup?.unique_count;
  if (key === 'use') return row.intended_use || 'unknown';
  if (key === 'quality') return qualityOrder[row.quality];
  if (key === 'result') return resultKey(row);
  if (key === 'contamination') return contaminationText(row);
  if (key === 'daytona') return row.daytona.state;
  if (key === 'area') return row.coverage?.top_area?.name;
  if (key === 'updated') return row.updated_at;
  return row[key];
}
// The default view keeps sources whose QA executed and did not find them defective.
function shownByDefault(row) { return row.quality !== 'QUICK' && row.result !== 'BAD'; }
function filtered() {
  const search = $('search').value.trim().toLowerCase();
  const showAll = $('show-all').checked;
  return state.sources.filter(row => {
    if (!showAll && !shownByDefault(row)) return false;
    if (state.origin !== 'all' && originOf(row) !== state.origin) return false;
    if ($('use').value !== 'all' && value(row, 'use') !== $('use').value) return false;
    if ($('quality').value !== 'all' && row.quality !== $('quality').value) return false;
    if ($('result').value !== 'all' && resultKey(row) !== $('result').value) return false;
    const text = [row.display_name, row.name, row.dataset_id, row.family, ...(row.tags || [])].join(' ').toLowerCase();
    return !search || text.includes(search);
  }).sort((a, b) => {
    const x = value(a, state.sort), y = value(b, state.sort);
    if (x === null || x === undefined) return y === null || y === undefined ? a.id.localeCompare(b.id) : 1;
    if (y === null || y === undefined) return -1;
    const order = typeof x === 'number' ? x - y : String(x).localeCompare(String(y), undefined, {numeric: true});
    return order * state.direction || a.id.localeCompare(b.id);
  });
}
// The backend survey serves reviewed groups with canonical source IDs and per-member overlap.
function sourceOverlap(row) {
  return row.duplicate_group.members.find(member => member.source_id === row.id).overlap_with_canonical;
}
function collapsed(rows) {
  const groups = new Map(), entries = [];
  for (const row of rows) {
    const group = row.duplicate_group;
    if (!group?.group_id) { entries.push({row, others: []}); continue; }
    if (!groups.has(group.group_id)) { const entry = {row: null, members: []}; groups.set(group.group_id, entry); entries.push(entry); }
    groups.get(group.group_id).members.push(row);
  }
  for (const entry of entries) {
    if (!entry.members) continue;
    const canonicalId = entry.members[0].duplicate_group.canonical_source_id;
    const canonical = entry.members.find(row => row.id === canonicalId) || entry.members[0];
    entry.row = canonical;
    entry.others = entry.members.filter(row => row !== canonical).sort((a, b) => (sourceOverlap(b) ?? 0) - (sourceOverlap(a) ?? 0));
    delete entry.members;
  }
  return entries.sort((a, b) => rows.indexOf(a.row) - rows.indexOf(b.row));
}
const expandedGroups = new Set();

function curriculumLink(row, area) {
  const params = new URLSearchParams({source_id: row.id});
  if (area) { params.set('subject_id', area.subject_id); params.set('parent_id', area.node_id); }
  return `curriculum.html#${params}`;
}
function evidenceLink(row) { return `source.html#${new URLSearchParams({source_id: row.id})}`; }
function contaminationText(row) {
  const evidence = row.lexical_contamination;
  if (!evidence) return 'Pending measurement';
  const flagged = `${number.format(evidence.contaminated_tasks)} / ${number.format(evidence.measured_tasks)} flagged`;
  const receipt = evidence.receipt;
  return receipt ? `${receipt.status}: ${flagged}` : `Measured: ${flagged}`;
}
function daytonaCell(row) {
  const cell = node('td'), daytona = row.daytona;
  const safe = daytona.state === 'safe';
  const label = daytonaLabel(daytona);
  const chip = node('span', label, `chip quality-${safe ? 'good' : daytona.state === 'unsafe' ? 'bad' : 'unassessed'}`);
  chip.title = daytona.reason || daytona.basis || '';
  cell.append(chip);
  if (safe) cell.append(node('span', `${number.format(daytona.snapshots)} snapshots`, 'source-subtitle'));
  return cell;
}
function renderTabs() {
  const eligible = state.sources.filter(row => $('show-all').checked || shownByDefault(row));
  const counts = {all: collapsed(eligible).length};
  const origins = ['all', ...[...new Set(state.sources.map(originOf))].sort()];
  for (const origin of origins.slice(1)) counts[origin] = collapsed(eligible.filter(row => originOf(row) === origin)).length;
  $('origin-tabs').replaceChildren(...origins.map(origin => {
    const tab = node('button', origin === 'all' ? 'Origin' : origin, `tab${state.origin === origin ? ' selected' : ''}`);
    tab.type = 'button'; tab.setAttribute('aria-pressed', String(state.origin === origin));
    tab.append(node('span', number.format(counts[origin] || 0)));
    tab.addEventListener('click', () => { state.origin = origin; update(); });
    return tab;
  }));
}
function render() {
  renderTabs();
  $('headers').replaceChildren(...columns.map(([key, title]) => {
    const th = node('th'); th.scope = 'col';
    if (key === 'examples') th.className = 'num';
    if (key === 'evidence') { th.append(node('span', title, 'header-label')); return th; }
    const button = node('button', title);
    th.setAttribute('aria-sort', state.sort === key ? state.direction === 1 ? 'ascending' : 'descending' : 'none');
    if (state.sort === key) button.append(node('span', state.direction === 1 ? '↑' : '↓', 'sort-indicator'));
    button.addEventListener('click', () => {
      state.direction = state.sort === key ? -state.direction : ['examples', 'quality', 'updated'].includes(key) ? -1 : 1;
      state.sort = key; update();
    });
    th.append(button); return th;
  }));
  const rows = filtered();
  const entries = collapsed(rows);
  const displayed = entries.flatMap(({row, others}) => [row, ...(expandedGroups.has(row.duplicate_group?.group_id) ? others : [])]);
  metrics(displayed);
  $('rows').replaceChildren(...entries.flatMap(({row, others}) => {
    const groupId = row.duplicate_group?.group_id;
    const open = groupId && expandedGroups.has(groupId);
    return [sourceRow(row, others), ...(open ? others.map(other => sourceRow(other, [], row)) : [])];
  }));
  $('empty').hidden = displayed.length > 0;
  const hidden = state.sources.length - state.sources.filter(row => $('show-all').checked || shownByDefault(row)).length;
  const folded = rows.length - entries.length;
  $('visible-count').textContent = `${number.format(displayed.length)} displayed sources` +
    (folded ? ` · ${number.format(folded)} likely duplicates folded` : '') + (hidden ? ` · ${number.format(hidden)} QUICK or BAD hidden` : '');
}
function sourceRow(row, others, canonical) {
    const tr = node('tr', undefined, canonical ? 'duplicate-member' : ''); tr.dataset.sourceId = row.id;
    const name = node('td'), open = node('button', row.display_name, 'source-name');
    open.title = row.display_name; open.addEventListener('click', () => details(row));
    name.append(open, node('span', [originOf(row), row.build_path].filter(Boolean).join(' · '), 'source-subtitle'));
    const datasets = datasetLinks(row);
    if (datasets) name.append(datasets);
    if (canonical) {
      const reference = state.sources.find(item => item.id === row.duplicate_group.canonical_source_id);
      name.append(node('span', `${sourceOverlap(row) == null ? "Unmeasured overlap" : `${Math.round(100 * sourceOverlap(row))}% overlap · ${sourceOverlap(row) === 1 ? "duplicate" : "partial overlap"}`} with ${reference.display_name}`, 'source-subtitle duplicate-note'));
    } else if (others.length) {
      const groupId = row.duplicate_group.group_id, expanded = expandedGroups.has(groupId);
      const toggle = node('button', expanded ? '▾' : '▸', 'text-button duplicate-toggle');
      toggle.title = `${others.length} likely duplicate member${others.length === 1 ? '' : 's'}`;
      toggle.setAttribute('aria-label', `${expanded ? 'Collapse' : 'Expand'} likely duplicates for ${row.display_name}`);
      toggle.setAttribute('aria-expanded', String(expanded));
      toggle.addEventListener('click', () => { expanded ? expandedGroups.delete(groupId) : expandedGroups.add(groupId); render(); });
      name.prepend(toggle);
    }
    const examples = node('td', row.within_source_dedup?.unique_count != null ? number.format(row.within_source_dedup.unique_count) : 'Unknown', 'num');
    examples.title = row.coverage?.basis || 'No coverage measurement in this catalog';
    const use = node('td', useLabels[value(row, 'use')], 'use-cell');
    const quality = node('td'); quality.append(node('span', row.quality, `chip level level-${row.quality.toLowerCase()}`));
    const result = node('td'), key = resultKey(row);
    const badge = node('span', resultLabels[key], `chip quality-${resultClasses[key]}`);
    result.append(badge);
    if (row.quality_result?.issue_rate != null) result.append(node('span', `${(100 * row.quality_result.issue_rate).toFixed(2)}% issues`, 'source-subtitle'));
    if (row.quality === 'FULL') quality.append(node('span', 'Content QA complete', 'source-subtitle'));
    else quality.append(node('span', row.content_progress ? `Provisional: ≥${number.format(row.content_progress.valid_reviews_lower_bound)} / ${number.format(row.content_progress.retained_count)} reviewed` : 'Content QA pending / incomplete', 'source-subtitle'));
    if (row.quality_result && row.quality !== 'FULL') result.append(node('span', `${number.format(row.quality_result.reviewed)} reviewed · provisional`, 'source-subtitle'));
    const area = node('td');
    if (row.coverage?.top_area) {
      const anchor = node('a', row.coverage.top_area.name, 'area-link');
      anchor.href = curriculumLink(row, row.coverage.top_area);
      anchor.title = `${number.format(row.coverage.top_area.instances)} placed examples in this subject`;
      area.append(anchor);
    } else {
      const gap = node('span', 'Unknown', 'muted'); gap.title = row.coverage?.unavailable_reason || 'No curriculum release covers this source'; area.append(gap);
    }
    const updated = node('td', day(row.updated_at), 'date-cell');
    updated.title = row.updated_at ? 'Load time of the selected catalog snapshot (UTC)' : 'Catalog load time unavailable';
    const evidence = node('td'), link = node('a', 'Evidence →', 'evidence-link'); link.href = evidenceLink(row); evidence.append(link);
    const contamination = node('td', contaminationText(row));
    contamination.title = row.lexical_contamination ? 'Matched evaluations and the decontamination receipt are in Evidence.' : 'No lexical measurement is recorded.';
    tr.append(name, examples, use, quality, result, contamination, daytonaCell(row), area, updated, evidence); return tr;
}
function details(row) {
  $('detail-name').textContent = row.display_name; $('detail-notes').textContent = row.notes;
  const controls = row.curation?.verification;
  const fields = [
    ['Origin', originOf(row)], ['Build path', row.build_path], ['Status', row.status], ['Quality', row.quality], ['Result', resultLabels[resultKey(row)]],
    ['Unique examples', row.within_source_dedup?.unique_count], ['Raw converted tasks', row.within_source_dedup?.raw_count], ['Within-source duplicates dropped', row.within_source_dedup?.within_source_duplicates], ['Use', useLabels[value(row, 'use')]],
    ['Coverage basis', row.coverage?.basis], ['Coverage gap', row.coverage?.unavailable_reason],
    ['Oracle validation', row.oracle_validation && (oracleValidationLabels[row.oracle_validation] ?? row.oracle_validation)],
    ['Daytona', daytonaLabel(row.daytona)], ['Daytona evidence', row.daytona.reason || row.daytona.basis],
    ['Contamination', row.lexical_contamination],
    ['Last updated', row.updated_at ? day(row.updated_at) : null], ['Input rows', row.task_count], ['Source ID', row.id],
    ['Dataset revision', row.dataset_revision], ['Pipeline', row.pipeline?.name], ['Curation', row.curation?.status],
    ['Admission counts', row.curation?.admission_counts], ['Control status', controls?.status], ['Control counts', controls?.counts],
    ['Duplicate group', row.duplicate_group && `${row.duplicate_group.group_id}; canonical ${row.duplicate_group.canonical_source_id}; overlap ${sourceOverlap(row) == null ? "unknown" : `${Math.round(100 * sourceOverlap(row))}%`}`],
    ['Backend capabilities', row.backend_capabilities],
  ];
  $('detail-fields').replaceChildren(...fields.filter(([, item]) => item !== null && item !== undefined).map(([title, item]) => {
    const field = node('dl', undefined, 'field');
    field.append(node('dt', title), node('dd', typeof item === 'object' ? JSON.stringify(item) : String(item)));
    return field;
  }));
  const links = [];
  const evidence = node('a', 'All quality evidence'); evidence.href = evidenceLink(row); links.push(evidence);
  if (row.coverage?.top_area) { const anchor = node('a', 'Curriculum coverage'); anchor.href = curriculumLink(row); links.push(anchor); }
  if (row.assessment.status === 'complete') { const anchor = node('a', 'Sampled review'); anchor.href = `assessment.html?source=${encodeURIComponent(row.id)}`; links.push(anchor); }
  const datasets = datasetLinks(row);
  if (datasets) links.push(datasets);
  if (row.within_source_dedup?.drop_manifest_uri) {
    const anchor = node('a', 'Duplicate exclusions in Filer ↗');
    anchor.href = `https://filer.marina.oa.dev/#${encodeURIComponent(row.within_source_dedup.drop_manifest_uri)}`;
    links.push(anchor);
  }
  $('detail-links').replaceChildren(...links); $('details').showModal();
}
function metrics(displayed) {
  const total = collapsed(state.sources).map(entry => entry.row);
  const samples = rows => rows.reduce((sum, row) => sum + (row.within_source_dedup?.unique_count ?? 0), 0);
  $('metric-sources').textContent = `${number.format(total.length)} / ${number.format(displayed.length)}`;
  $('metric-examples').textContent = `${number.format(samples(total))} / ${number.format(samples(displayed))}`;
  $('metric-quality').textContent = ['QUICK', 'SAMPLE', 'FULL'].map(level => number.format(displayed.filter(row => row.quality === level).length)).join(' / ');
  $('metric-date').textContent = day(state.syncedAt);
}
async function load() {
  const [data, progress] = await Promise.all([json('api/sources'), json('measurements/content-progress.json')]);
  state.sources = data.sources.map(row => ({...row, content_progress: boundProgress(row, progress)}));
  state.syncedAt = data.synced_at;
  $('message').hidden = true;
  $('snapshot-info').textContent = `Last updated ${day(data.synced_at)}`;
  readLocation();
  history.replaceState(null, '', viewUrl());
  render();
}
function viewUrl() {
  const params = new URLSearchParams();
  for (const id of ['search', 'use', 'quality', 'result']) {
    const item = $(id).value.trim();
    if (item && item !== 'all') params.set(id, item);
  }
  if (state.origin !== 'all') params.set('origin', state.origin);
  if ($('show-all').checked) params.set('all', '1');
  if (state.sort !== 'examples') params.set('sort', state.sort);
  if (state.direction !== -1) params.set('direction', state.direction);
  return location.pathname + (params.size ? `#${params}` : '');
}
function readLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  $('filters').reset(); $('search').value = params.get('search') || '';
  for (const id of ['use', 'quality', 'result']) if (params.has(id)) $(id).value = params.get(id);
  $('show-all').checked = params.get('all') === '1';
  state.origin = params.get('origin') === 'Task Trove' ? 'TaskTrove' : params.get('origin') || 'all';
  state.sort = params.get('sort') || 'examples';
  state.direction = params.get('direction') === '1' ? 1 : -1;
}
function update(replace = false) {
  history[replace ? 'replaceState' : 'pushState'](null, '', viewUrl());
  render();
}
for (const id of ['use', 'quality', 'result']) $(id).addEventListener('change', () => update());
$('show-all').addEventListener('change', () => update());
$('filters').addEventListener('submit', event => event.preventDefault());
$('search').addEventListener('input', () => update(true));
$('close-details').addEventListener('click', () => $('details').close());
$('clear-filters').addEventListener('click', () => { $('filters').reset(); $('search').value = ''; state.origin = 'all'; update(); });
window.addEventListener('popstate', () => { readLocation(); render(); });
$('export').addEventListener('click', () => {
  const quote = item => `"${String(item ?? '').replaceAll('"', '""')}"`;
  const fields = columns.filter(([key]) => key !== 'evidence');
  const cell = (row, key) => key === 'quality' ? row.quality : key === 'daytona' ? row.daytona.state : value(row, key);
  const csv = [fields.map(([, title]) => quote(title)).join(','), ...filtered().map(row => fields.map(([key]) => quote(cell(row, key))).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const anchor = node('a'); anchor.href = url; anchor.download = 'rl-data-atlas.csv'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
load().catch(error => { $('message').hidden = false; $('message').textContent = error.message; $('message').classList.add('error'); });
