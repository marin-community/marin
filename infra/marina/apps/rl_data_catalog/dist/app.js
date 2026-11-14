/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat('en-US');
const resultLabels = {GOOD: 'Good', SOME_ISSUES: 'Some issues', BAD: 'Bad', UNKNOWN: 'Unknown', none: 'Not assessed'};
const resultClasses = {GOOD: 'good', SOME_ISSUES: 'some_issues', BAD: 'bad', UNKNOWN: 'unknown', none: 'unassessed'};
const useLabels = {train: 'Training', eval: 'Evaluation', unknown: 'Unknown'};
const oracleValidationLabels = {executed: 'Oracle executed', verifier_executed: 'Verifier executed', skipped: 'Skipped'};
const columns = [
  ['display_name', 'Source'], ['examples', 'Examples'], ['filtration', 'Filtration'], ['use', 'Use'], ['quality', 'Quality'], ['result', 'Result'],
  ['issue_rate', 'Issue rate'], ['contamination', 'Contamination'], ['overlap', 'Overlap'], ['daytona', 'Daytona'], ['backends', 'Backends'], ['grading', 'Grading'], ['area', 'Top curriculum area'], ['updated', 'Last updated'], ['evidence', 'Evidence'], ['difficulty', 'Difficulty'],
];
const qualityOrder = {QUICK: 0, SAMPLE: 1, FULL: 2};
const state = {sources: [], sort: 'display_name', direction: 1, origin: 'all', syncedAt: null};
const day = value => value ? new Date(value).toLocaleDateString('en-US', {month: 'short', day: '2-digit', year: 'numeric', timeZone: 'UTC'}) : '—';

const originLabels = {tasktrove: 'TaskTrove', evalchemy: 'Evalchemy', harbor_hub: 'Harbor Hub', nemotron: 'Nemotron', 'Task Trove': 'TaskTrove'};
function originOf(row) { return originLabels[row.origin] || row.origin; }
function resultKey(row) { return row.result || 'none'; }
function value(row, key) {
  if (key === 'examples') return row.filtration.example_count;
  if (key === 'filtration') return row.filtration.state;
  if (key === 'use') return row.intended_use || 'unknown';
  if (key === 'quality') return qualityOrder[row.quality];
  if (key === 'result') return resultKey(row);
  if (key === 'contamination') return row.contamination_status;
  if (key === 'overlap') return row.contamination_status === 'NOT MEASURED' ? null : row.lexical_contamination?.contaminated_tasks;
  if (key === 'daytona') return row.daytona.state;
  if (key === 'backends') return backendLabels(row).map(item => item.label).join('; ');
  if (key === 'grading') return Object.keys(row.grading_environment?.required_counts || {}).join('; ');
  if (key === 'area') return row.coverage?.top_area?.name;
  if (key === 'updated') return row.updated_at;
  if (key === 'difficulty') return AtlasDifficulty.selected(row.difficulty, $('difficulty-model').value)?.solve_rate;
  return row[key];
}
function shownByDefault(row) { return row.quality !== 'QUICK' && row.result !== 'BAD'; }
function matchesFilters(row) {
  const search = $('search').value.trim().toLowerCase();
  if (state.origin !== 'all' && originOf(row) !== state.origin) return false;
  if ($('use').value !== 'all' && value(row, 'use') !== $('use').value) return false;
  if ($('quality').value !== 'all' && row.quality !== $('quality').value) return false;
  if ($('result').value !== 'all' && resultKey(row) !== $('result').value) return false;
  const text = [row.display_name, row.name, row.dataset_id, row.family, ...(row.tags || [])].join(' ').toLowerCase();
  return !search || text.includes(search);
}
function filtered() {
  return state.sources.filter(row => ($('show-all').checked || shownByDefault(row)) && matchesFilters(row)).sort((a, b) => {
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
  return row.contamination_status + (row.lexical_contamination ? `: ${number.format(row.lexical_contamination.contaminated_tasks)} / ${number.format(row.lexical_contamination.measured_tasks)} overlapping tasks` : '');
}
function contaminationCell(row) {
  const cell = node('td', undefined, 'contamination-cell');
  cell.append(contaminationChip(row.contamination_status));
  cell.title = contaminationText(row) + '. Open Evidence for the measured input, exclusions and current release pin.';
  return cell;
}
function daytonaCell(row) {
  const cell = node('td'), daytona = row.daytona;
  const safe = daytona.state === 'safe';
  const label = daytonaLabel(daytona);
  const chip = node('span', label, `chip quality-${safe ? 'good' : daytona.state === 'unsafe' ? 'bad' : 'unassessed'}`);
  chip.title = daytona.reason || daytona.basis || '';
  cell.append(chip);
  return cell;
}
function renderTabs() {
  const eligible = state.sources.filter(row => $('show-all').checked || shownByDefault(row));
  const origins = ['all', ...[...new Set(state.sources.map(originOf))].sort()];
  $('origin-tabs').replaceChildren(...origins.map(origin => {
    const tab = node('button', origin === 'all' ? 'Origin' : origin, `tab${state.origin === origin ? ' selected' : ''}`);
    tab.type = 'button'; tab.setAttribute('aria-pressed', String(state.origin === origin));
    const tasks = eligible.filter(row => origin === 'all' || originOf(row) === origin);
    tab.append(node('span', `${number.format(taskCounts(tasks).population)} tasks`));
    tab.addEventListener('click', () => { state.origin = origin; update(); });
    return tab;
  }));
}
function render() {
  renderTabs();
  $('headers').replaceChildren(...columns.map(([key, title]) => {
    const th = node('th'); th.scope = 'col';
    if (['examples', 'issue_rate', 'overlap'].includes(key)) th.className = 'num';
    if (['evidence', 'examples'].includes(key) || (key === 'difficulty' && $('difficulty-model').value === 'all')) {
      th.append(node('span', title, 'header-label')); return th;
    }
    const button = node('button', title);
    th.setAttribute('aria-sort', state.sort === key ? state.direction === 1 ? 'ascending' : 'descending' : 'none');
    if (state.sort === key) button.append(node('span', state.direction === 1 ? '↑' : '↓', 'sort-indicator'));
    button.addEventListener('click', () => {
      state.direction = state.sort === key ? -state.direction : ['quality', 'updated', 'difficulty'].includes(key) ? -1 : 1;
      state.sort = key; update();
    });
    th.append(button); return th;
  }));
  const rows = filtered();
  const entries = collapsed(rows);
  const displayed = entries.flatMap(({row, others}) => [row, ...(expandedGroups.has(row.duplicate_group?.group_id) ? others : [])]);
  metrics(displayed, rows);
  $('rows').replaceChildren(...entries.flatMap(({row, others}) => {
    const groupId = row.duplicate_group?.group_id;
    const open = groupId && expandedGroups.has(groupId);
    return [sourceRow(row, others), ...(open ? others.map(other => sourceRow(other, [], row)) : [])];
  }));
  $('empty').hidden = displayed.length > 0;
  const hidden = state.sources.length - state.sources.filter(row => $('show-all').checked || shownByDefault(row)).length;
  const folded = rows.length - displayed.length;
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
      toggle.addEventListener('click', () => { expanded ? expandedGroups.delete(groupId) : expandedGroups.add(groupId); update(); });
      name.prepend(toggle);
    }
    const examples = node('td', row.filtration.example_count == null ? (row.filtration.state === 'pending_oracle' ? 'Pending oracle filtration' : 'Pending') : number.format(row.filtration.example_count), 'num');
    examples.title = row.filtration.reason;
    const filtration = node('td', undefined, 'filtration-cell'); filtration.append(filtrationChip(row.filtration));
    const use = node('td', useLabels[value(row, 'use')], 'use-cell');
    const quality = node('td'); quality.append(node('span', row.quality, `chip level level-${row.quality.toLowerCase()}`));
    if (row.quality_review_status) quality.firstElementChild.title = row.quality_review_status;
    const result = node('td'), key = resultKey(row);
    const provisional = row.result && row.result_provisional;
    const badge = node('span', resultLabels[key] + (provisional ? ' · provisional' : ''), `chip quality-${resultClasses[key]}${provisional ? ' result-provisional' : ''}`);
    result.append(badge);
    const issueRate = node('td', row.issue_rate == null || !row.result ? '—' : `${(100 * row.issue_rate).toFixed(2)}%`, 'num issue-rate-cell');
    const overlap = node('td', value(row, 'overlap') == null ? '—' : number.format(value(row, 'overlap')), 'num overlap-cell');
    overlap.title = row.contamination_status === 'NOT MEASURED' ? 'No applicable current-pin lexical measurement. Historical evidence stays on the source page.' : row.lexical_contamination ? `${number.format(row.lexical_contamination.measured_tasks)} measured input tasks; exclusions and matched evaluations are in Evidence.` : 'No lexical measurement is recorded.';
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
    const contamination = contaminationCell(row);
    const difficulty = node('td', undefined, 'difficulty-cell'); difficulty.append(AtlasDifficulty.comparison(row.difficulty));
    const backends = node('td', undefined, 'backends-cell'); backends.append(backendLink(row.id));
    const grading = node('td'); grading.append(gradingChips(row));
    tr.append(name, examples, filtration, use, quality, result, issueRate, contamination, overlap, daytonaCell(row), backends, grading, area, updated, evidence, difficulty); return tr;
}
function details(row) {
  $('detail-name').textContent = row.display_name; $('detail-notes').textContent = row.notes;
  $('detail-difficulty-bars').replaceChildren(AtlasDifficulty.comparison(row.difficulty));
  const controls = row.curation?.verification;
  const fields = [
    ['Origin', originOf(row)], ['Build path', row.build_path], ['Status', row.status], ['Quality', row.quality], ['Result', resultLabels[resultKey(row)] + (row.result && row.result_provisional ? ' · provisional' : '')],
    ['Content review', row.quality_review_status],
    ...filtrationRows(row), ['Use', useLabels[value(row, 'use')]],
    ['Coverage basis', row.coverage?.basis], ['Coverage gap', row.coverage?.unavailable_reason],
    ['Sampled grader qualification', row.oracle_validation && (oracleValidationLabels[row.oracle_validation] ?? row.oracle_validation)],
    ['Daytona', daytonaLabel(row.daytona)], ['Daytona evidence', row.daytona.reason || row.daytona.basis],
    ['Contamination', contaminationChip(row.contamination_status)], ['Lexical measurement', contaminationText(row)],
    ['Last updated', row.updated_at ? day(row.updated_at) : null], ['Input rows', row.task_count], ['Source ID', row.id],
    ['Dataset revision', row.dataset_revision], ['Pipeline', row.pipeline?.name], ['Curation', row.curation?.status],
    ['Control status', controls?.status], ['Control counts', controls?.counts],
    ['Duplicate group', row.duplicate_group && `${row.duplicate_group.group_id}; canonical ${row.duplicate_group.canonical_source_id}; overlap ${sourceOverlap(row) == null ? "unknown" : `${Math.round(100 * sourceOverlap(row))}%`}`],
    ['Grading boundary required', gradingChips(row)], ['Grading census', row.grading_environment],
    ['Backend fit by role', backendDetails(row)],
    ['Lexical measurement method', row.lexical_contamination && `${row.lexical_contamination.ngram_length}-gram overlap`],
    ['Matched evaluation sets', row.lexical_contamination?.matched_evals],
  ];
  $('detail-fields').replaceChildren(...fields.filter(([, item]) => item !== null && item !== undefined).map(([title, item]) => {
    const field = node('dl', undefined, 'field');
    const value = node('dd');
    value.append(readableRecord(item));
    field.append(node('dt', title), value);
    return field;
  }));
  const links = [];
  const remediated = remediatedReleaseLink(row); if (remediated) links.push(remediated);
  if (row.lexical_contamination?.evidence_url) links.push(receiptLink(row.lexical_contamination.evidence_url, 'Lexical measurement receipt ↗'));
  if (row.oracle_inventory) links.push(receiptLink(row.oracle_inventory.evidence_url, 'Full-population oracle inventory ↗'));
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
function taskCounts(rows) {
  const total = {population: 0, admitted: 0, train: 0, eval: 0, pending: 0, excluded: 0, unmeasured: 0};
  for (const row of rows) {
    const count = row.filtration;
    total.population += count.population_count;
    total.admitted += count.admitted_count ?? 0;
    if (row.intended_use === 'train') total.train += count.admitted_count ?? 0;
    if (row.intended_use === 'eval') total.eval += count.admitted_count ?? 0;
    total.excluded += count.excluded_count ?? 0;
    if (count.state === 'pending_oracle') total.pending += count.population_count;
    else if (count.admitted_count == null) total.unmeasured += count.population_count;
  }
  return total;
}
function metrics(displayed, rows) {
  const samples = rows => {
    const count = taskCounts(rows);
    return `${number.format(count.admitted)} tasks admitted (TRAIN ${number.format(count.train)} · EVAL ${number.format(count.eval)})` +
      ` · ${number.format(count.pending)} tasks pending oracle filtration · ${number.format(count.excluded)} tasks excluded` +
      (count.unmeasured ? ` · ${number.format(count.unmeasured)} tasks with admission unmeasured` : '');
  };
  const qualityCounts = rows => ['QUICK', 'SAMPLE', 'FULL'].map(level => number.format(taskCounts(rows.filter(row => row.quality === level)).population)).join(' / ');
  $('metric-sources').textContent = `${number.format(taskCounts(state.sources).population)} / ${number.format(taskCounts(displayed).population)}`;
  $('metric-examples').textContent = samples(state.sources);
  $('metric-examples-shown').textContent = `Shown ${samples(displayed)}`;
  $('metric-quality').textContent = qualityCounts(state.sources);
  $('metric-quality-shown').textContent = `Shown ${qualityCounts(displayed)}`;
  $('metric-date').textContent = day(state.syncedAt);
  const hidden = state.sources.filter(row => !$('show-all').checked && !shownByDefault(row));
  const filtered = state.sources.filter(row => !hidden.includes(row) && !matchesFilters(row));
  const folded = rows.filter(row => !displayed.includes(row));
  const reasons = [[hidden, 'hidden (QUICK/BAD)'], [filtered, 'filtered'], [folded, 'folded (likely duplicates)']];
  $('task-reconciliation').textContent = reasons.filter(([items]) => items.length).map(([items, reason]) => {
    const count = taskCounts(items);
    return `${number.format(count.population)} tasks ${reason}; ${number.format(count.admitted)} admitted`;
  }).join(' · ') || 'All tasks shown.';
}
async function load() {
  const [data, inventory] = await Promise.all([json('api/sources'), json('measurements/oracle-inventory.json')]);
  state.sources = data.sources.map(source => ({...source, oracle_inventory: boundOracleInventory(source, inventory)}));
  state.syncedAt = data.synced_at;
  $('message').hidden = true;
  $('snapshot-info').textContent = `Last updated ${day(data.synced_at)}`;
  readLocation();
  history.replaceState(null, '', viewUrl());
  render();
}
function viewUrl() {
  const params = new URLSearchParams();
  for (const id of ['search', 'use', 'quality', 'result', 'difficulty-model']) {
    const item = $(id).value.trim();
    if (item && item !== 'all') params.set(id, item);
  }
  if (state.origin !== 'all') params.set('origin', state.origin);
  if ($('show-all').checked) params.set('all', '1');
  if (expandedGroups.size) params.set('expanded', [...expandedGroups].sort().join(','));
  if (state.sort !== 'display_name') params.set('sort', state.sort);
  if (state.direction !== 1) params.set('direction', state.direction);
  return location.pathname + (params.size ? `#${params}` : '');
}
function readLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  $('filters').reset(); $('search').value = params.get('search') || '';
  for (const id of ['use', 'quality', 'result']) if (params.has(id)) $(id).value = params.get(id);
  if (AtlasDifficulty.solvers.includes(params.get('difficulty-model'))) $('difficulty-model').value = params.get('difficulty-model');
  $('show-all').checked = params.get('all') === '1';
  expandedGroups.clear();
  for (const group of (params.get('expanded') || '').split(',').filter(Boolean)) expandedGroups.add(group);
  state.origin = params.get('origin') === 'Task Trove' ? 'TaskTrove' : params.get('origin') || 'all';
  const sortable = columns.filter(([key]) => !['examples', 'evidence'].includes(key)).map(([key]) => key);
  state.sort = sortable.includes(params.get('sort')) ? params.get('sort') : 'display_name';
  state.direction = params.get('direction') === '-1' ? -1 : 1;
}
function update(replace = false) {
  history[replace ? 'replaceState' : 'pushState'](null, '', viewUrl());
  render();
}
for (const id of ['use', 'quality', 'result']) $(id).addEventListener('change', () => update());
$('difficulty-model').addEventListener('change', () => {
  state.sort = $('difficulty-model').value === 'all' ? 'display_name' : 'difficulty';
  state.direction = state.sort === 'difficulty' ? -1 : 1;
  update();
});
$('show-all').addEventListener('change', () => update());
$('filters').addEventListener('submit', event => event.preventDefault());
$('search').addEventListener('input', () => update(true));
$('close-details').addEventListener('click', () => $('details').close());
function resetFilters() {
  $('filters').reset(); $('search').value = ''; state.origin = 'all';
  state.sort = 'display_name'; state.direction = 1; expandedGroups.clear();
}
$('clear-filters').addEventListener('click', () => { resetFilters(); update(); });
$('show-everything').addEventListener('click', () => {
  resetFilters(); $('show-all').checked = true;
  for (const row of state.sources) if (row.duplicate_group) expandedGroups.add(row.duplicate_group.group_id);
  update();
});
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
