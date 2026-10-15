/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1});
const percent = value => `${(100 * value).toFixed(1)}%`;
const outcomes = {assigned: 'Capability match', subject_only: 'Subject only', out_of_scope: 'Outside cutoff', insufficient_context: 'Insufficient context'};
const state = {data: null, parent: '', subject: '', nodes: new Map(), request: 0, taskRequest: 0, next: '', taskQuery: null};
let searchTimer;

function parameters() {
  return new URLSearchParams({subject_distance: $('subject-distance').value, capability_distance: $('capability-distance').value,
    count_mode: $('count-mode').value, group_by: $('group-by').value, intended_use: $('use').value, source_id: $('source').value});
}
function labelFor(id, subject) { return state.nodes.get(`${subject}/${id}`)?.name || id; }
function button(text, className, action) { const item = node('button', text, className); item.type = 'button'; item.addEventListener('click', action); return item; }

async function loadCoverage() {
  const request = ++state.request;
  ++state.taskRequest;
  $('task-panel').hidden = true;
  $('task-detail').close();
  const params = parameters();
  params.set('parent_id', state.parent); params.set('subject_id', state.subject); params.set('search', $('search').value.trim());
  $('message').textContent = 'Updating coverage from saved scores…';
  $('heatmap').setAttribute('aria-busy', 'true');
  $('heatmap').inert = true; $('outcomes').inert = true;
  try {
    const data = await json(`api/curriculum?${params}`);
    if (request !== state.request) return;
    state.data = data;
    state.nodes = new Map(data.hierarchy.map(item => [`${item.subject_id}/${item.node_id}`, item]));
    const selected = $('source').value;
    $('source').replaceChildren(node('option', 'All sources'));
    $('source').firstChild.value = '';
    for (const source of data.sources) { const option = node('option', source.name); option.value = source.source_id; $('source').append(option); }
    $('source').value = selected;
    renderCoverage();
    const hash = parameters(); hash.set('parent_id', state.parent); hash.set('subject_id', state.subject);
    if ($('search').value) hash.set('search', $('search').value);
    history.replaceState(null, '', `#${hash}`);
    const processed = data.sources.filter(source => source.alignment).length;
    $('message').textContent = `${data.sample_size ? `Sample: up to ${data.sample_size} tasks per source. ` : ''}${number.format(processed)} of ${number.format(data.sources.length)} inventory sources are in this release. Saved ${new Date(data.created_at).toLocaleString()}.`;
    $('provenance').textContent = JSON.stringify({release_id: data.release_id, curriculum_key: data.curriculum_key, population: data.population, sample_size: data.sample_size, lineage: data.lineage}, null, 2);
  } catch (error) { if (request === state.request) $('message').textContent = `Coverage could not be loaded. ${error.message}`; }
  finally { if (request === state.request) { $('heatmap').removeAttribute('aria-busy'); $('heatmap').inert = false; $('outcomes').inert = false; } }
}

function navigate(item) {
  $('search').value = '';
  state.parent = item?.node_id || ''; state.subject = item?.subject_id || '';
  loadCoverage();
}
function renderBreadcrumbs() {
  const path = [];
  let current = state.nodes.get(`${state.subject}/${state.parent}`);
  while (current) { path.unshift(current); current = state.nodes.get(`${current.subject_id}/${current.parent_id}`); }
  const root = button('All subjects', 'breadcrumb', () => navigate(null));
  if (!path.length) root.setAttribute('aria-current', 'page');
  $('breadcrumbs').replaceChildren(root);
  for (const item of path) {
    const part = button(item.name, 'breadcrumb', () => navigate(item));
    if (item === path.at(-1)) part.setAttribute('aria-current', 'page');
    $('breadcrumbs').append(node('span', '›'), part);
  }
}
function renderCoverage() {
  const data = state.data;
  if (!data) return;
  const total = data.totals.denominator;
  const mapped = data.totals.assigned + data.totals.subject_only;
  $('task-count').textContent = number.format(total);
  $('count-label').textContent = $('count-mode').value === 'content' ? 'UNIQUE PUBLIC INPUTS' : 'TASK INSTANCES';
  $('mapped-count').textContent = total ? percent(mapped / total) : '—';
  $('capability-count').textContent = number.format(data.totals.assigned);
  $('broad-count').textContent = `${number.format(data.totals.subject_only)} with a subject-only match`;
  const sourceCount = data.sources.filter(source => source.alignment && (!$('source').value || source.source_id === $('source').value) && ($('use').value === 'all' || source.alignment.intended_use === $('use').value)).length;
  $('source-count').textContent = `${number.format(sourceCount)} processed sources in this view`;
  const selectedSource = data.sources.find(source => source.source_id === $('source').value)?.alignment;
  $('source-ledger').hidden = !selectedSource || selectedSource.input_rows === null;
  if (selectedSource?.input_rows !== null && selectedSource?.input_rows !== undefined) {
    const rejected = Object.values(selectedSource.rejections).reduce((a, b) => a + b, 0);
    $('source-ledger').textContent = `Complete source conversion: ${number.format(selectedSource.input_rows)} input rows → ${number.format(selectedSource.converted_rows)} converted tasks; ${number.format(rejected)} conversion rejections. This release contains ${number.format(selectedSource.task_count)} task instances from that source.`;
  }
  $('population').textContent = `${data.sample_size ? 'Sampled ' : ''}${data.population} population`;
  renderBreadcrumbs();
  const scope = data.scope_summary;
  $('coverage-scope').hidden = !scope || !!$('search').value;
  if (scope) $('coverage-scope').textContent = `${labelFor(`subject:${state.subject}`, state.subject)} totals: ${number.format(scope.subject_members)} within the subject cutoff · ${number.format(scope.capability_matches)} with a capability match · ${number.format(scope.subject_members - scope.capability_matches)} subject-only. Group and capability cells count capability matches.`;
  const columns = data.columns.filter(column => $('include-empty').checked || column.processed_sources > 0);
  const head = node('tr'); head.append(node('th', $('search').value ? 'Search results' : state.parent ? 'Groups & capabilities' : 'Curriculum subject'));
  for (const column of columns) {
    const th = node('th'); th.scope = 'col';
    const title = button(column.name, 'column-name', () => openTasks(null, column)); title.title = column.name;
    th.append(title, node('span', `${number.format(column.denominator)} tasks · ${column.processed_sources}/${column.sources} sources`, 'column-meta'));
    head.append(th);
  }
  $('matrix-head').replaceChildren(head);
  const cells = new Map(data.cells.map(cell => [`${cell.subject_id}/${cell.node_id}/${cell.column_id}`, cell.count]));
  const maximum = Math.max(1, ...data.cells.map(cell => cell.count));
  $('heat-max').textContent = compact.format(maximum);
  const rows = [];
  const ordered = [...data.nodes].sort((a, b) => {
    const count = area => columns.reduce((sum, column) => sum + (cells.get(`${area.subject_id}/${area.node_id}/${column.column_id}`) || 0), 0);
    return count(b) - count(a) || a.name.localeCompare(b.name);
  });
  for (const area of ordered) {
    const row = node('tr'), name = node('th'); name.scope = 'row';
    const label = node('span'); label.append(node('span', area.name, 'area-label'), node('span', area.kind, 'area-kind'));
    const action = button('', 'area-button', () => area.kind === 'capability' ? openTasks(area, null) : navigate(area));
    action.title = area.description || area.name; action.append(label, node('span', area.kind === 'capability' ? '↗' : '›', 'area-arrow')); name.append(action); row.append(name);
    for (const column of columns) {
      const count = cells.get(`${area.subject_id}/${area.node_id}/${column.column_id}`) || 0;
      const td = node('td'), cell = button(number.format(count), 'heat-cell', () => openTasks(area, column));
      const fraction = column.denominator ? percent(count / column.denominator) : '0.0%';
      cell.title = `${area.name} · ${column.name}: ${number.format(count)} / ${number.format(column.denominator)} (${fraction}). ${column.processed_sources}/${column.sources} sources processed.`;
      cell.setAttribute('aria-label', cell.title);
      if (!column.processed_sources) { cell.classList.add('unprocessed'); cell.textContent = '—'; cell.disabled = true; cell.title = 'No processed tasks in this source group'; }
      else if (!count) cell.classList.add('zero');
      else { const intensity = Math.log1p(count) / Math.log1p(maximum); cell.style.backgroundColor = `hsl(148 28% ${94 - intensity * 62}%)`; cell.style.color = intensity > .62 ? '#fff' : '#244e36'; }
      td.append(cell); row.append(td);
    }
    rows.push(row);
  }
  $('matrix-body').replaceChildren(...rows);
  $('empty').hidden = rows.length > 0;
  $('level-help').textContent = $('search').value ? 'Showing up to 100 matching areas. Open a group or select a cell.' : 'Open a row to explore. Select a count to inspect source task instances.';
  $('matrix-summary').textContent = `Curriculum areas: ${data.nodes.length} · Source ${$('group-by').value === 'source' ? 'columns' : 'families'}: ${columns.length}${$('matrix-body').parentElement.parentElement.scrollHeight > $('matrix-body').parentElement.parentElement.clientHeight ? ' · Scroll within the table for all rows' : ''}`;
  $('outcomes').replaceChildren(...Object.entries(outcomes).map(([value, text]) => {
    const item = button('', 'outcome-button', () => openTasks(null, null, value)); item.append(node('span', text), node('b', number.format(data.totals[value]))); return item;
  }));
}

async function openTasks(area, column, outcome = 'all') {
  state.taskQuery = new URLSearchParams(state.data.filters); state.taskQuery.set('release', state.data.release_id);
  if (area) { state.taskQuery.set('node_id', area.node_id); state.taskQuery.set('subject_id', area.subject_id); }
  if (column) state.taskQuery.set('column', column.column_id);
  state.taskQuery.set('outcome', outcome);
  state.next = ''; $('task-list').replaceChildren(); $('task-panel').hidden = false;
  $('task-heading').textContent = area?.name || (outcome === 'all' ? 'Tasks in this source group' : outcomes[outcome]);
  $('task-scope').textContent = `${column?.name || 'Selected sources'} · ${state.data.population} population · source instances, including repeated public inputs`;
  $('task-panel').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start'});
  await loadTasks();
}
async function loadTasks() {
  const request = ++state.taskRequest;
  const params = new URLSearchParams(state.taskQuery); if (state.next) params.set('cursor', state.next);
  $('task-message').textContent = 'Loading tasks…'; $('more-tasks').hidden = true;
  try {
    const data = await json(`api/curriculum/tasks?${params}`);
    if (request !== state.taskRequest) return;
    for (const task of data.tasks) {
      const item = button('', 'task-row', () => openDetail(task));
      const source = state.data.sources.find(source => source.source_id === task.source_id);
      const identity = node('span', source?.name || task.source_id, 'task-source'); identity.append(node('span', task.task_id, 'task-id'));
      item.append(identity, node('span', task.description || 'Instructions do not establish the requested operation.', 'task-description'), node('span', outcomes[task.outcome], 'task-outcome'));
      $('task-list').append(item);
    }
    state.next = data.next_cursor; $('more-tasks').hidden = !state.next;
    $('task-message').textContent = `${number.format($('task-list').children.length)} task instances shown${state.next ? '; more available.' : '.'}`;
  } catch (error) { if (request === state.taskRequest) $('task-message').textContent = `Tasks could not be loaded. ${error.message}`; }
}
async function openDetail(task) {
  const params = new URLSearchParams({source_id: task.source_id, task_id: task.task_id, release: state.data.release_id});
  $('detail-title').textContent = 'Task details'; $('detail-status').textContent = outcomes[task.outcome];
  $('detail-source').textContent = state.data.sources.find(source => source.source_id === task.source_id)?.name || task.source_id;
  $('detail-task-id').textContent = task.task_id;
  $('detail-description').textContent = 'Loading…'; $('detail-matches').replaceChildren(); $('detail-instructions').textContent = ''; $('preview-note').textContent = ''; $('detail-provenance').textContent = '';
  $('task-detail').showModal();
  $('task-detail').scrollTop = 0;
  try {
    const {task: detail} = await json(`api/curriculum/task?${params}`);
    $('detail-description').textContent = detail.description || detail.reason;
    $('detail-instructions').textContent = detail.task_preview;
    $('preview-note').textContent = detail.preview_truncated ? `Preview: ${number.format(detail.task_preview.length)} of ${number.format(detail.instruction_characters)} instruction characters.` : `${number.format(detail.instruction_characters)} instruction characters. Public task files are not read for this description.`;
    for (const match of detail.matches?.subjects || []) {
      const subjectInside = match.membership_similarity >= 1 - state.data.filters.subject_distance;
      $('detail-matches').append(node('h3', `Primary subject · ${labelFor(`subject:${match.subject_id}`, match.subject_id)}`));
      scoreTable([{name: 'Subject membership', similarity: match.membership_similarity, status: subjectInside ? 'Within subject cutoff' : 'Outside subject cutoff'}, ...match.candidates.map((candidate, index) => ({name: labelFor(candidate.capability_id, candidate.subject_id), similarity: candidate.similarity,
        status: !subjectInside ? 'Outside subject cutoff' : candidate.similarity < 1 - state.data.filters.capability_distance ? 'Outside capability cutoff' : index === 0 ? 'Primary capability match' : 'Alternative within cutoff'}))]);
    }
    if (detail.matches?.capabilities?.length) {
      const alternatives = node('details'); alternatives.append(node('summary', 'Nearest capabilities across all subjects'));
      scoreTable(detail.matches.capabilities.map(candidate => ({name: `${labelFor(`subject:${candidate.subject_id}`, candidate.subject_id)} / ${labelFor(candidate.capability_id, candidate.subject_id)}`, similarity: candidate.similarity,
        status: candidate.similarity >= 1 - state.data.filters.capability_distance ? 'Alternative within capability cutoff' : 'Outside capability cutoff'})), alternatives);
      $('detail-matches').append(alternatives);
    }
    $('detail-provenance').textContent = JSON.stringify({source_id: detail.source_id, task_id: detail.task_id, source_key: detail.source_key, task_hash: detail.task_hash, content_hash: detail.content_hash, intended_use: detail.intended_use, population: detail.population, release_id: state.data.release_id}, null, 2);
  } catch (error) { $('detail-description').textContent = `Task could not be loaded. ${error.message}`; }
}
function scoreTable(scores, target = $('detail-matches')) {
  const table = node('table'), head = node('tr'); for (const text of ['Area', 'Cosine', 'Distance', 'At active cutoffs']) head.append(node('th', text)); const thead = node('thead'); thead.append(head); table.append(thead);
  const body = node('tbody'); for (const score of scores) { const row = node('tr'); row.append(node('td', score.name), node('td', score.similarity.toFixed(3), 'score'), node('td', (1-score.similarity).toFixed(3), 'score'), node('td', score.status, 'score-status')); body.append(row); } table.append(body); target.append(table);
}

$('controls').addEventListener('submit', event => { event.preventDefault(); loadCoverage(); });
for (const id of ['source', 'use', 'count-mode', 'group-by']) $(id).addEventListener('change', loadCoverage);
$('include-empty').addEventListener('change', renderCoverage);
$('search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadCoverage, 300); });
$('more-tasks').addEventListener('click', loadTasks);
$('close-tasks').addEventListener('click', () => { ++state.taskRequest; $('task-panel').hidden = true; });
$('close-detail').addEventListener('click', () => $('task-detail').close());
$('reset-view').addEventListener('click', () => { $('controls').reset(); $('search').value = ''; state.parent = ''; state.subject = ''; loadCoverage(); });
const initial = new URLSearchParams(location.hash.slice(1));
for (const [param, id] of Object.entries({subject_distance: 'subject-distance', capability_distance: 'capability-distance', count_mode: 'count-mode', group_by: 'group-by', intended_use: 'use', search: 'search'})) if (initial.has(param)) $(id).value = initial.get(param);
if (initial.get('source_id')) { const option = node('option', initial.get('source_id')); option.value = initial.get('source_id'); $('source').append(option); $('source').value = option.value; }
state.parent = initial.get('parent_id') || ''; state.subject = initial.get('subject_id') || '';
loadCoverage();
