/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1});
const percent = value => `${(100 * value).toFixed(1)}%`;
const outcomes = {assigned: 'Capability match', subject_only: 'Subject only', out_of_scope: 'Outside cutoff', insufficient_context: 'Insufficient context', skipped: 'Embedding skipped'};
const state = { data: null, parent: '', subject: '', nodes: new Map(), request: 0, taskRequest: 0, next: '', taskQuery: null};

function parameters() {
  return new URLSearchParams({intended_use: $('use').value, source_id: $('source').value});
}
function labelFor(id, subject) { return state.nodes.get(`${subject}/${id}`)?.name || id; }
function button(text, className, action) { const item = node('button', text, className); item.type = 'button'; item.addEventListener('click', action); return item; }

async function loadCoverage() {
  const request = ++state.request;
  ++state.taskRequest;
  $('task-panel').hidden = true;
  $('task-detail').close();
  const params = parameters();
  params.set('subject_id', state.subject);
  $('message').textContent = 'Loading curriculum…';
  $('curriculum-map').setAttribute('aria-busy', 'true'); $('curriculum-map').inert = true; $('outcomes').inert = true;
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
    $('domain').replaceChildren(node('option', 'All domains'));
    $('domain').firstChild.value = '';
    for (const area of data.hierarchy.filter(area => area.kind === 'subject')) {
      const option = node('option', area.name); option.value = area.subject_id; $('domain').append(option);
    }
    $('domain').value = state.subject;
    renderCoverage();
    const processed = data.sources.filter(source => source.alignment).length;
    $('message').textContent = `${number.format(processed)} of ${number.format(data.sources.length)} sources included · ${new Date(data.created_at).toLocaleDateString()}`;
    $('provenance').textContent = JSON.stringify({release_id: data.release_id, curriculum_key: data.curriculum_key, population: data.population, sample_size: data.sample_size, lineage: data.lineage}, null, 2);
    const area = state.nodes.get(`${state.subject}/${state.parent}`);
    if (area) await showTasks(area);
  } catch (error) { if (request === state.request) $('message').textContent = `Coverage could not be loaded. ${error.message}`; }
  finally { if (request === state.request) { $('curriculum-map').removeAttribute('aria-busy'); $('curriculum-map').inert = false; $('outcomes').inert = false; } }
}

function navigate(item) {
  $('search').value = '';
  state.parent = item?.node_id || ''; state.subject = item?.subject_id || '';
  history.pushState(null, '', viewUrl());
  loadCoverage();
}
function renderBreadcrumbs() {
  const path = [];
  let current = state.nodes.get(`${state.subject}/${state.parent}`);
  while (current) { path.unshift(current); current = state.nodes.get(`${current.subject_id}/${current.parent_id}`); }
  const root = button('All domains', 'breadcrumb', () => navigate(null));
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
  $('sample-note').textContent = `Examples are sampled up to ${data.examples.source_sample_size} per source and ${data.examples.area_sample_size} per curriculum area.`;
  $('mapped-count').textContent = total ? percent(mapped / total) : '—';
  $('capability-count').textContent = number.format(data.totals.assigned);
  $('broad-count').textContent = `${number.format(data.totals.subject_only)} have a subject-only match`;
  const sourceCount = data.sources.filter(source => source.alignment && (!$('source').value || source.source_id === $('source').value) && ($('use').value === 'all' || source.alignment.intended_use === $('use').value)).length;
  $('source-count').textContent = `${number.format(sourceCount)} processed sources`;
  const selectedSource = data.sources.find(source => source.source_id === $('source').value)?.alignment;
  $('source-ledger').hidden = !selectedSource || selectedSource.input_rows === null;
  if (selectedSource?.input_rows !== null && selectedSource?.input_rows !== undefined) {
    const rejected = Object.values(selectedSource.rejections).reduce((a, b) => a + b, 0);
    $('source-ledger').textContent = `Complete source conversion: ${number.format(selectedSource.input_rows)} input rows → ${number.format(selectedSource.converted_rows)} converted tasks; ${number.format(rejected)} conversion rejections. This release contains ${number.format(selectedSource.task_count)} task instances from that source.`;
  }
  renderBreadcrumbs();
  const scope = data.scope_summary;
  $('coverage-scope').hidden = !scope || !!$('search').value;
  if (scope) $('coverage-scope').textContent = `${number.format(scope.subject_members)} examples mapped to ${labelFor(`subject:${state.subject}`, state.subject)} · ${number.format(scope.capability_matches)} with a skill match.`;
  renderMap();
  $('outcomes').replaceChildren(...Object.entries(outcomes).map(([value, text]) => {
    const item = button('', 'outcome-button', () => openTasks(null, value)); item.append(node('span', text), node('b', number.format(data.totals[value]))); return item;
  }));
  history.replaceState(null, '', viewUrl());
}

function shade(element, count, maximum) {
  if (!count) { element.classList.add('zero'); return; }
  const intensity = Math.log1p(count) / Math.log1p(Math.max(1, maximum));
  element.style.backgroundColor = `hsl(148 28% ${94 - intensity * 62}%)`;
  element.style.color = intensity > .87 ? '#fff' : '#000';
}

function renderMap() {
  const data = state.data;
  const counts = new Map(data.area_counts.map(area => [`${area.subject_id}/${area.node_id}`, area.count]));
  const children = new Map();
  const key = area => `${area.subject_id}/${area.node_id}`;
  for (const area of data.hierarchy) {
    const parent = `${area.subject_id}/${area.parent_id}`;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(area);
  }
  const query = $('search').value.trim().toLowerCase();
  const matches = new Map();
  function matching(area) {
    if (!matches.has(key(area))) matches.set(key(area), !query || `${area.name} ${area.description}`.toLowerCase().includes(query) || (children.get(key(area)) || []).some(matching));
    return matches.get(key(area));
  }
  const focused = state.nodes.get(`${state.subject}/${state.parent}`);
  const roots = focused ? [focused] : data.hierarchy.filter(area => area.kind === 'subject');
  const showSkills = !!focused;
  const maximum = Math.max(1, ...counts.values());
  $('heat-max').textContent = compact.format(maximum);

  function areaRow(area) {
    const count = counts.get(key(area)) || 0;
    const row = node('div', '', 'map-area');
    const name = button(area.name, 'map-area-name', () => area.kind === 'capability' ? openTasks(area) : navigate(area));
    name.title = area.description || area.name;
    const amount = button(number.format(count), 'map-area-count', () => openTasks(area));
    amount.setAttribute('aria-label', `${area.name}: ${number.format(count)} tasks. View sampled tasks.`);
    amount.title = `${number.format(count)} of ${number.format(data.totals.denominator)} task instances`;
    row.append(name, amount); shade(row, count, maximum);
    return row;
  }
  function branch(area, includeAll = false) {
    if (query && !includeAll && !matching(area)) return null;
    const section = node('section', '', `map-branch ${area.kind}`);
    section.append(areaRow(area));
    const descendants = children.get(key(area)) || [];
    if ((showSkills || query) && descendants.length) {
      const nested = node('div', '', 'map-children');
      const includeChildren = includeAll || (query && `${area.name} ${area.description}`.toLowerCase().includes(query));
      for (const child of descendants) { const item = branch(child, includeChildren); if (item) nested.append(item); }
      section.append(nested);
    }
    return section;
  }
  const cards = [];
  let topics = 0;
  for (const area of roots.filter(matching)) {
    const card = node('article', '', 'domain-card');
    card.dataset.subject = area.subject_id;
    const header = node('header', '', 'domain-header');
    const title = node('h2'); title.append(button(area.name, 'domain-title', () => navigate(area)));
    const count = counts.get(key(area)) || 0;
    const amount = button(number.format(count), 'domain-count', () => openTasks(area));
    amount.setAttribute('aria-label', `${area.name}: ${number.format(count)} tasks. View sampled tasks.`);
    header.append(title, amount);
    const descendants = children.get(key(area)) || [];
    const body = node('div', '', 'map-topics');
    const includeAll = !!query && `${area.name} ${area.description}`.toLowerCase().includes(query);
    for (const child of descendants) { const item = branch(child, includeAll); if (item) { body.append(item); topics++; } }
    card.append(header, node('p', `${descendants.length} ${area.kind === 'subject' ? 'topics' : 'skills'}`, 'domain-subtitle'));
    if (focused) card.append(button('Read sampled examples', 'domain-examples text-button', () => openTasks(area)));
    card.append(body);
    cards.push(card);
  }
  $('curriculum-map').classList.toggle('focused', !!focused);
  $('curriculum-map').replaceChildren(...cards);
  if (!cards.length) $('curriculum-map').append(node('p', 'No matching curriculum areas.', 'empty'));
  $('matrix-summary').textContent = focused ? `${focused.name} · ${topics} areas · all skills shown` : `${cards.length} domains · ${topics} topics`;
}

function openTasks(area, outcome = 'all') {
  const loading = showTasks(area, outcome);
  $('task-panel').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start'});
  return loading;
}

async function showTasks(area, outcome = 'all') {
  state.taskQuery = new URLSearchParams(state.data.filters); state.taskQuery.set('release', state.data.release_id);
  if (area) { state.taskQuery.set('node_id', area.node_id); state.taskQuery.set('subject_id', area.subject_id); }
  state.taskQuery.set('outcome', outcome);
  state.next = ''; $('task-list').replaceChildren(); $('task-panel').hidden = false;
  $('task-heading').textContent = area?.name || (outcome === 'all' ? 'Sampled examples' : outcomes[outcome]);
  $('task-scope').textContent = `${$('source').selectedOptions[0].textContent} · sampled examples`;
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
      item.append(identity, node('span', task.task_preview || (task.outcome === 'skipped' ? 'Embedding skipped; open details for the failure.' : 'Instructions do not establish the requested operation.'), 'task-description'), node('span', outcomes[task.outcome], 'task-outcome'));
      $('task-list').append(item);
    }
    state.next = data.next_cursor; $('more-tasks').hidden = !state.next;
    $('task-message').textContent = $('task-list').children.length ? `${number.format($('task-list').children.length)} sampled examples${state.next ? ' · more available' : ''}` : 'No task examples are available in this area at the release cutoffs.';
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
    $('preview-note').textContent = detail.preview_truncated ? `Preview: ${number.format(detail.task_preview.length)} of ${number.format(detail.instruction_characters)} instruction characters.` : `${number.format(detail.instruction_characters)} instruction characters.`;
    for (const match of detail.matches?.subjects || []) {
      const subjectInside = match.membership_similarity >= 1 - state.data.thresholds.subject_distance;
      $('detail-matches').append(node('h3', `Primary subject · ${labelFor(`subject:${match.subject_id}`, match.subject_id)}`));
      scoreTable([{name: 'Subject membership', similarity: match.membership_similarity, within: subjectInside, status: subjectInside ? 'Within subject cutoff' : 'Outside subject cutoff'}, ...match.candidates.map((candidate, index) => ({name: labelFor(candidate.capability_id, candidate.subject_id), similarity: candidate.similarity, within: subjectInside && candidate.similarity >= 1 - state.data.thresholds.capability_distance,
        status: !subjectInside ? 'Outside subject cutoff' : candidate.similarity < 1 - state.data.thresholds.capability_distance ? 'Outside capability cutoff' : index === 0 ? 'Primary capability match' : 'Alternative within cutoff'}))]);
    }
    if (detail.matches?.capabilities?.length) {
      const alternatives = node('details'); alternatives.append(node('summary', 'Nearest capabilities across all subjects'));
      scoreTable(detail.matches.capabilities.map(candidate => ({name: `${labelFor(`subject:${candidate.subject_id}`, candidate.subject_id)} / ${labelFor(candidate.capability_id, candidate.subject_id)}`, similarity: candidate.similarity, within: candidate.similarity >= 1 - state.data.thresholds.capability_distance,
        status: candidate.similarity >= 1 - state.data.thresholds.capability_distance ? 'Alternative within capability cutoff' : 'Outside capability cutoff'})), alternatives);
      $('detail-matches').append(alternatives);
    }
    $('detail-provenance').textContent = JSON.stringify({source_id: detail.source_id, task_id: detail.task_id, source_key: detail.source_key, task_hash: detail.task_hash, content_hash: detail.content_hash, intended_use: detail.intended_use, population: detail.population, release_id: state.data.release_id}, null, 2);
  } catch (error) { $('detail-description').textContent = `Task could not be loaded. ${error.message}`; }
}
function scoreTable(scores, target = $('detail-matches')) {
  const table = node('table'), head = node('tr'); for (const text of ['Area', 'Cosine', 'Distance', 'At release cutoffs']) head.append(node('th', text)); const thead = node('thead'); thead.append(head); table.append(thead);
  const body = node('tbody'); for (const score of scores) { const row = node('tr'); row.append(node('td', score.name), node('td', score.similarity.toFixed(3), 'score'), node('td', (1-score.similarity).toFixed(3), 'score'), node('td', score.status, `score-status${score.within ? '' : ' outside'}`)); body.append(row); } table.append(body); target.append(table);
}

function viewUrl() {
  const hash = new URLSearchParams();
  if ($('use').value !== 'all') hash.set('intended_use', $('use').value);
  if ($('source').value) hash.set('source_id', $('source').value);
  if (state.parent) { hash.set('parent_id', state.parent); hash.set('subject_id', state.subject); }
  if ($('search').value) hash.set('search', $('search').value);
  return location.pathname + (hash.size ? `#${hash}` : '');
}
function readLocation() {
  const initial = new URLSearchParams(location.hash.slice(1));
  $('use').value = initial.get('intended_use') || 'all';
  $('search').value = initial.get('search') || '';
  const source = initial.get('source_id') || '';
  $('source').value = source;
  if (source && $('source').value !== source) {
    const option = node('option', source); option.value = source; $('source').append(option); $('source').value = source;
  }
  state.parent = initial.get('parent_id') || ''; state.subject = initial.get('subject_id') || '';
}
$('controls').addEventListener('submit', event => event.preventDefault());
for (const id of ['source', 'use']) $(id).addEventListener('change', () => {
  history.pushState(null, '', viewUrl()); loadCoverage();
});
$('domain').addEventListener('change', () => navigate($('domain').value ? state.nodes.get(`${$('domain').value}/subject:${$('domain').value}`) : null));
$('search').addEventListener('input', () => {
  ++state.taskRequest; $('task-panel').hidden = true; renderCoverage();
});
$('more-tasks').addEventListener('click', loadTasks);
$('close-tasks').addEventListener('click', () => { ++state.taskRequest; $('task-panel').hidden = true; });
$('close-detail').addEventListener('click', () => $('task-detail').close());
$('reset-view').addEventListener('click', () => {
  $('controls').reset(); state.parent = ''; state.subject = '';
  history.pushState(null, '', viewUrl()); loadCoverage();
});
window.addEventListener('popstate', () => { readLocation(); loadCoverage(); });
readLocation();
loadCoverage();
