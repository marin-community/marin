/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1});
const percent = value => `${(100 * value).toFixed(1)}%`;
// Use labels fill from the served canonical taxonomy (/api/taxonomy) before the
// first render; the page defines no Use label of its own.
let useLabels = {};
const outcomes = {assigned: 'Capability match', subject_only: 'Subject only', out_of_scope: 'Outside cutoff', insufficient_context: 'Insufficient context', skipped: 'Embedding skipped'};
const state = { data: null, parent: '', subject: '', nodes: new Map(), request: 0, taskRequest: 0, next: '', taskQuery: null, raw: null, level: 0, taskSelection: null, pools: new Map(), breakdowns: new Map()};

function parameters() {
  return new URLSearchParams({intended_use: $('use').value, source_id: $('source').value, release: $('release').value});
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
  let selectedArea = null;
  try {
    const data = await json(`api/curriculum?${params}`);
    if (request !== state.request) return;
    state.raw = data;
    $('confidence').replaceChildren();
    const levels = data.confidence?.levels || [{label: 'Release cutoffs', subject_min_similarity: 1-data.thresholds.subject_distance, capability_min_similarity: 1-data.thresholds.capability_distance}];
    levels.forEach((level, index) => {
      const option = node('option', `${level.label} · subject ≥${level.subject_min_similarity.toFixed(2)} / capability ≥${level.capability_min_similarity.toFixed(2)}`);
      option.value = index; $('confidence').append(option);
    });
    state.level = Math.min(state.level, levels.length-1); $('confidence').value = state.level;
    applyConfidence();
    state.nodes = new Map(data.hierarchy.map(item => [`${item.subject_id}/${item.node_id}`, item]));
    const selected = $('source').value;
    $('source').replaceChildren(node('option', 'All sources'));
    $('source').firstChild.value = '';
    for (const source of data.sources) { const option = node('option', source.name); option.value = source.source_id; $('source').append(option); }
    if (selected && !data.sources.some(source => source.source_id === selected)) {
      const option = node('option', `Source not covered: ${selected}`);
      option.value = selected; $('source').append(option);
    }
    $('source').value = selected;
    $('domain').replaceChildren(node('option', 'All domains'));
    $('domain').firstChild.value = '';
    for (const area of data.hierarchy.filter(area => area.kind === 'subject')) {
      const option = node('option', area.name); option.value = area.subject_id; $('domain').append(option);
    }
    $('domain').value = state.subject;
    renderCoverage();
    $('message').textContent = `Measured ${new Date(data.created_at).toLocaleDateString()}`;
    $('provenance').textContent = JSON.stringify({release_id: data.release_id, curriculum_key: data.curriculum_key, population: data.population, sample_size: data.sample_size, lineage: data.lineage, role_projection: data.role_projection}, null, 2);
    $('role-note').hidden = !data.role_projection;
    if (data.role_projection) {
      const link = node('a', 'Role-correction receipt ↗');
      link.href = `https://filer.marina.oa.dev/#${encodeURIComponent(data.role_projection.uri)}`;
      $('role-note').replaceChildren('Source Use includes an approved metadata correction. Sampled evidence preserves its historical role. ', link);
    }
    selectedArea = state.nodes.get(`${state.subject}/${state.parent}`);
  } catch (error) { if (request === state.request) $('message').textContent = `Coverage could not be loaded. ${error.message}`; }
  finally { if (request === state.request) { $('curriculum-map').removeAttribute('aria-busy'); $('curriculum-map').inert = false; $('outcomes').inert = false; } }
  if (request === state.request && selectedArea) await showTasks(selectedArea);
}

function applyConfidence() {
  const raw = state.raw;
  if (!raw.confidence) { state.data = raw; return; }
  const level = raw.confidence.levels[state.level], totals = {...raw.totals}, area_counts = [];
  let subject_members = 0, capability_matches = 0;
  for (const row of raw.confidence.statistics) {
    const count = row.instances[state.level];
    if (row.metric === 'placement') area_counts.push({subject_id: row.subject_id, node_id: row.node_id, count});
    if (!row.subject_id && !row.node_id) totals[row.metric] = count;
    if (row.subject_id === state.subject && row.node_id === `subject:${state.subject}`) {
      if (row.metric === 'placement') subject_members = count;
      if (row.metric === 'capability') capability_matches = count;
    }
  }
  state.data = {...raw, totals, area_counts, thresholds: {subject_distance: 1-level.subject_min_similarity, capability_distance: 1-level.capability_min_similarity}, scope_summary: state.subject ? {subject_members, capability_matches} : null};
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
  const measured = data.sources.filter(source => source.alignment?.basis === 'matched' &&
    (!$('source').value || source.source_id === $('source').value) &&
    ($('use').value === 'all' || source.alignment.intended_use === $('use').value));
  if (!measured.length) {
    for (const id of ['task-count', 'mapped-count', 'capability-count']) $(id).textContent = '—';
    $('source-count').textContent = 'No measured sources in this selection';
    $('broad-count').textContent = 'Coverage unavailable';
    $('sample-note').textContent = 'No matching population was measured in this release.';
    $('source-ledger').hidden = true; $('coverage-scope').hidden = true;
    renderBreadcrumbs();
    $('curriculum-map').replaceChildren(node('p', 'Coverage has not been measured for this selection.', 'scope-note'));
    $('outcomes').replaceChildren(); $('matrix-summary').textContent = 'Coverage unavailable';
    history.replaceState(null, '', viewUrl());
    return;
  }
  const total = data.totals.denominator;
  const mapped = data.totals.assigned + data.totals.subject_only;
  $('task-count').textContent = number.format(total);
  $('sample-note').textContent = `Examples are sampled up to ${data.examples.source_sample_size} per source and ${data.examples.area_sample_size} per curriculum area.`;
  $('mapped-count').textContent = total ? percent(mapped / total) : '—';
  $('capability-count').textContent = number.format(data.totals.assigned);
  $('broad-count').textContent = `${number.format(data.totals.subject_only)} have a subject-only match`;
  $('source-count').textContent = 'Converted tasks before admission';
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
  element.style.backgroundColor = `var(--heat-${Math.max(1, Math.ceil(intensity * 5))})`;
  element.style.color = 'var(--ink)';
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
  const compareAreas = (a, b) => $('area-sort').value === 'count' ?
    (counts.get(key(b)) || 0) - (counts.get(key(a)) || 0) || a.name.localeCompare(b.name) : a.name.localeCompare(b.name);
  for (const siblings of children.values()) siblings.sort(compareAreas);
  const matches = new Map();
  function matching(area) {
    if (!matches.has(key(area))) matches.set(key(area), !query || `${area.name} ${area.description}`.toLowerCase().includes(query) || (children.get(key(area)) || []).some(matching));
    return matches.get(key(area));
  }
  const focused = state.nodes.get(`${state.subject}/${state.parent}`);
  const roots = focused ? [focused] : data.hierarchy.filter(area => area.kind === 'subject').sort(compareAreas);
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
    section.dataset.nodeId = area.node_id;
    section.append(areaRow(area));
    const descendants = children.get(key(area)) || [];
    if ((showSkills || query) && descendants.length) {
      const nested = node('div', '', 'map-children');
      const includeChildren = includeAll || (query && `${area.name} ${area.description}`.toLowerCase().includes(query));
      appendChildren(area, nested, includeChildren);
      section.append(nested);
    }
    return section;
  }
  function appendChildren(area, container, includeAll) {
    const descendants = children.get(key(area)) || [];
    let displayedCount = 0, childCount = 0, displayed = 0;
    for (const child of descendants) {
      const count = counts.get(key(child)) || 0;
      childCount += count;
      const item = branch(child, includeAll);
      if (item) { container.append(item); displayedCount += count; displayed++; }
    }
    // Domain membership uses a broader cutoff than capability placement.
    // Search can also hide children; keep both differences visible in the sum.
    const count = counts.get(key(area)) || 0;
    const subjectOnly = count - childCount;
    const hidden = childCount - displayedCount;
    if (subjectOnly > 0) {
      const row = node('div', '', 'map-area map-remainder');
      const unlisted = {...area, name: 'Other/unlisted (subject only)', scope: 'unlisted'};
      const action = () => openTasks(unlisted);
      row.append(button(unlisted.name, 'map-area-name', action), button(number.format(subjectOnly), 'map-area-count', action));
      row.title = 'subject-only: matched this subject but no specific capability';
      container.append(row);
    }
    if (hidden > 0) {
      const row = node('div', '', 'map-area map-remainder map-search-hidden');
      const action = () => { $('search').value = ''; renderCoverage(); };
      row.append(button('Hidden by search (show all)', 'map-area-name', action), button(number.format(hidden), 'map-area-count', action));
      container.append(row);
    }
    return displayed;
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
    topics += appendChildren(area, body, includeAll);
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

async function showTasks(area, outcome = 'all', sourceId = '') {
  state.taskSelection = {area, outcome, sourceId};
  state.taskQuery = new URLSearchParams(state.data.filters); state.taskQuery.set('release', state.data.release_id);
  if (area) { state.taskQuery.set('node_id', area.node_id); state.taskQuery.set('subject_id', area.subject_id); state.taskQuery.set('scope', area.scope || 'all'); }
  if (sourceId) state.taskQuery.set('source_id', sourceId);
  state.taskQuery.set('outcome', outcome);
  const request = ++state.taskRequest;
  state.next = ''; $('task-list').setAttribute('aria-busy', 'true'); $('area-sources').replaceChildren(); $('area-summary').textContent = ''; $('task-list').replaceChildren(); $('task-panel').hidden = false;
  $('task-heading').textContent = area?.name || (outcome === 'all' ? 'Sampled examples' : outcomes[outcome]);
  $('task-scope').textContent = `${sourceId || $('source').selectedOptions[0].textContent} · sampled examples`;
  if (area) {
    try {
      const key = state.taskQuery.toString();
      if (!state.breakdowns.has(key)) state.breakdowns.set(key, await json(`api/curriculum/area?${state.taskQuery}`));
      const stored = state.breakdowns.get(key);
      const sources = stored.confidence_sources ? stored.confidence_sources.map(source => ({...source, count: source.instances[state.level]})).filter(source => source.count > 0).sort((a,b) => b.count-a.count || a.source_id.localeCompare(b.source_id)) : stored.sources;
      const details = {...stored, sources, count: sources.reduce((sum, source) => sum + source.count, 0)};
      if (request !== state.taskRequest) return;
      $('area-summary').textContent = `${number.format(details.count)} task instances · complete source breakdown. Task examples below are sampled.`;
      for (const source of details.sources) {
        const row = node('div', '', 'area-source');
        const select = button(source.name || source.source_id, 'text-button', () => {
          showTasks(area, outcome, source.source_id);
        });
        const evidence = node('a', 'Source evidence ↗'); evidence.href = `source.html#source_id=${encodeURIComponent(source.source_id)}`;
        row.append(select, node('b', number.format(source.count)), evidence); $('area-sources').append(row);
      }
    } catch (error) { if (request === state.taskRequest) $('area-summary').textContent = `Source breakdown could not be loaded. ${error.message}`; }
  }
  if (request === state.taskRequest) await (state.raw.confidence ? loadPool() : loadTasks());
}
function confidenceOutcome(task) {
  if (task.description_status === 'skipped' || task.description_status === 'insufficient_context') return task.description_status;
  const members = (task.matches || []).filter(match => match.membership_similarity >= 1-state.data.thresholds.subject_distance);
  if (members.some(match => match.candidates[0].similarity >= 1-state.data.thresholds.capability_distance)) return 'assigned';
  return members.length ? 'subject_only' : 'out_of_scope';
}
function selectedSample(task) {
  const {area, outcome} = state.taskSelection;
  const result = confidenceOutcome(task);
  if (outcome !== 'all' && outcome !== result) return null;
  if (!area) return {...task, outcome: result};
  const match = (task.matches || []).find(match => match.subject_id === area.subject_id && match.membership_similarity >= 1-state.data.thresholds.subject_distance);
  if (!match) return null;
  const placed = new Set([`subject:${area.subject_id}`]);
  if (match.candidates[0].similarity >= 1-state.data.thresholds.capability_distance) {
    let current = state.nodes.get(`${area.subject_id}/${match.candidates[0].capability_id}`);
    while (current) { placed.add(current.node_id); current = state.nodes.get(`${current.subject_id}/${current.parent_id}`); }
  }
  if (!placed.has(area.node_id)) return null;
  if (area.scope === 'unlisted' && state.data.hierarchy.some(node => node.subject_id === area.subject_id && node.parent_id === area.node_id && placed.has(node.node_id))) return null;
  return {...task, outcome: area.scope === 'unlisted' ? 'subject_only' : result};
}
function appendTask(task) {
  const item = button('', 'task-row', () => openDetail(task));
  const source = state.data.sources.find(source => source.source_id === task.source_id);
  const identity = node('span', '', 'task-source'); identity.append(node('span', 'Source', 'task-column-label'), node('span', source?.name || task.source_id), node('span', task.task_id, 'task-id')); identity.title = task.source_id;
  item.append(identity, node('span', task.task_preview || (task.outcome === 'skipped' ? 'Embedding skipped; open details for the failure.' : 'Instructions do not establish the requested operation.'), 'task-description'), node('span', outcomes[task.outcome], 'task-outcome'));
  $('task-list').append(item);
}
async function loadPool() {
  const request = ++state.taskRequest;
  const params = new URLSearchParams(state.data.filters);
  params.set('release', state.data.release_id); params.set('sample_pool', 'true'); params.set('limit', '1000');
  const {area, sourceId} = state.taskSelection;
  if (area) params.set('subject_id', area.subject_id);
  if (sourceId) params.set('source_id', sourceId);
  const key = params.toString();
  if (!state.pools.has(key)) {
    const entry = {tasks: [], complete: false, onProgress: null, promise: null};
    state.pools.set(key, entry);
    entry.promise = (async () => {
      let cursor = '';
      do {
        if (cursor) params.set('cursor', cursor);
        const data = await json(`api/curriculum/tasks?${params}`);
        entry.tasks.push(...data.tasks); cursor = data.next_cursor;
        entry.onProgress?.();
      } while (cursor);
      entry.complete = true;
      entry.onProgress?.();
    })().catch(error => { state.pools.delete(key); throw error; });
  }
  const entry = state.pools.get(key);
  let rendered = 0;
  entry.onProgress = () => {
    if (request !== state.taskRequest) return;
    for (const task of entry.tasks.slice(rendered)) { const selected = selectedSample(task); if (selected) appendTask(selected); }
    rendered = entry.tasks.length;
    const count = $('task-list').children.length;
    $('task-list').setAttribute('aria-busy', String(!entry.complete));
    $('task-message').textContent = entry.complete ?
      (count ? `${number.format(count)} sampled examples · all stored samples in this selection` : 'No stored sample examples match this confidence level.') :
      `${number.format(count)} sampled examples · loading more stored samples…`;
  };
  $('more-tasks').hidden = true;
  entry.onProgress();
  try { await entry.promise; }
  catch (error) {
    if (request === state.taskRequest) { $('task-list').setAttribute('aria-busy', 'false'); $('task-message').textContent = `Tasks could not be loaded. ${error.message}`; }
  }
}
async function loadTasks() {
  const request = ++state.taskRequest;
  const params = new URLSearchParams(state.taskQuery); if (state.next) params.set('cursor', state.next);
  $('task-message').textContent = 'Loading tasks…'; $('more-tasks').hidden = true;
  try {
    const data = await json(`api/curriculum/tasks?${params}`);
    if (request !== state.taskRequest) return;
    for (const task of data.tasks) appendTask(task);
    state.next = data.next_cursor; $('more-tasks').hidden = !state.next;
    $('task-message').textContent = $('task-list').children.length ? `${number.format($('task-list').children.length)} sampled examples${state.next ? ' · more available' : ''}` : 'No task examples are available in this area at the release cutoffs.';
  } catch (error) { if (request === state.taskRequest) $('task-message').textContent = `Tasks could not be loaded. ${error.message}`; }
  finally { if (request === state.taskRequest) $('task-list').setAttribute('aria-busy', 'false'); }
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
    if (contentWithheld(detail)) { $('detail-status').textContent = contentWithheldMessage(detail); return; }
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
    $('detail-use').textContent = `Use: ${useLabels[detail.intended_use] || '—'}` + (detail.recorded_intended_use !== detail.intended_use ? ` · Historical recorded Use: ${useLabels[detail.recorded_intended_use] || '—'}` : '');
    $('detail-provenance').textContent = JSON.stringify({source_id: detail.source_id, task_id: detail.task_id, source_key: detail.source_key, task_hash: detail.task_hash, content_hash: detail.content_hash, intended_use: detail.intended_use, recorded_intended_use: detail.recorded_intended_use, population: detail.population, release_id: state.data.release_id}, null, 2);
  } catch (error) { $('detail-description').textContent = `Task could not be loaded. ${error.message}`; }
}
function scoreTable(scores, target = $('detail-matches')) {
  const table = node('table'), head = node('tr'); for (const text of ['Area', 'Cosine', 'Distance', 'At selected cutoffs']) head.append(node('th', text)); const thead = node('thead'); thead.append(head); table.append(thead);
  const body = node('tbody'); for (const score of scores) { const row = node('tr'); row.append(node('td', score.name), node('td', score.similarity.toFixed(3), 'score'), node('td', (1-score.similarity).toFixed(3), 'score'), node('td', score.status, `score-status${score.within ? '' : ' outside'}`)); body.append(row); } table.append(body); target.append(table);
}

function viewUrl() {
  const hash = new URLSearchParams();
  if ($('use').value !== 'all') hash.set('intended_use', $('use').value);
  if (state.level) hash.set('confidence', state.level);
  if ($('source').value) hash.set('source_id', $('source').value);
  if ($('release').value) hash.set('release', $('release').value);
  if (state.parent) { hash.set('parent_id', state.parent); hash.set('subject_id', state.subject); }
  if ($('search').value) hash.set('search', $('search').value);
  return location.pathname + (hash.size ? `#${hash}` : '');
}
function readLocation() {
  const initial = new URLSearchParams(location.hash.slice(1));
  state.level = Math.max(0, Number(initial.get('confidence')) || 0);
  $('use').value = initial.get('intended_use') || 'all';
  $('release').value = initial.get('release') || '';
  $('search').value = initial.get('search') || '';
  const source = initial.get('source_id') || '';
  $('source').value = source;
  if (source && $('source').value !== source) {
    const option = node('option', source); option.value = source; $('source').append(option); $('source').value = source;
  }
  state.parent = initial.get('parent_id') || ''; state.subject = initial.get('subject_id') || '';
}
$('controls').addEventListener('submit', event => event.preventDefault());
for (const id of ['source', 'use', 'release']) $(id).addEventListener('change', () => {
  history.pushState(null, '', viewUrl()); loadCoverage();
});
$('confidence').addEventListener('change', () => {
  state.level = Number($('confidence').value); applyConfidence(); renderCoverage();
  if (!$('task-panel').hidden && state.taskSelection) { const {area, outcome, sourceId} = state.taskSelection; showTasks(area, outcome, sourceId); }
});
$('domain').addEventListener('change', () => navigate($('domain').value ? state.nodes.get(`${$('domain').value}/subject:${$('domain').value}`) : null));
$('search').addEventListener('input', () => {
  ++state.taskRequest; $('task-panel').hidden = true; renderCoverage();
});
$('area-sort').addEventListener('change', renderMap);
$('more-tasks').addEventListener('click', loadTasks);
$('close-tasks').addEventListener('click', () => { ++state.taskRequest; $('task-panel').hidden = true; });
$('close-detail').addEventListener('click', () => $('task-detail').close());
$('reset-view').addEventListener('click', () => {
  $('controls').reset(); state.level = 0; state.parent = ''; state.subject = '';
  history.pushState(null, '', viewUrl()); loadCoverage();
});
window.addEventListener('popstate', () => { readLocation(); loadCoverage(); });
async function start() {
  const [data, taxonomy] = await Promise.all([json('api/curriculum/releases'), json('api/taxonomy')]);
  useLabels = Object.fromEntries(Object.entries(taxonomy.use).map(([value, entry]) => [value, entry.label]));
  for (const release of data.releases.filter(item => !item.selected)) {
    const option = node('option', 'Previous population (evidence)');
    option.value = release.release_id; $('release').append(option);
  }
  readLocation();
  await loadCoverage();
}
start().catch(error => { $('message').textContent = `Coverage could not be loaded. ${error.message}`; });
