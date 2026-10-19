/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat('en-US');
const labels = {good: 'Good', some_issues: 'Some issues', bad: 'Bad', unknown: 'Unknown', unassessed: 'Not assessed'};
const useLabels = {train: 'Training', eval: 'Evaluation', unprocessed: 'Not processed'};
const columns = [['display_name', 'Source'], ['examples', 'Examples'], ['use', 'Use'], ['quality', 'Quality'], ['area', 'Top curriculum area']];
const state = {sources: [], sort: 'examples', direction: -1};

function grade(row) {
  return row.assessment.status === 'complete' ? row.assessment.admitted.grade : 'unassessed';
}
function value(row, key) {
  if (key === 'examples') return row.coverage?.example_count;
  if (key === 'use') return row.coverage?.intended_use || 'unprocessed';
  if (key === 'quality') return grade(row);
  if (key === 'area') return row.coverage?.top_area?.name;
  return row[key];
}
function filtered() {
  const search = $('search').value.trim().toLowerCase();
  return state.sources.filter(row => {
    const matches = ['origin', 'quality', 'use'].every(key => $(key).value === 'all' || $(key).value === value(row, key));
    const text = [row.display_name, row.name, row.dataset_id, row.family, ...(row.tags || [])].join(' ').toLowerCase();
    return matches && (!search || text.includes(search));
  }).sort((a, b) => {
    const x = value(a, state.sort), y = value(b, state.sort);
    if (x === null || x === undefined) return y === null || y === undefined ? a.id.localeCompare(b.id) : 1;
    if (y === null || y === undefined) return -1;
    const order = typeof x === 'number' ? x - y : String(x).localeCompare(String(y), undefined, {numeric: true});
    return order * state.direction || a.id.localeCompare(b.id);
  });
}
function curriculumLink(row, area) {
  const params = new URLSearchParams({source_id: row.id});
  if (area) { params.set('subject_id', area.subject_id); params.set('parent_id', area.node_id); }
  return `curriculum.html#${params}`;
}
function render() {
  const headings = columns.map(([key, title]) => {
    const th = node('th'), button = node('button', title);
    th.scope = 'col';
    th.setAttribute('aria-sort', state.sort === key ? state.direction === 1 ? 'ascending' : 'descending' : 'none');
    if (key === 'examples') th.className = 'num';
    if (state.sort === key) button.append(node('span', state.direction === 1 ? '↑' : '↓', 'sort-indicator'));
    button.addEventListener('click', () => {
      state.direction = state.sort === key ? -state.direction : key === 'examples' ? -1 : 1;
      state.sort = key; update();
    });
    th.append(button); return th;
  });
  $('headers').replaceChildren(...headings);
  const rows = filtered();
  $('rows').replaceChildren(...rows.map(row => {
    const tr = node('tr'); tr.dataset.sourceId = row.id;
    const name = node('td');
    const open = node('button', row.display_name, 'source-name');
    open.title = row.display_name; open.addEventListener('click', () => details(row));
    name.append(open, node('span', row.origin, 'source-subtitle'));
    const examples = node('td', row.coverage ? number.format(row.coverage.example_count) : '—', 'num');
    examples.title = row.coverage ? 'Converted examples before curation admission' : 'No processed examples in this release';
    const use = node('td', row.coverage ? useLabels[row.coverage.intended_use] : '—', 'use-cell');
    const quality = node('td');
    const badge = node('span', labels[grade(row)], `chip quality-${grade(row)}`);
    if (row.assessment.status === 'complete') {
      const anchor = node('a'); anchor.href = `assessment.html?source=${encodeURIComponent(row.id)}`;
      anchor.title = row.assessment.admitted.grade_reason; anchor.append(badge);
      quality.append(anchor, node('span', `${number.format(row.assessment.admitted.reviewed)} reviewed`, 'source-subtitle'));
    } else quality.append(badge);
    const area = node('td');
    if (row.coverage?.top_area) {
      const anchor = node('a', row.coverage.top_area.name, 'area-link');
      anchor.href = curriculumLink(row, row.coverage.top_area);
      anchor.title = `${number.format(row.coverage.top_area.instances)} examples in this area`;
      area.append(anchor);
    } else area.append(node('span', '—', 'muted'));
    tr.append(name, examples, use, quality, area); return tr;
  }));
  $('empty').hidden = rows.length > 0;
  $('visible-count').textContent = `${number.format(rows.length)} of ${number.format(state.sources.length)} sources`;
}
function details(row) {
  $('detail-name').textContent = row.display_name; $('detail-notes').textContent = row.notes;
  const controls = row.curation?.verification;
  const fields = [
    ['Catalog', row.origin], ['Converted examples', row.coverage?.example_count],
    ['Use', useLabels[row.coverage?.intended_use]], ['Quality', labels[grade(row)]],
    ['Top curriculum area', row.coverage?.top_area?.name],
    ['Input rows', row.task_count], ['Source ID', row.id], ['Dataset revision', row.dataset_revision],
    ['Pipeline', row.pipeline?.name], ['Curation', row.curation?.status],
    ['Assessment model', row.assessment.model?.model], ['Assessment note', row.assessment.reason],
    ['Admission counts', row.curation?.admission_counts], ['Control status', controls?.status], ['Control counts', controls?.counts],
  ];
  $('detail-fields').replaceChildren(...fields.filter(([, item]) => item !== null && item !== undefined).map(([title, item]) => {
    const field = node('dl', undefined, 'field');
    field.append(node('dt', title), node('dd', typeof item === 'object' ? JSON.stringify(item) : String(item)));
    return field;
  }));
  const links = [];
  if (row.coverage) { const anchor = node('a', 'Curriculum coverage'); anchor.href = curriculumLink(row); links.push(anchor); }
  if (row.assessment.status === 'complete') {
    const anchor = node('a', 'Quality assessment'); anchor.href = `assessment.html?source=${encodeURIComponent(row.id)}`; links.push(anchor);
  }
  if (row.url?.startsWith('https://')) {
    const anchor = node('a', 'Original dataset ↗'); anchor.href = row.url; anchor.rel = 'noopener'; anchor.target = '_blank'; links.push(anchor);
  }
  $('detail-links').replaceChildren(...links); $('details').showModal();
}
async function load() {
  const data = await json('api/sources');
  state.sources = data.sources;
  const processed = data.sources.filter(row => row.coverage);
  const examples = processed.reduce((total, row) => total + row.coverage.example_count, 0);
  $('message').textContent = `${number.format(data.sources.length)} sources · ${number.format(examples)} examples · ${processed.length} sources processed`;
  $('snapshot-info').textContent = `Curation inventory dated ${new Date(data.created_at).toLocaleDateString()}.`;
  for (const origin of [...new Set(data.sources.map(row => row.origin))].sort()) {
    const option = node('option', origin); option.value = origin; $('origin').append(option);
  }
  readLocation();
  render();
}
function viewUrl() {
  const params = new URLSearchParams();
  for (const id of ['search', 'origin', 'quality', 'use']) {
    const item = $(id).value.trim();
    if (item && item !== 'all') params.set(id, item);
  }
  if (state.sort !== 'examples') params.set('sort', state.sort);
  if (state.direction !== -1) params.set('direction', state.direction);
  return location.pathname + (params.size ? `#${params}` : '');
}
function readLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  $('filters').reset();
  for (const id of ['search', 'origin', 'quality', 'use']) if (params.has(id)) $(id).value = params.get(id);
  state.sort = params.get('sort') || 'examples';
  state.direction = params.get('direction') === '1' ? 1 : -1;
}
function update(replace = false) {
  history[replace ? 'replaceState' : 'pushState'](null, '', viewUrl());
  render();
}
for (const id of ['origin', 'quality', 'use']) $(id).addEventListener('change', () => update());
$('filters').addEventListener('submit', event => event.preventDefault());
$('search').addEventListener('input', () => update(true));
$('close-details').addEventListener('click', () => $('details').close());
$('clear-filters').addEventListener('click', () => { $('filters').reset(); update(); });
window.addEventListener('popstate', () => { readLocation(); render(); });
$('export').addEventListener('click', () => {
  const quote = item => `"${String(item ?? '').replaceAll('"', '""')}"`;
  const csv = [columns.map(([, title]) => quote(title)).join(','), ...filtered().map(row => columns.map(([key]) => quote(value(row, key))).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const anchor = node('a'); anchor.href = url; anchor.download = 'rl-data-atlas.csv'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
load().catch(error => { $('message').textContent = error.message; $('message').classList.add('error'); });
