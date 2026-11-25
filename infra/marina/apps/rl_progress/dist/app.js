import { showProgress } from './progress.js';

const $ = (id) => document.getElementById(id);
let runs = [];
const selectedAttempts = new Map();

function element(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
const series = (run) => run.series ?? run.label;
const grafanaLink = (run) => run.grafana_url && /^https:\/\/grafana\.oa\.dev\//.test(run.grafana_url) ? element('a', 'Grafana ↗', { href: run.grafana_url, target: '_blank', rel: 'noopener noreferrer' }) : null;
const number = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const percent = (value) => value == null ? 'unavailable' : `${number(value, 2)}%`;
const comparison = (row) => `${row.metric}: ${number(row.control)} → ${number(row.run)}, Δ ${percent(row.delta_percent)}, noise ±${percent(row.noise_percent)}${row.passes == null ? '' : row.passes ? ' · PASS' : ' · FAIL'}`;
const invariant = (row) => `r${row.replicate} ${row.name}: ${row.value} ${row.comparator ?? '(comparator unknown)'} ${row.limit} · ${row.holds ? 'PASS' : 'FAIL'}`;
const hours = (run) => run.elapsed_seconds == null ? null : run.elapsed_seconds / 3600;
const attemptTime = (run) => Date.parse(run.started_at ?? run.run_date ?? run.completed_at) || 0;
function groupedRuns(records = runs) {
  const groups = new Map();
  for (const run of records) {
    if (!groups.has(run.run_id)) groups.set(run.run_id, []);
    groups.get(run.run_id).push(run);
  }
  for (const attempts of groups.values()) attempts.sort((a, b) => attemptTime(b) - attemptTime(a) || b.attempt_id.localeCompare(a.attempt_id));
  return groups;
}
function visibleRuns(groups) {
  return [...groups.values()].map((attempts) => attempts.find((run) => run.attempt_id === selectedAttempts.get(run.run_id)) ?? attempts[0]);
}
function route(run) { return '#' + new URLSearchParams({ run: run.run_id, attempt: run.attempt_id }); }
function selectedFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const runId = params.get('run');
  if (!runId) return null;
  const attempts = groupedRuns().get(runId) ?? [];
  const aliases = runs.filter((run) => run.native_run_id === runId);
  const attemptId = params.get('attempt');
  if (attemptId != null) return [...attempts, ...aliases].find((run) => run.attempt_id === attemptId) ?? null;
  return attempts[0] ?? aliases.sort((a, b) => attemptTime(b) - attemptTime(a))[0] ?? null;
}

function addField(list, title, value) {
  list.append(element('dt', title), element('dd', value ?? 'Not recorded'));
}
function configuration(run) {
  const details = element('details');
  const content = element('pre');
  details.append(element('summary', 'Composed launch configuration'), content);
  details.addEventListener('toggle', async () => {
    if (!details.open || content.textContent) return;
    content.textContent = 'Loading configuration…';
    const query = new URLSearchParams({ run_id: run.run_id, attempt_id: run.attempt_id });
    try {
      const response = await fetch(`./api/configuration?${query}`);
      if (response.status === 404) {
        content.textContent = 'No configuration recorded for this attempt.';
      } else if (response.ok) {
        content.textContent = JSON.stringify(await response.json(), null, 2);
      } else {
        throw new Error(`Could not load configuration (${response.status}).`);
      }
    } catch (error) {
      content.textContent = error.message;
    }
  });
  return details;
}
function showDetails(run) {
  $('detail-title').textContent = run.label;
  const body = $('detail-body'); body.replaceChildren();
  body.append(element('p', run.status === 'failed' ? 'The native attempt or a recorded invariant failed. See the named facts.' : 'Recorded native attempt.', { class: 'note' }));
  const links = element('div', null, { class: 'links' });
  const explicitWandb = run.wandb_url;
  if (explicitWandb && /^https:\/\/wandb\.ai\//.test(explicitWandb)) links.append(element('a', 'W&B run ↗', { href: explicitWandb, target: '_blank', rel: 'noopener noreferrer' }));
  const grafana = grafanaLink(run);
  if (grafana) links.append(grafana);
  if (typeof run.issue === 'number') links.append(element('a', `Experiment #${run.issue} ↗`, { href: `https://github.com/marin-community/marin/issues/${run.issue}`, target: '_blank', rel: 'noopener noreferrer' }));
  if (/^[0-9a-f]{7,40}$/.test(run.commit ?? '')) links.append(element('a', 'Source commit ↗', { href: `https://github.com/marin-community/marin/commit/${run.commit}`, target: '_blank', rel: 'noopener noreferrer' }));
  if (run.has_grader_report) links.append(element('a', 'Grader stats', { href: './graders.html?' + route(run).slice(1) }));
  links.append(element('a', 'Link to this run', { href: route(run) }));
  body.append(links);
  const fields = element('dl', null, { class: 'detail-grid' });
  const facts = [
    ['Model', run.model], ['Preset', series(run)], ['Preset version', run.preset_version],
    ['Base version', run.base_preset_version], ['Main base', run.base_commit], ['Evaluation set', run.evaluation_data_identity],
    ['User', run.user], ['Branch', run.branch], ['Commit', run.commit], ['Uncommitted changes', run.dirty == null ? null : run.dirty ? 'Yes' : 'No'],
    ['PR', run.pr_number], ['Cluster', run.cluster], ['Issue', run.issue],
    ['Run ID', run.run_id], ['Attempt ID', run.attempt_id], ['Iris job', run.iris_job_id],
    ['Verdict', run.gate_result?.verdict], ['Effect', run.gate_result?.effect], ['Gate', run.gate], ['Failed invariants', (run.failed_facts ?? []).join(', ') || null],
    ['Status', `${run.status} · ${run.optimizer_steps ?? 'Unknown'} optimizer updates`],
    ['Planned updates', run.planned_updates], ['Measured updates', run.measured_updates?.join(', ')],
    ['Elapsed', `${number(hours(run), 2)} hours (${number(run.elapsed_seconds, 1)} seconds)`],
    ['Elapsed excluding evaluation', run.elapsed_excluding_eval_seconds == null ? null : `${number(run.elapsed_excluding_eval_seconds, 1)} seconds`],
    ['Timing scope', run.timing_scope], ['Run date', run.run_date], ['Started', run.started_at], ['Completed', run.completed_at],
    ['Source', run.source_revision], ['Training data', run.data_identity], ['Initializer', run.initializer_identity],
    ['Artifact location', run.artifact_uri],
    ['Explicit settings', run.settings?.join(', ') || 'None recorded'],
    ['Engine facts', run.engine == null ? null : JSON.stringify(run.engine)],
  ];
  for (const [title, value] of facts) addField(fields, title, value);
  body.append(fields, configuration(run));
  if (run.gate_result) {
    body.append(element('h3', `Gate report · ${run.gate_result.replicates} replicate(s)`));
    for (const row of run.gate_result.comparisons) body.append(element('p', comparison(row)));
    for (const row of run.gate_result.invariants) body.append(element('p', invariant(row), { class: row.holds ? 'note' : 'warning' }));
    if (run.gate_result.reason) body.append(element('p', run.gate_result.reason));
  }
  if (run.evaluations?.length) {
    body.append(element('h3', 'Evaluations'));
    for (const row of run.evaluations) body.append(element('p', `${row.metric}: ${number(row.run)}${row.tasks == null ? '' : ` · ${row.tasks} tasks`}`));
  }
  body.append(element('h3', 'Evidence and caveats'));
  const evidence = element('ul', null, { class: 'evidence' });
  for (const text of run.evidence ?? []) evidence.append(element('li', text));
  body.append(evidence);
  if (!$('run-detail').open) $('run-detail').showModal();
}
function followHash() {
  const run = selectedFromHash();
  if (run) selectedAttempts.set(run.run_id, run.attempt_id);
  render();
  if (run) showDetails(run);
  else if ($('run-detail').open) $('run-detail').close();
}

function render() {
  const model = $('model-filter').value;
  const matching = model ? runs.filter((run) => run.model === model) : runs;
  const groups = groupedRuns(matching), selected = visibleRuns(groups);
  const preset = $('progress-preset'), previous = preset.value;
  const choices = new Map(matching.filter((run) => run.model && run.series && run.evaluations?.length)
    .map((run) => [JSON.stringify([run.model, run.series]), `${run.model} · ${run.series}`]));
  preset.replaceChildren(...[...choices].sort((a, b) => a[1].localeCompare(b[1]))
    .map(([value, label]) => element('option', label, { value })));
  if (choices.has(previous)) preset.value = previous;
  $('progress-panel').hidden = !choices.size;
  const history = matching.filter((run) => JSON.stringify([run.model, run.series]) === preset.value);
  const charts = $('progress-charts');
  showProgress(charts, history, $('show-branches').checked, route);
  $('progress-empty').hidden = charts.childElementCount > 0;
  $('count').textContent = `${selected.length} runs · ${matching.length} recorded attempts`;
  $('attempts').replaceChildren(...[...selected].sort((a, b) => attemptTime(b) - attemptTime(a)).map((run) => {
    const tr = element('tr');
    const name = element('td'), button = element('button', run.label); button.onclick = () => { location.hash = route(run); }; name.append(button);
    const attempts = element('select', null, { 'aria-label': `Attempt for ${run.label}` });
    for (const attempt of groups.get(run.run_id)) {
      const option = element('option', `${attempt.started_at ?? attempt.run_date ?? attempt.completed_at ?? 'Time unavailable'} · ${attempt.attempt_id} · ${attempt.status}`, { value: attempt.attempt_id });
      option.selected = attempt.attempt_id === run.attempt_id;
      attempts.append(option);
    }
    attempts.onchange = () => { location.hash = route(groups.get(run.run_id).find((attempt) => attempt.attempt_id === attempts.value)); };
    name.append(attempts, element('span', run.status, { class: run.status === 'failed' ? 'warning' : 'note' }));
    const grafana = grafanaLink(run);
    if (grafana) name.append(grafana);
    const report = run.gate_result;
    const objectives = element('td'), constraints = element('td'), verdict = element('td');
    if (report) {
      for (const row of report.comparisons) {
        if (row.role === 'slice') continue;
        const cell = row.role === 'objective' ? objectives : constraints;
        cell.append(element('p', comparison(row), { class: row.passes === false ? 'warning' : '' }));
      }
      for (const row of report.invariants) constraints.append(element('p', invariant(row), { class: row.holds ? 'note' : 'warning' }));
      objectives.append(element('span', `Report across ${report.replicates} replicate(s)`, { class: 'note' }));
      verdict.append(element('strong', `${report.verdict}${report.effect ? ` · ${report.effect}` : ''}`));
      if (report.reason) verdict.append(element('p', report.reason));
    } else {
      objectives.textContent = 'Report details unavailable';
      constraints.textContent = '—';
      verdict.textContent = '—';
    }
    tr.append(name, element('td', run.model ?? 'Not recorded'), objectives, constraints, verdict);
    return tr;
  }));
}
async function refresh() {
  $('refresh').disabled = true; $('error').hidden = true;
  try {
    const response = await fetch('./api/runs');
    if (!response.ok) throw new Error(`Could not load runs (${response.status}).`);
    const data = await response.json(); runs = data.runs;
    const filter = $('model-filter'), previous = filter.value;
    const models = [...new Set(runs.map((run) => run.model).filter(Boolean))].sort();
    filter.replaceChildren(element('option', 'All models', { value: '' }), ...models.map((model) => element('option', model, { value: model })));
    if (models.includes(previous)) filter.value = previous;
    $('warnings').hidden = !(data.skipped_records?.length);
    $('warnings').textContent = (data.skipped_records ?? []).map((record) => `Skipped malformed record ${record.run_id}/${record.attempt_id}: ${record.error}`).join('\n');
    followHash();
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
  finally { $('refresh').disabled = false; }
}
$('refresh').onclick = refresh;
$('model-filter').onchange = render;
$('progress-preset').onchange = render;
$('show-branches').onchange = render;
function closeDetails() {
  if (location.hash.startsWith('#run=')) history.replaceState(null, '', location.pathname + location.search);
  $('run-detail').close();
}
$('close-detail').onclick = closeDetails;
$('run-detail').addEventListener('cancel', (event) => { event.preventDefault(); closeDetails(); });
$('run-detail').addEventListener('click', (event) => {
  const dialog = $('run-detail');
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeDetails();
});
window.addEventListener('hashchange', followHash);
refresh();
