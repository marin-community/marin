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
const number = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const percent = (value) => value == null ? 'unavailable' : `${number(value, 2)}%`;
const comparison = (row) => `${row.metric}: ${number(row.control)} → ${number(row.run)}, Δ ${percent(row.delta_percent)}, noise ±${percent(row.noise_percent)}${row.passes == null ? '' : row.passes ? ' · PASS' : ' · FAIL'}`;
const invariant = (row) => {
  const name = row.name.split('/').pop();
  const limit = name === 'grader_error_fraction' ? `at most ${row.limit}` : name === 'panel_scored_fraction' ? `at least ${row.limit}` : row.limit;
  return `r${row.replicate} ${row.name}: ${row.value} (required ${limit}) · ${row.holds ? 'PASS' : 'FAIL'}`;
};
const hours = (run) => run.elapsed_seconds == null ? null : run.elapsed_seconds / 3600;
const attemptTime = (run) => Date.parse(run.started_at ?? run.run_date ?? run.completed_at) || 0;
function groupedRuns() {
  const groups = new Map();
  for (const run of runs) {
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
  const attempts = groupedRuns().get(params.get('run')) ?? [];
  return params.has('attempt') ? attempts.find((run) => run.attempt_id === params.get('attempt')) ?? null : attempts[0] ?? null;
}

function addField(list, title, value) {
  list.append(element('dt', title), element('dd', value ?? 'Not recorded'));
}
function showDetails(run) {
  $('detail-title').textContent = run.label;
  const body = $('detail-body'); body.replaceChildren();
  body.append(element('p', run.status === 'failed' ? 'The native attempt or a recorded invariant failed. See the named facts.' : 'Recorded native attempt.', { class: 'note' }));
  const links = element('div', null, { class: 'links' });
  const explicitWandb = run.wandb_url;
  if (explicitWandb && /^https:\/\/wandb\.ai\//.test(explicitWandb)) links.append(element('a', 'W&B run ↗', { href: explicitWandb, target: '_blank', rel: 'noopener noreferrer' }));
  if (run.issue) links.append(element('a', `Experiment #${run.issue} ↗`, { href: `https://github.com/marin-community/marin/issues/${run.issue}`, target: '_blank', rel: 'noopener noreferrer' }));
  links.append(element('a', 'Link to this run', { href: route(run) }));
  body.append(links);
  const fields = element('dl', null, { class: 'detail-grid' });
  const facts = [
    ['Series', series(run)], ['Run ID', run.run_id], ['Attempt ID', run.attempt_id], ['Iris job', run.iris_job_id],
    ['Verdict', run.gate_result?.verdict ?? run.verdict], ['Label', run.gate_result?.label], ['Gate', run.gate], ['Failed facts', (run.failed_facts ?? []).join(', ') || null],
    ['Status', `${run.status} · ${run.optimizer_steps ?? 'Unknown'} optimizer updates`],
    ['Reward', `${number(run.baseline?.reward)} → ${number(run.final?.reward)}`],
    ['Coverage', `Baseline ${run.baseline?.scored ?? 'Unknown'}/${run.baseline?.attempted ?? 'Unknown'}; final ${run.final?.scored ?? 'Unknown'}/${run.final?.attempted ?? 'Unknown'}`],
    ['Elapsed', `${number(hours(run), 2)} hours (${number(run.elapsed_seconds, 1)} seconds)`],
    ['Timing scope', run.timing_scope], ['Run date', run.run_date], ['Started', run.started_at], ['Completed', run.completed_at],
    ['Source', run.source_revision], ['Training data', run.data_identity], ['Initializer', run.initializer_identity],
    ['Evaluation', run.evaluation ? `${run.evaluation.name} · ${run.evaluation.version}` : null], ['Task set', run.evaluation?.task_set_sha256],
    ['Evaluation protocol', run.evaluation?.protocol_sha256], ['Artifact location', run.artifact_uri],
    ['Engine facts', run.engine == null ? null : JSON.stringify(run.engine)],
  ];
  for (const [title, value] of facts) addField(fields, title, value);
  body.append(fields);
  if (run.gate_result) {
    body.append(element('h3', `Gate report · ${run.gate_result.replicates} replicate(s)`));
    for (const row of run.gate_result.comparisons) body.append(element('p', comparison(row)));
    for (const row of run.gate_result.invariants) body.append(element('p', invariant(row), { class: row.holds ? 'note' : 'warning' }));
    if (run.gate_result.reason) body.append(element('p', run.gate_result.reason));
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
  const groups = groupedRuns(), selected = visibleRuns(groups);
  $('count').textContent = `${selected.length} runs · ${runs.length} recorded attempts`;
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
    const report = run.gate_result;
    const objectives = element('td'), constraints = element('td'), verdict = element('td');
    if (report) {
      for (const row of report.comparisons) {
        const cell = row.role === 'objective' ? objectives : constraints;
        cell.append(element('p', comparison(row), { class: row.passes === false ? 'warning' : '' }));
      }
      for (const row of report.invariants) constraints.append(element('p', invariant(row), { class: row.holds ? 'note' : 'warning' }));
      objectives.append(element('span', `Report across ${report.replicates} replicate(s)`, { class: 'note' }));
      verdict.append(element('strong', `${report.verdict}${report.label ? ` · ${report.label}` : ''}`));
      if (report.reason) verdict.append(element('p', report.reason));
    } else {
      objectives.textContent = 'Report details unavailable';
      constraints.textContent = '—';
      verdict.textContent = run.verdict ?? '—';
    }
    tr.append(name, objectives, constraints, verdict);
    return tr;
  }));
}
async function refresh() {
  $('refresh').disabled = true; $('error').hidden = true;
  try {
    const response = await fetch('./api/runs');
    if (!response.ok) throw new Error(`Could not load runs (${response.status}).`);
    const data = await response.json(); runs = data.runs;
    $('warnings').hidden = !(data.skipped_records?.length);
    $('warnings').textContent = (data.skipped_records ?? []).map((record) => `Skipped malformed record ${record.run_id}/${record.attempt_id}: ${record.error}`).join('\n');
    followHash();
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
  finally { $('refresh').disabled = false; }
}
$('refresh').onclick = refresh;
$('close-detail').onclick = () => $('run-detail').close();
$('run-detail').addEventListener('click', (event) => {
  const dialog = $('run-detail');
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
$('run-detail').addEventListener('close', () => { if (location.hash.startsWith('#run=')) history.replaceState(null, '', location.pathname + location.search); });
window.addEventListener('hashchange', followHash);
refresh();
