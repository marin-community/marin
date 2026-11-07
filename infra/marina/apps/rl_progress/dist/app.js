const $ = (id) => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const colors = ['#286451', '#a06534', '#526bb2', '#9a5079'];
let runs = [];
const selectedAttempts = new Map();

function element(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}
const identity = (run) => JSON.stringify([run.run_id, run.attempt_id]);
const series = (run) => run.series ?? run.label;
const campaign = (run) => run.campaign ?? 'Unspecified';
const color = (run) => colors[Array.from(series(run)).reduce((sum, letter) => sum + letter.codePointAt(0), 0) % colors.length];
const number = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const hours = (run) => run.elapsed_seconds == null ? null : run.elapsed_seconds / 3600;
const attemptTime = (run) => Date.parse(run.started_at ?? run.run_date ?? run.completed_at) || 0;
const day = (date) => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Los_Angeles', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(date));
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
  return [...groups.values()].map((attempts) => attempts.find((run) => run.attempt_id === selectedAttempts.get(run.run_id)) ?? attempts[0])
    .filter((run) => $('campaign').value === 'all' || campaign(run) === $('campaign').value);
}
function route(run) { return '#' + new URLSearchParams({ run: run.run_id, attempt: run.attempt_id }); }
function selectedFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const attempts = groupedRuns().get(params.get('run')) ?? [];
  return params.has('attempt') ? attempts.find((run) => run.attempt_id === params.get('attempt')) ?? null : attempts[0] ?? null;
}

function chart(id, rows, xValue, yValue, xLabel, yLabel, { dateAxis = false, labelDots = false, baseline = false } = {}) {
  const host = $(id); host.replaceChildren();
  const points = rows.flatMap((run) => {
    const entries = [{ run, x: xValue(run), y: yValue(run), phase: 'final' }];
    if (baseline && run.baseline) entries.unshift({ run, x: xValue(run), y: run.baseline.reward, phase: 'baseline' });
    return entries.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  });
  if (!points.length) { host.append(element('p', 'No complete measurements for this view.', { class: 'note' })); return; }
  const width = Math.max(330, host.clientWidth || 700), height = id === 'pareto' ? 350 : 285;
  const margin = { left: 55, right: 30, top: 30, bottom: 55 };
  let xMin = Math.min(...points.map((p) => p.x)), xMax = Math.max(...points.map((p) => p.x));
  const xPadding = (xMax - xMin || (dateAxis ? 86400000 : Math.max(xMax, 1))) * 0.12;
  xMin -= xPadding; xMax += xPadding;
  const yMin = Math.min(0, ...points.map((p) => p.y));
  const yMax = Math.max(...points.map((p) => p.y), 0.01) * 1.14;
  const px = (x) => margin.left + (x - xMin) / (xMax - xMin) * (width - margin.left - margin.right);
  const py = (y) => height - margin.bottom - (y - yMin) / (yMax - yMin) * (height - margin.top - margin.bottom);
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': `${yLabel} versus ${xLabel}. Each point links to run details.` });
  for (let i = 0; i <= 4; i++) {
    const value = yMin + (yMax - yMin) * i / 4, y = py(value);
    root.append(svg('line', { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: 'axis' }));
    root.append(svg('text', { x: margin.left - 9, y: y + 4, 'text-anchor': 'end', class: 'tick' }, number(value, yLabel.includes('Reward') ? 3 : 2)));
  }
  for (let i = 0; i <= 3; i++) {
    const value = xMin + (xMax - xMin) * i / 3;
    root.append(svg('text', { x: px(value), y: height - margin.bottom + 23, 'text-anchor': 'middle', class: 'tick' }, dateAxis ? day(value).slice(5) : number(value, 2)));
  }
  root.append(svg('text', { x: margin.left, y: 15, class: 'axis-label' }, yLabel));
  root.append(svg('text', { x: width / 2, y: height - 8, 'text-anchor': 'middle', class: 'axis-label' }, xLabel));
  const eligibleForChart = (run) => id === 'gain' ? run.trusted_gain_eligible : run.pareto_eligible;
  const eligible = rows.filter(eligibleForChart);
  const frontier = new Set(eligible.filter((a) => !eligible.some((b) => a.comparison_key === b.comparison_key && hours(b) <= hours(a) && b.final.reward >= a.final.reward && (hours(b) < hours(a) || b.final.reward > a.final.reward))).map(identity));
  for (const point of points) {
    const { run, phase } = point;
    const link = svg('a', { href: route(run), class: 'dot', tabindex: '0', 'aria-label': `View ${run.label}: ${yLabel.toLowerCase()} ${number(point.y)}${phase === 'baseline' ? ' (baseline)' : ''}, ${number(hours(run), 2)} elapsed hours` });
    link.append(svg('title', {}, `${run.label}\n${yLabel}${phase === 'baseline' ? ' (baseline)' : ''}: ${number(point.y)}\n${eligibleForChart(run) ? 'Eligible for comparison' : run.status === 'failed' ? 'Failed attempt; descriptive' : 'Provisional; unranked'}\nClick for run details`));
    link.append(svg('circle', { cx: px(point.x), cy: py(point.y), r: phase === 'baseline' ? 5 : 6, fill: phase === 'baseline' || run.partial_coverage ? 'var(--panel)' : color(run), stroke: color(run), 'stroke-width': id === 'pareto' && frontier.has(identity(run)) ? 4 : 2, 'stroke-dasharray': run.partial_coverage ? '2 2' : 'none' }));
    root.append(link);
    if (labelDots && phase === 'final') root.append(svg('text', { x: px(point.x) + 10, y: py(point.y) - 10, class: 'dot-label' }, series(run)));
  }
  host.append(root);
}

function addField(list, title, value) {
  list.append(element('dt', title), element('dd', value ?? 'Not recorded'));
}
function showDetails(run) {
  $('detail-title').textContent = run.label;
  const body = $('detail-body'); body.replaceChildren();
  const warning = run.status === 'failed' ? 'The native attempt failed. Its retained evidence is descriptive.' : run.provisional ? 'Unranked: the recorded coverage, timing or evaluation identity does not permit comparison.' : run.partial_coverage ? 'This attempt meets its recorded coverage threshold with some unscored answers.' : 'Complete comparison metadata.';
  body.append(element('p', warning, { class: run.partial_coverage ? 'warning' : 'note' }));
  const links = element('div', null, { class: 'links' });
  const explicitWandb = run.wandb_url;
  if (explicitWandb && /^https:\/\/wandb\.ai\//.test(explicitWandb)) links.append(element('a', 'W&B run ↗', { href: explicitWandb, target: '_blank', rel: 'noopener noreferrer' }));
  if (run.issue) links.append(element('a', `Experiment #${run.issue} ↗`, { href: `https://github.com/marin-community/marin/issues/${run.issue}`, target: '_blank', rel: 'noopener noreferrer' }));
  links.append(element('a', 'Link to this run', { href: route(run) }));
  body.append(links);
  const fields = element('dl', null, { class: 'detail-grid' });
  const facts = [
    ['Campaign', campaign(run)], ['Series', series(run)], ['Run ID', run.run_id], ['Attempt ID', run.attempt_id], ['Iris job', run.iris_job_id], ['Resume source', run.resume_from], ['Baseline update', run.baseline_step],
    ['Verdict', run.verdict], ['Gate', run.gate], ['Failed facts', (run.failed_facts ?? []).join(', ') || null],
    ['Status', `${run.status} · ${run.optimizer_steps} optimizer updates`],
    ['Reward', `${number(run.baseline?.reward)} → ${number(run.final?.reward)}`],
    ['Coverage', `Baseline ${run.baseline?.scored ?? 0}/${run.baseline?.attempted ?? 0}; final ${run.final?.scored ?? 0}/${run.final?.attempted ?? 0}`],
    ['Elapsed', `${number(hours(run), 2)} hours (${number(run.elapsed_seconds, 1)} seconds)`],
    ['Timing scope', run.timing_scope], ['Submission date', run.run_date], ['Started', run.started_at], ['Completed', run.completed_at],
    ['Source', run.source_revision], ['Training data', run.data_identity], ['Initializer', run.initializer_identity],
    ['Evaluation', `${run.evaluation.name} · ${run.evaluation.version}`], ['Task set', run.evaluation.task_set_sha256],
    ['Evaluation protocol', run.evaluation.protocol_sha256 ?? 'Unknown · unranked'], ['Artifact location', run.artifact_uri],
    ['Engine facts', run.engine == null ? null : JSON.stringify(run.engine)],
  ];
  for (const [title, value] of facts) addField(fields, title, value);
  body.append(fields, element('h3', 'Evidence and caveats'));
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
  $('count').textContent = `${selected.length} runs · ${runs.length} recorded attempts · ${selected.filter((run) => run.pareto_eligible).length} eligible for frontier ranking`;
  chart('pareto', selected, hours, (run) => run.final?.reward, 'Elapsed hours', 'Final reward', { labelDots: true });
  chart('reward', selected, (run) => Date.parse(run.run_date ?? run.started_at), (run) => run.final?.reward, 'Run date (Los Angeles)', 'Reward', { dateAxis: true, baseline: true });
  chart('duration', selected, (run) => Date.parse(run.run_date ?? run.started_at), hours, 'Run date (Los Angeles)', 'Elapsed hours', { dateAxis: true });
  chart('gain', selected, (run) => Date.parse(run.run_date ?? run.started_at), (run) => run.gain_per_hour, 'Run date (Los Angeles)', 'Reward gain / hour', { dateAxis: true });
  $('frontier-note').textContent = selected.some((run) => run.pareto_eligible) ? 'Rings mark the frontier within each verified comparison group.' : 'No eligible comparison points in this view; see each run’s coverage, timing and evaluation identity.';
  $('legend').replaceChildren(...[...new Set(selected.map(series))].map((name) => {
    const item = element('span'); item.append(element('i', null, { class: 'swatch', style: `background:${color(selected.find((run) => series(run) === name))}` }), document.createTextNode(name)); return item;
  }));
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
    tr.append(name, element('td', `${number(run.baseline?.reward)} → ${number(run.final?.reward)}`), element('td', `${number(hours(run), 2)} h`), element('td', run.optimizer_steps), element('td', `${run.baseline?.scored ?? 0}/${run.baseline?.attempted ?? 0} → ${run.final?.scored ?? 0}/${run.final?.attempted ?? 0}`), element('td', run.status === 'failed' ? 'Failed' : run.provisional ? 'Unranked' : run.partial_coverage ? 'Meets coverage threshold' : 'Complete'));
    return tr;
  }));
}
async function refresh() {
  $('refresh').disabled = true; $('error').hidden = true;
  try {
    const response = await fetch('./api/runs');
    if (!response.ok) throw new Error(`Could not load runs (${response.status}).`);
    const data = await response.json(); runs = data.runs;
    const previous = $('campaign').value;
    $('campaign').replaceChildren(element('option', 'All campaigns', { value: 'all' }), ...[...new Set(runs.map(campaign))].sort().map((value) => element('option', value, { value })));
    $('campaign').value = [...$('campaign').options].some((option) => option.value === previous) ? previous : 'all';
    $('warnings').hidden = !(data.skipped_records?.length);
    $('warnings').textContent = (data.skipped_records ?? []).map((record) => `Skipped malformed record ${record.run_id}/${record.attempt_id}: ${record.error}`).join('\n');
    followHash();
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
  finally { $('refresh').disabled = false; }
}
$('refresh').onclick = refresh;
$('campaign').onchange = render;
$('close-detail').onclick = () => $('run-detail').close();
$('run-detail').addEventListener('click', (event) => {
  const dialog = $('run-detail');
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
$('run-detail').addEventListener('close', () => { if (location.hash.startsWith('#run=')) history.replaceState(null, '', location.pathname + location.search); });
window.addEventListener('hashchange', followHash);
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 120); });
refresh();
