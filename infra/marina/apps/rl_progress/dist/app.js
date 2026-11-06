const $ = (id) => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
const colors = { A: '#286451', B: '#a06534', C: '#526bb2', D: '#9a5079' };
let runs = [];

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
const fixed = (run) => run.run_id.includes('snowball-inventory-batch-fixed');
const arm = (run) => run.series ?? run.label.match(/\b([A-D]\d+)\b/)?.[1] ?? run.label;
const seriesColors = { 'learning-medium': '#286451', 'data-medium': '#a06534', 'learning-long': '#526bb2', 'data-long': '#9a5079', 'train-long': '#547e86' };
const armColor = (name) => seriesColors[name] ?? colors[name[0]] ?? '#6d6a60';
const color = (run) => armColor(arm(run));
const number = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const hours = (run) => run.elapsed_seconds / 3600;
const gain = (run) => run.final && run.baseline && hours(run) > 0 ? (run.final.reward - run.baseline.reward) / hours(run) : null;
const day = (date) => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Los_Angeles', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(date));
function visibleRuns() {
  return runs.filter((run) => $('campaign').value === 'all' || (run.campaign ?? (fixed(run) ? 'fixed' : 'earlier')) === $('campaign').value);
}
function coverageLabel(run) {
  if (run.partial_coverage) return run.trusted_gain_eligible ? 'Incomplete · eligible at declared coverage threshold' : 'Incomplete · excluded';
  return run.provisional ? 'Provisional · unranked' : 'Complete';
}
function route(run) { return '#run=' + encodeURIComponent(identity(run)); }
function selectedFromHash() {
  if (!location.hash.startsWith('#run=')) return null;
  return runs.find((run) => route(run) === location.hash) ?? null;
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
  const eligible = rows.filter((run) => run.pareto_eligible);
  const frontier = new Set(eligible.filter((a) => !eligible.some((b) => a.comparison_key === b.comparison_key && hours(b) <= hours(a) && b.final.reward >= a.final.reward && (hours(b) < hours(a) || b.final.reward > a.final.reward))).map(identity));
  for (const point of points) {
    const { run, phase } = point;
    const link = svg('a', { href: route(run), class: 'dot', tabindex: '0', 'aria-label': `View ${run.label}: ${yLabel.toLowerCase()} ${number(point.y)}${phase === 'baseline' ? ' (baseline)' : ''}, ${number(hours(run), 2)} elapsed hours` });
    link.append(svg('title', {}, `${run.label}\n${yLabel}${phase === 'baseline' ? ' (baseline)' : ''}: ${number(point.y)}\n${coverageLabel(run)}\nClick for run details`));
    link.append(svg('circle', { cx: px(point.x), cy: py(point.y), r: phase === 'baseline' ? 5 : 6, fill: phase === 'baseline' || run.partial_coverage ? 'var(--panel)' : color(run), stroke: color(run), 'stroke-width': id === 'pareto' && frontier.has(identity(run)) ? 4 : 2, 'stroke-dasharray': run.partial_coverage ? '2 2' : 'none' }));
    root.append(link);
    if (labelDots && phase === 'final') root.append(svg('text', { x: px(point.x) + 10, y: py(point.y) - 10, class: 'dot-label' }, arm(run)));
  }
  host.append(root);
}

function addField(list, title, value) {
  list.append(element('dt', title), element('dd', value ?? 'Not recorded'));
}
function showDetails(run) {
  $('detail-title').textContent = run.label;
  const body = $('detail-body'); body.replaceChildren();
  const warning = run.partial_coverage ? (run.trusted_gain_eligible ? 'Incomplete evaluation meets the declared coverage threshold for comparison.' : 'Incomplete evaluation: this attempt is excluded from comparison. Training completion and evaluation validity are separate.') : run.provisional ? 'Descriptive result. Runtime evaluation protocol is unknown, so this run is unranked.' : 'Complete comparison metadata.';
  body.append(element('p', warning, { class: run.partial_coverage ? 'warning' : 'note' }));
  const links = element('div', null, { class: 'links' });
  const explicitWandb = run.wandb_url ?? run.links?.wandb;
  if (explicitWandb && /^https:\/\/wandb\.ai\//.test(explicitWandb)) links.append(element('a', 'W&B run ↗', { href: explicitWandb, target: '_blank', rel: 'noopener noreferrer' }));
  else if (fixed(run)) {
    links.append(element('a', 'W&B project ↗', { href: 'https://wandb.ai/marin-community/marin-snowball-curriculum', target: '_blank', rel: 'noopener noreferrer' }));
    links.append(element('span', 'Exact W&B run URL was not recorded.', { class: 'note' }));
  }
  links.append(element('a', 'Link to this run', { href: route(run) }));
  body.append(links);
  const fields = element('dl', null, { class: 'detail-grid' });
  const facts = [
    ['Campaign', run.campaign ?? (fixed(run) ? 'Fixed runtime' : 'Earlier runtime')], ['Run ID', run.run_id], ['Attempt ID', run.attempt_id],
    ['Status', `${run.status} · ${run.optimizer_steps} optimizer updates`],
    ['Verdict', run.verdict], ['Required scored fraction', run.min_scored_fraction ?? 1],
    ['Reward', `${number(run.baseline?.reward)} → ${number(run.final?.reward)}`],
    ['Coverage', `Baseline ${run.baseline?.scored ?? 0}/${run.baseline?.attempted ?? 0}; final ${run.final?.scored ?? 0}/${run.final?.attempted ?? 0}`],
    ['Elapsed', `${number(hours(run), 2)} hours (${number(run.elapsed_seconds, 1)} seconds)`],
    ['Timing scope', run.timing_scope], ['Submission date', run.run_date], ['Started', run.started_at], ['Completed', run.completed_at],
    ['Layout', run.layout], ['Config fingerprint', run.config_fingerprint], ['Iris job', run.iris_job_id],
    ['Source', run.source_revision], ['Training data', run.data_identity], ['Initializer', run.initializer_identity],
    ['Evaluation', `${run.evaluation.name} · ${run.evaluation.version}`], ['Task set', run.evaluation.task_set_sha256],
    ['Evaluation protocol', run.evaluation.protocol_sha256 ?? 'Unknown · unranked'], ['Artifact location', run.artifact_uri],
  ];
  for (const [title, value] of facts) addField(fields, title, value);
  body.append(fields, element('h3', 'Evidence and caveats'));
  const evidence = element('ul', null, { class: 'evidence' });
  for (const text of run.evidence ?? []) evidence.append(element('li', typeof text === 'string' ? text : JSON.stringify(text)));
  body.append(evidence);
  if (!$('run-detail').open) $('run-detail').showModal();
}
function followHash() {
  const run = selectedFromHash();
  if (run) showDetails(run);
  else if ($('run-detail').open) $('run-detail').close();
}

function render() {
  const selected = visibleRuns();
  $('count').textContent = `${selected.length} attempts · ${selected.filter((run) => run.pareto_eligible).length} eligible for frontier ranking`;
  chart('pareto', selected, hours, (run) => run.final?.reward, 'Elapsed hours', 'Final reward', { labelDots: true });
  chart('reward', selected, (run) => Date.parse(run.run_date ?? run.started_at), (run) => run.final?.reward, 'Run date (Los Angeles)', 'Reward', { dateAxis: true, baseline: true });
  chart('duration', selected, (run) => Date.parse(run.run_date ?? run.started_at), hours, 'Run date (Los Angeles)', 'Elapsed hours', { dateAxis: true });
  chart('gain', selected.filter((run) => !run.partial_coverage || run.trusted_gain_eligible), (run) => Date.parse(run.run_date ?? run.started_at), gain, 'Run date (Los Angeles)', 'Reward gain / hour', { dateAxis: true });
  $('frontier-note').textContent = selected.some((run) => run.pareto_eligible) ? 'Rings mark the frontier within each verified comparison group.' : 'These points are descriptive. No verified frontier is drawn because the runtime evaluation protocol is not recorded.';
  $('legend').replaceChildren(...[...new Set(selected.map(arm))].map((name) => {
    const item = element('span'); item.append(element('i', null, { class: 'swatch', style: `background:${armColor(name)}` }), document.createTextNode(name)); return item;
  }));
  $('attempts').replaceChildren(...[...selected].sort((a, b) => Date.parse(b.run_date) - Date.parse(a.run_date)).map((run) => {
    const tr = element('tr');
    const name = element('td'), button = element('button', run.label); button.onclick = () => { location.hash = route(run); followHash(); }; name.append(button);
    tr.append(name, element('td', `${number(run.baseline?.reward)} → ${number(run.final?.reward)}`), element('td', `${number(hours(run), 2)} h`), element('td', run.optimizer_steps), element('td', `${run.baseline?.scored ?? 0}/${run.baseline?.attempted ?? 0} → ${run.final?.scored ?? 0}/${run.final?.attempted ?? 0}`), element('td', coverageLabel(run)));
    return tr;
  }));
}
async function refresh() {
  $('refresh').disabled = true; $('error').hidden = true;
  try {
    const response = await fetch('./api/runs');
    if (!response.ok) throw new Error(`Could not load runs (${response.status}).`);
    const data = await response.json(); runs = data.runs;
    for (const campaign of new Set(runs.map((run) => run.campaign).filter(Boolean))) {
      if (![...$('campaign').options].some((option) => option.value === campaign)) $('campaign').append(element('option', campaign, { value: campaign }));
    }
    if ($('campaign').value === 'fixed' && !runs.some(fixed)) $('campaign').value = 'all';
    render(); followHash();
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
