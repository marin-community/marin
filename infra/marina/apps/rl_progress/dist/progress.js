const SVG = 'http://www.w3.org/2000/svg';
const evaluationMetric = /^([^@]+)@(\d+)\/(.+)$/;
const score = (value) => `${(100 * value).toFixed(1)}%`;
const time = (run) => Date.parse(run.completed_at ?? run.started_at ?? run.run_date);
const isMainControl = (run) => run.is_control && run.branch === 'main' && !run.dirty && run.status === 'succeeded';

function metrics(run) {
  const scores = new Map();
  for (const row of run.evaluations ?? []) {
    const match = row.metric.match(evaluationMetric);
    if (!match || !Number.isFinite(row.run)) continue;
    const [, evaluation, step, statistic] = match;
    const key = `${evaluation}/${statistic}`;
    if (!scores.has(key) || Number(step) > scores.get(key).step) scores.set(key, { value: row.run, step: Number(step) });
  }
  return scores;
}

function comparable(previous, next) {
  const a = previous.run, b = next.run;
  return a.evaluation_task_set != null && a.evaluation_task_set === b.evaluation_task_set
    && a.evaluation_protocol_identity != null && a.evaluation_protocol_identity === b.evaluation_protocol_identity;
}

function svg(tag, attributes, label) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label != null) node.textContent = label;
  return node;
}

function plot(metric, controls, branches, route) {
  const chart = svg('svg', { viewBox: '0 0 680 200', role: 'group', 'aria-label': metric, class: 'progress-chart' });
  const points = [...controls, ...branches];
  const values = points.map((row) => row.value);
  const low = Math.min(...values), high = Math.max(...values), padding = Math.max(.025, (high - low) * .12);
  const y = (value) => 160 - 130 * (value - low + padding) / (high - low + 2 * padding);
  const times = [...new Set(points.map((row) => row.time))].sort((a, b) => a - b);
  const positions = new Map(times.map((value, index) => [value, index]));
  const x = (at) => times.length === 1 ? 346 : 68 + 555 * positions.get(at) / (times.length - 1);
  const line = (a, b, className) => chart.append(svg('line', {
    x1: x(a.time), y1: y(a.value), x2: x(b.time), y2: y(b.value), class: className,
  }));

  for (const value of new Set([low, (low + high) / 2, high])) {
    chart.append(svg('line', { x1: 55, x2: 650, y1: y(value), y2: y(value), class: 'history-grid' }),
      svg('text', { x: 50, y: y(value) + 3, 'text-anchor': 'end' }, score(value)));
  }
  for (const [index, point] of controls.entries()) {
    const previous = controls[index - 1];
    if (!previous) continue;
    line(previous, point, 'history-global');
    if (comparable(previous, point)) {
      line(previous, point, 'history-chain');
    } else {
      const marker = svg('line', { x1: x(point.time), y1: 20, x2: x(point.time), y2: 165,
        class: 'history-marker', 'aria-label': 'Evaluation comparability changed' });
      marker.append(svg('title', {}, 'Evaluation tasks or scoring protocol changed or unavailable'));
      chart.append(marker);
    }
  }
  for (const point of points) {
    const label = `${point.run.label} · ${score(point.value)}`;
    const link = svg('a', { href: route(point.run), 'aria-label': label });
    link.append(svg('circle', { cx: x(point.time), cy: y(point.value), r: 4.5,
      class: isMainControl(point.run) ? 'history-chain' : 'history-branch' }),
      svg('title', {}, label));
    chart.append(link);
  }
  const ticks = times.length <= 5 ? times : [times[0], times.at(-1)];
  for (const at of ticks) {
    chart.append(svg('text', { x: x(at), y: 181, 'text-anchor': 'middle' }, new Date(at).toISOString().slice(5, 10)));
  }
  return chart;
}

export function showProgress(container, runs, showBranches, route) {
  const latestByOutput = new Map();
  for (const run of runs.filter(isMainControl).sort((a, b) => time(b) - time(a))) {
    if (Number.isFinite(time(run)) && !latestByOutput.has(run.artifact_uri)) latestByOutput.set(run.artifact_uri, run);
  }
  const controls = [...latestByOutput.values()].reverse();
  const branches = showBranches ? runs.filter((run) => run.branch !== 'main' && Number.isFinite(time(run))) : [];
  const scores = new Map([...controls, ...branches].map((run) => [run, metrics(run)]));
  const selected = [...new Set([...scores.values()].flatMap((values) => [...values.keys()]))]
    .filter((key) => /^([^/]+)\/(family_mean|family\/[^/]+\/mean|mean)$/.test(key));
  const metricsToShow = selected.filter((key) => !/^[^/]+\/mean$/.test(key)
    || !selected.includes(key.replace(/\/mean$/, '/family_mean')));
  const points = (rows, metric) => rows.flatMap((run) => {
    const value = scores.get(run).get(metric)?.value;
    return Number.isFinite(value) ? [{ run, time: time(run), value }] : [];
  });
  const initialOnly = runs.some((run) => run.planned_updates === 0)
    && runs.every((run) => run.planned_updates == null || run.planned_updates === 0);
  container.replaceChildren();
  for (const metric of metricsToShow.sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b))) {
    const figure = document.createElement('figure');
    const label = document.createElement('figcaption');
    label.textContent = metric.replaceAll('/', ' · ').replaceAll('_', ' ');
    if (initialOnly) label.textContent += ' · initializer inference';
    figure.append(label, plot(metric, points(controls, metric), points(branches, metric), route));
    container.append(figure);
  }
}
