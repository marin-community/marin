const SVG = 'http://www.w3.org/2000/svg';
const score = (value) => `${(100 * value).toFixed(1)}%`;
const trainerTime = 'Trainer seconds per optimizer update';
const time = (run) => Date.parse(run.completed_at ?? run.started_at ?? run.run_date);
export const isCleanMain = (run) => run.branch === 'main' && run.dirty === false
  && /^[0-9a-f]{7,40}$/.test(run.commit ?? '') && run.base_commit?.startsWith(run.commit);
const isControl = (run) => run.is_control === true && run.dirty === false
  && run.status === 'succeeded' && Number.isFinite(time(run));

export function controlHistory(runs) {
  const branches = new Map();
  for (const run of runs.filter(isControl).sort((a, b) => time(b) - time(a))) {
    if (!run.branch || run.branch === 'main' && !isCleanMain(run)) continue;
    if (!branches.has(run.branch)) branches.set(run.branch, new Map());
    const outputs = branches.get(run.branch);
    if (!outputs.has(run.artifact_uri)) outputs.set(run.artifact_uri, run);
  }
  return new Map([...branches].map(([branch, outputs]) => [
    branch, [...outputs.values()].reverse().filter(hasProgress),
  ]).filter(([, controls]) => controls.length));
}

function scores(run) {
  const scores = new Map();
  for (const row of run.evaluations ?? []) {
    const coordinate = row.evaluation_score;
    if (!coordinate?.final || !Number.isFinite(row.run)) continue;
    if (!/^(family_mean|family\/[^/]+\/mean|mean)$/.test(coordinate.statistic)) continue;
    scores.set(`${coordinate.profile}/${coordinate.statistic}`, {
      value: row.run, comparability: coordinate.comparability,
    });
  }
  return scores;
}

function timing(run) {
  const elapsed = run.elapsed_excluding_eval_seconds, updates = run.optimizer_steps;
  if (run.status !== 'succeeded' || !Number.isFinite(elapsed) || elapsed <= 0 || !Number.isInteger(updates) || updates <= 0) return null;
  return {
    value: elapsed / updates,
    comparability: run.preset_version && run.timing_scope
      ? JSON.stringify([run.preset_version, run.timing_scope, updates]) : null,
  };
}

export const hasProgress = (run) => scores(run).size > 0 || timing(run) != null;

function comparable(previous, next) {
  return previous.comparability != null && previous.comparability === next.comparability;
}

function svg(tag, attributes, label) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label != null) node.textContent = label;
  return node;
}

function plot(metric, controls, branches, route) {
  const isTiming = metric === trainerTime;
  const chart = svg('svg', { viewBox: '0 0 680 200', role: 'group', 'aria-label': metric, class: 'progress-chart' });
  const points = [...controls, ...branches].sort((a, b) => a.time - b.time);
  const values = points.map((row) => row.value);
  const low = Math.min(...values), high = Math.max(...values), padding = Math.max(.025, (high - low) * .12);
  const timingDigits = high === low ? 1 : Math.max(1, Math.min(4, 1 - Math.floor(Math.log10(high - low))));
  const format = isTiming ? (value) => `${value.toFixed(timingDigits)} s` : score;
  const y = (value) => 160 - 130 * (value - low + padding) / (high - low + 2 * padding);
  const positions = new Map(points.map((point, index) => [point, index]));
  const x = (point) => points.length === 1 ? 346 : 68 + 555 * positions.get(point) / (points.length - 1);
  const line = (a, b, className) => chart.append(svg('line', {
    x1: x(a), y1: y(a.value), x2: x(b), y2: y(b.value), class: className,
    'data-from-attempt': a.run.attempt_id, 'data-to-attempt': b.run.attempt_id,
  }));

  for (const value of new Set([low, (low + high) / 2, high])) {
    chart.append(svg('line', { x1: 55, x2: 650, y1: y(value), y2: y(value), class: 'history-grid' }),
      svg('text', { x: 50, y: y(value) + 3, 'text-anchor': 'end' }, format(value)));
  }
  for (const [index, point] of controls.entries()) {
    const previous = controls[index - 1];
    if (!previous) continue;
    line(previous, point, 'history-global');
    if (comparable(previous, point)) {
      line(previous, point, 'history-chain');
    } else {
      const marker = svg('line', { x1: x(point), y1: 20, x2: x(point), y2: 165,
        class: 'history-marker', 'aria-label': isTiming ? 'Timing comparability changed' : 'Evaluation comparability changed',
        'data-attempt': point.run.attempt_id });
      marker.append(svg('title', {}, isTiming ? 'Preset version, timing scope or update count changed or unavailable'
        : 'Evaluation tasks or scoring protocol changed or unavailable'));
      chart.append(marker);
    }
  }
  for (const [rows, className] of [[controls, 'history-chain'], [branches, 'history-branch']]) {
    for (const point of rows) {
      const label = `${point.run.label} · ${format(point.value)}`;
      const link = svg('a', { href: route(point.run), 'aria-label': label });
      link.append(svg('circle', { cx: x(point), cy: y(point.value), r: 4.5, class: className }),
        svg('title', {}, `${label} · ${new Date(point.time).toISOString()} · ${point.run.attempt_id}`));
      chart.append(link);
    }
  }
  const ticks = points.length <= 6 ? points : [points[0], points.at(-1)];
  const dates = new Map(ticks.map((point) => [point, new Date(point.time).toISOString()]));
  const sameDay = dates.get(ticks[0]).slice(0, 10) === dates.get(ticks.at(-1)).slice(0, 10);
  const labels = new Map();
  for (const [point, date] of dates) {
    const label = sameDay ? date.slice(11, 19) : date.slice(5, 16).replace('T', ' ');
    if (!labels.has(label)) labels.set(label, []);
    labels.get(label).push(x(point));
  }
  for (const [label, columns] of labels) {
    chart.append(svg('text', { x: (columns[0] + columns.at(-1)) / 2, y: 181, 'text-anchor': 'middle' }, label));
  }
  chart.append(svg('text', { x: 346, y: 198, 'text-anchor': 'middle' }, 'Attempts by recorded time (UTC)'));
  return chart;
}

export function showProgress(container, runs, controls, showBranches, route) {
  const branches = showBranches
    ? runs.filter((run) => run.branch && run.branch !== 'main' && run.is_control !== true && Number.isFinite(time(run))) : [];
  const rows = [...controls, ...branches];
  const evaluations = new Map(rows.map((run) => [run, scores(run)]));
  const selected = [...new Set(rows.flatMap((run) => [...evaluations.get(run).keys()]))]
    .sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b));
  if (rows.some((run) => !evaluations.get(run).size && timing(run) != null)) selected.push(trainerTime);
  const points = (rows, metric) => rows.flatMap((run) => {
    const measurement = metric === trainerTime ? timing(run) : evaluations.get(run).get(metric);
    return measurement ? [{ run, time: time(run), ...measurement }] : [];
  });
  const initialOnly = runs.some((run) => run.planned_updates === 0)
    && runs.every((run) => run.planned_updates == null || run.planned_updates === 0);
  container.replaceChildren();
  let hasControls = false;
  for (const metric of selected) {
    const controlPoints = points(controls, metric), branchPoints = points(branches, metric);
    if (!controlPoints.length && !branchPoints.length) continue;
    hasControls ||= controlPoints.length > 0;
    const figure = document.createElement('figure');
    const label = document.createElement('figcaption');
    label.textContent = metric.replaceAll('/', ' · ').replaceAll('_', ' ');
    if (initialOnly) label.textContent += ' · initializer inference';
    figure.append(label, plot(metric, controlPoints, branchPoints, route));
    container.append(figure);
  }
  return hasControls;
}
