const SVG = 'http://www.w3.org/2000/svg';
const controlPath = /\/control\/[^/]+-([a-f0-9]{7,40})-r\d+(?:-retry\d+)?\/([^/]+)\/?$/;
const evaluationMetric = /^([^@]+)@(\d+)\/(.+)$/;
const comparableVerdicts = new Set(['pass', 'fail']);
const score = (value) => `${(100 * value).toFixed(1)}%`;
const time = (run) => run.completed_at ?? run.started_at ?? run.run_date ?? '';
const sameCommit = (a, b) => a && b && (a.startsWith(b) || b.startsWith(a));

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

function controlSets(runs) {
  const sets = new Map();
  for (const run of runs) {
    const match = run.artifact_uri?.match(controlPath);
    if (!match) continue;
    const [, commit, version] = match;
    const revision = run.preset_revision ?? version;
    const key = `${revision}/${commit}`;
    if (!sets.has(key)) sets.set(key, { revision, commit, rows: [] });
    sets.get(key).rows.push(run);
  }
  return [...sets.values()].flatMap((set) => {
    const report = set.rows.find((run) => run.gate_result?.attempt_ids.includes(run.attempt_id))?.gate_result;
    const latestByOutput = new Map();
    for (const run of [...set.rows].sort((a, b) => time(b).localeCompare(time(a)))) {
      if (run.status !== 'succeeded' || latestByOutput.has(run.artifact_uri)) continue;
      if (report?.attempt_ids.length && !report.attempt_ids.includes(run.attempt_id)) continue;
      if (!report && (run.dirty || (run.branch && run.branch !== 'main' && !sameCommit(run.source_revision, set.commit)))) continue;
      latestByOutput.set(run.artifact_uri, run);
    }
    const rows = [...latestByOutput.values()];
    if (!rows.length) return [];
    const [latest] = rows;
    return [{ ...set, rows, report, time: time(latest), data: latest.evaluation_data_identity,
      protocol: latest.evaluation_protocol_identity }];
  }).sort((a, b) => a.time.localeCompare(b.time));
}

function comparison(report, metric) {
  if (!comparableVerdicts.has(report?.verdict)) return null;
  const row = report.comparisons.find((item) => item.metric === metric);
  if (!row) return null;
  if (Number.isFinite(row.delta)) return { delta: row.delta, tolerance: row.tolerance };
  return null;
}

function controlPoints(sets, scores, metric) {
  const points = sets.flatMap((set) => {
    const values = set.rows.map((run) => scores.get(run).get(metric)?.value).filter(Number.isFinite);
    if (!values.length) return [];
    return [{ ...set, global: values.reduce((total, value) => total + value, 0) / values.length,
      min: Math.min(...values), max: Math.max(...values) }];
  });
  for (const [index, point] of points.entries()) {
    const previous = points[index - 1];
    if (!previous) { point.segment = 0; continue; }
    const protocolChanged = (point.protocol && previous.protocol && point.protocol !== previous.protocol)
      || point.report?.reason === 'evaluation protocol differs';
    const dataChanged = point.data && previous.data && point.data !== previous.data;
    point.marker = protocolChanged ? 'Protocol changed' : dataChanged ? 'Evaluation set changed' : null;
    const shared = point.report?.invariants.some((fact) => fact.name === `${metric.split('/')[0]}/shared_tasks` && fact.holds);
    if (shared && !protocolChanged && point.report.control_revision === previous.revision
      && sameCommit(point.report.control_commit, previous.commit)) {
      point.link = comparison(point.report, metric);
    }
    point.segment = previous.segment + (point.link ? 0 : 1);
  }
  for (let index = points.length - 1; index >= 0; index--) {
    const next = points[index + 1];
    points[index].chained = next?.link ? next.chained - next.link.delta : points[index].global;
  }
  return points;
}

function branchPoints(runs, scores, controls, metric) {
  const groups = new Set();
  return runs.flatMap((run) => {
    const global = scores.get(run).get(metric)?.value;
    if (!Number.isFinite(global) || controlPath.test(run.artifact_uri ?? '')) return [];
    const report = run.gate_result;
    const matching = controls.filter((control) => control.revision === run.base_preset_revision);
    const base = matching.find((control) => sameCommit(report?.control_commit, control.commit)) ?? matching.at(-1);
    const gateControl = controls.find((control) => report?.control_revision === control.revision
      && sameCommit(report.control_commit, control.commit));
    const group = report?.attempt_ids.length > 1 ? JSON.stringify([...report.attempt_ids].sort()) : null;
    const delta = base && gateControl?.segment === base.segment && !groups.has(group) ? comparison(report, metric) : null;
    if (delta && group) groups.add(group);
    return [{ run, global, time: base?.time ?? time(run), chained: delta ? gateControl.chained + delta.delta : null,
      tolerance: delta?.tolerance, replicateCount: delta && group ? report.attempt_ids.length : 1 }];
  }).sort((a, b) => time(a.run).localeCompare(time(b.run)));
}

function svg(tag, attributes, label) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label != null) node.textContent = label;
  return node;
}

function plot(metric, controls, branches, route) {
  const chart = svg('svg', { viewBox: '0 0 680 200', role: 'group', 'aria-label': metric, class: 'progress-chart' });
  const values = [...controls.flatMap((row) => [row.min, row.max, row.chained]),
    ...branches.flatMap((row) => [row.global, row.chained, row.chained == null ? null : row.chained + (row.tolerance ?? 0),
      row.chained == null ? null : row.chained - (row.tolerance ?? 0)])].filter(Number.isFinite);
  const low = Math.min(...values), high = Math.max(...values), padding = Math.max(.025, (high - low) * .12);
  const y = (value) => 160 - 130 * (value - low + padding) / (high - low + 2 * padding);
  const times = [...new Set([...controls, ...branches].map((row) => row.time))].sort();
  const positions = new Map(times.map((value, index) => [value, index]));
  const x = (at) => times.length === 1 ? 346 : 68 + 555 * positions.get(at) / (times.length - 1);
  const line = (index, a, end, b, className, label) => {
    const node = svg('line', { x1: x(index), y1: y(a), x2: x(end), y2: y(b), class: className });
    if (label) node.append(svg('title', {}, label));
    chart.append(node);
  };
  for (const value of new Set([low, (low + high) / 2, high])) {
    chart.append(svg('line', { x1: 55, x2: 650, y1: y(value), y2: y(value), class: 'history-grid' }),
      svg('text', { x: 50, y: y(value) + 3, 'text-anchor': 'end' }, score(value)));
  }
  for (const [index, point] of controls.entries()) {
    const previous = controls[index - 1];
    if (previous) line(previous.time, previous.global, point.time, point.global, 'history-global');
    if (point.link) line(previous.time, previous.chained, point.time, point.chained, 'history-chain',
      Number.isFinite(point.link.tolerance) ? `Shared-task tolerance ±${score(point.link.tolerance)}` : null);
    if (point.marker) {
      const marker = svg('line', { x1: x(point.time), y1: 20, x2: x(point.time), y2: 165, class: 'history-marker',
        'aria-label': point.marker });
      marker.append(svg('title', {}, `${point.marker}${point.data ? ` · ${point.data}` : ''}`));
      chart.append(marker, svg('text', { x: x(point.time) - 4, y: 17, 'text-anchor': 'end' }, point.marker));
    }
    line(point.time, point.min, point.time, point.max, 'history-global', `Replicate range ${score(point.min)}–${score(point.max)}`);
    const markers = [[point.global, 'global']];
    if (point.link || controls[index + 1]?.link) markers.push([point.chained, 'chain']);
    for (const [value, style] of markers) {
      const dot = svg('circle', { cx: x(point.time), cy: y(value), r: 3.5, class: `history-${style}` });
      dot.append(svg('title', {}, `${point.revision} · ${style === 'chain' ? 'Comparable' : 'Global'} ${score(value)}`));
      chart.append(dot);
    }
    if (index === 0 || index === controls.length - 1 || controls.length <= 5) {
      chart.append(svg('text', { x: x(point.time), y: 181, 'text-anchor': 'middle' }, point.revision));
    }
  }
  let previousMain;
  for (const point of branches) {
    const value = point.chained ?? point.global;
    const onMain = !controls.length && point.run.branch === 'main' && !point.run.dirty;
    if (onMain) {
      if (previousMain) line(previousMain.time, previousMain.value, point.time, value, 'history-global');
      previousMain = { time: point.time, value };
    }
    if (point.chained != null && Number.isFinite(point.tolerance)) {
      line(point.time, value - point.tolerance, point.time, value + point.tolerance, 'history-branch');
    }
    const detail = point.replicateCount > 1 ? ` · ${point.replicateCount} replicate mean` : '';
    const link = svg('a', { href: route(point.run), 'aria-label': `${point.run.label} · ${score(value)}${detail}` });
    link.append(svg('circle', { cx: x(point.time), cy: y(value), r: 4.5, class: onMain ? 'history-global' : 'history-branch' }),
      svg('title', {}, `${point.run.branch ?? point.run.label} · ${score(value)} · ${point.run.base_preset_revision ?? 'Base unavailable'}`));
    if (point.chained != null && point.chained !== point.global) {
      const dot = svg('circle', { cx: x(point.time), cy: y(point.global), r: 3, class: 'history-global' });
      dot.append(svg('title', {}, `Experiment global ${score(point.global)}`));
      link.append(dot);
    }
    chart.append(link);
  }
  return chart;
}

export function showProgress(container, runs, reference, route) {
  const scores = new Map([...runs, ...reference].map((run) => [run, metrics(run)]));
  const selected = [...new Set([...scores.values()].flatMap((values) => [...values.keys()]))]
    .filter((key) => /^([^/]+)\/(family_mean|family\/[^/]+\/mean|mean)$/.test(key));
  const metricsToShow = selected.filter((key) => !/^[^/]+\/mean$/.test(key)
    || !selected.includes(key.replace(/\/mean$/, '/family_mean')));
  const controls = controlSets([...runs, ...reference]);
  const initialOnly = runs.some((run) => run.planned_updates === 0)
    && runs.every((run) => run.planned_updates == null || run.planned_updates === 0);
  container.replaceChildren();
  for (const metric of metricsToShow.sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b))) {
    const points = controlPoints(controls, scores, metric);
    const branches = branchPoints(runs, scores, points, metric);
    if (!points.length && !branches.length) continue;
    const figure = document.createElement('figure');
    const label = document.createElement('figcaption');
    label.textContent = metric.replaceAll('/', ' · ').replaceAll('_', ' ');
    if (initialOnly) label.textContent += ' · initializer inference';
    figure.append(label, plot(metric, points, branches, route));
    container.append(figure);
  }
}
