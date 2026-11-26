import { signals } from './contracts.js';
import { startupPhases } from './phases.js';

export const BUCKET_MS = 10_000;
export const MAX_COUNTER_GAP_MS = 15_000;
const MAX_ENGINES = 64;

function parseObject(value) {
  if (typeof value !== 'string') return value ?? {};
  try {
    return JSON.parse(value) ?? {};
  } catch {
    return {};
  }
}

function stableJson(value) {
  if (Array.isArray(value)) return '[' + value.map(stableJson).join(',') + ']';
  if (value && typeof value === 'object') {
    return '{' + Object.keys(value).sort().map((key) =>
      JSON.stringify(key) + ':' + stableJson(value[key])).join(',') + '}';
  }
  return JSON.stringify(value);
}

function seriesKey(row, attributes = row.attributes) {
  if (row.source !== undefined) return stableJson([row.source, attributes]);
  return stableJson([
    row.cluster, row.service, row.run_id, row.job_id, row.execution_uid, row.region,
    row.node_name, row.process_index, row.resource, attributes,
  ]);
}

function bySeries(rows, key) {
  const result = new Map();
  for (const row of rows) {
    const id = key(row);
    if (!result.has(id)) result.set(id, []);
    result.get(id).push(row);
  }
  return result;
}

const counterKey = (row) => row.engineSeries ?? seriesKey(row);

function byTime(a, b) {
  return a.timestamp_ms - b.timestamp_ms || a.seq - b.seq;
}

function counterIntervals(rows) {
  const atTime = new Map(rows.slice().sort(byTime).map((row) => [row.timestamp_ms, row.value]));
  const snapshots = [...atTime];
  const intervals = [];
  for (let i = 1; i < snapshots.length; i++) {
    const [start, before] = snapshots[i - 1];
    const [end, after] = snapshots[i];
    if (after >= before && end - start <= MAX_COUNTER_GAP_MS) {
      intervals.push({ start, end, rate: (after - before) * 1000 / (end - start) });
    }
  }
  return intervals;
}

function bins({ from, to }) {
  const result = [];
  for (let start = from; start < to; start += BUCKET_MS) {
    const end = Math.min(start + BUCKET_MS, to);
    result.push({ t: (start + end) / 2, start, end, value: null });
  }
  return result;
}

function overlap(a, b) {
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}

export function cumulativeRate(rows, window) {
  const groups = bySeries(rows.filter((row) => Number.isFinite(row.value)), counterKey);
  const target = bins(window);
  if (!groups.size) return target;
  const totals = target.map(() => ({ integral: 0, coverage: 0, first: Infinity, last: -Infinity }));
  for (const series of groups.values()) {
    for (const interval of counterIntervals(series)) {
      const firstBin = Math.max(0, Math.floor((interval.start - window.from) / BUCKET_MS));
      const stopBin = Math.min(target.length, Math.ceil((interval.end - window.from) / BUCKET_MS));
      for (let i = firstBin; i < stopBin; i++) {
        const duration = overlap(interval, target[i]);
        totals[i].integral += interval.rate * duration;
        totals[i].coverage += duration;
        totals[i].first = Math.min(totals[i].first, interval.start);
        totals[i].last = Math.max(totals[i].last, interval.end);
      }
    }
  }
  return target.map((bin, i) => {
    const duration = bin.end - bin.start;
    const total = totals[i];
    if (total.coverage !== duration * groups.size) return bin;
    return { ...bin, value: total.integral / duration, first: total.first, last: total.last };
  });
}

function cumulativeMean(rows, window, countName) {
  const totals = rows.filter((row) => row.name !== countName && Number.isFinite(row.value));
  const counts = rows.filter((row) => row.name === countName && Number.isFinite(row.value));
  const totalSeries = new Set(totals.map(counterKey));
  const countSeries = new Set(counts.map(counterKey));
  if (!totalSeries.size || totalSeries.size !== countSeries.size ||
      [...totalSeries].some((key) => !countSeries.has(key))) return [];

  const requestRates = cumulativeRate(counts, window);
  return cumulativeRate(totals, window).map((bin, index) => {
    const rate = requestRates[index].value;
    return { ...bin, value: rate > 0 && bin.value !== null ? bin.value / rate : null };
  });
}

export function deltaRate(samples, window) {
  const target = bins(window);
  for (const sample of samples) {
    const index = Math.floor((sample.t - window.from) / BUCKET_MS);
    const bin = target[index];
    if (!bin || sample.t >= window.to) continue;
    bin.value = (bin.value ?? 0) + sample.value;
    bin.samples = (bin.samples ?? 0) + 1;
    bin.first = Math.min(bin.first ?? sample.t, sample.t);
    bin.last = Math.max(bin.last ?? sample.t, sample.t);
  }
  for (const bin of target) {
    if (bin.value !== null) {
      bin.count = bin.value;
      bin.value *= 1000 / (bin.end - bin.start);
    }
  }
  return target;
}

function engineSnapshots(rows, window, { average, factor = 1 }) {
  const target = bins(window);
  const groups = bySeries(rows, (row) => {
    const { step, ...labels } = row.attributes;
    return row.engineSeries ?? seriesKey(row, labels);
  });
  if (!groups.size) return target;
  const totals = target.map(() => []);
  for (const series of groups.values()) {
    const latest = new Map();
    for (const row of series) {
      const index = Math.floor((row.timestamp_ms - window.from) / BUCKET_MS);
      if (!target[index] || row.timestamp_ms >= window.to) continue;
      latest.set(index, row);
    }
    for (const [index, sample] of latest) totals[index].push(sample);
  }
  return target.map((bin, i) => {
    const samples = totals[i];
    if (samples.length !== groups.size) return bin;
    const total = samples.reduce((sum, row) => sum + row.value, 0);
    return { ...bin, engines: samples.length, value: total * factor / (average ? samples.length : 1),
      first: Math.min(...samples.map((row) => row.timestamp_ms)),
      last: Math.max(...samples.map((row) => row.timestamp_ms)) };
  });
}

function concurrency(rows, window) {
  const changes = rows.flatMap(({ body }) => {
    const { started_unix_ms: start, finished_unix_ms: end } = body;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return [];
    return [{ t: start, delta: 1 }, { t: end, delta: -1 }];
  }).sort((a, b) => a.t - b.t);
  let active = 0;
  let index = 0;
  return bins(window).map((bin) => {
    if (!changes.length || bin.t < changes[0].t || bin.t >= changes.at(-1).t) return bin;
    while (index < changes.length && changes[index].t <= bin.t) active += changes[index++].delta;
    return { ...bin, value: active };
  });
}

function mergedIntervals(spans) {
  const result = [];
  for (const span of [...spans].sort((a, b) => a.start - b.start)) {
    const previous = result.at(-1);
    if (previous && span.start <= previous.end) previous.end = Math.max(previous.end, span.end);
    else result.push({ start: span.start, end: span.end });
  }
  return result;
}

export function waitShare(waitSpans, phaseSpans, window) {
  const waits = mergedIntervals(waitSpans);
  const observed = mergedIntervals(phaseSpans);
  return bins(window).map((bin) => {
    const waited = waits.reduce((ms, span) => ms + overlap(bin, span), 0);
    const covered = observed.reduce((ms, span) => ms + overlap(bin, span), 0);
    const duration = bin.end - bin.start;
    return { ...bin, value: waited || covered === duration ? waited * 100 / duration : null };
  });
}

function waitTime(waitSpans, observedSpans, window) {
  return waitShare(waitSpans, observedSpans, window).map((bin) => ({
    ...bin, value: bin.value === null ? null : bin.value * (bin.end - bin.start) / 100_000,
  }));
}

function eventDetails(row) {
  return {
    step: row.attributes.step ?? null,
    reason: row.attributes.reason ?? row.body.reason ?? null,
    outcome: row.attributes.outcome ?? null,
    duration: row.body.duration_seconds ?? null,
    recorded: row.timestamp_ms,
    ...(row.attributes.disposition ? { disposition: row.attributes.disposition } : {}),
  };
}

function field(row, path) {
  if (path.startsWith('body_json.')) return row.body[path.slice('body_json.'.length)];
  return row[path];
}

function includes(row, signal) {
  return (row.name === signal.name || (signal.countName !== undefined && row.name === signal.countName)) &&
    Object.entries(signal.filter).every(([key, value]) =>
      Array.isArray(value) ? value.includes(row.attributes[key]) : row.attributes[key] === value) &&
    (!signal.metricPattern || new RegExp(signal.metricPattern).test(row.attributes.metric ?? ''));
}

function points(rows, s, window) {
  return rows.flatMap((row) => {
    const t = field(row, s.time);
    const { started_unix_ms: start, finished_unix_ms: end } = row.body;
    const value = s.transform === 'spanDuration' ?
      (Number.isFinite(start) && Number.isFinite(end) && end >= start ? (end - start) / 1000 : null) :
      field(row, s.value ?? (s.transform === 'duration' ? 'body_json.duration_seconds' : 'value'));
    if (!Number.isFinite(value) || !Number.isFinite(t) || t < window.from || t >= window.to) return [];
    const dispatchWait = s === signals.taskWait || s === signals.leaseWait;
    return [{ t, value: value * (s.factor ?? 1), ...eventDetails(row),
      ...(dispatchWait && value >= 0 ? { waitStart: t - Math.round(value * 1000) } : {}) }];
  }).sort((a, b) => a.t - b.t);
}

function splitName(row, s) {
  if (s.metricPattern) return row.attributes.metric.match(new RegExp(s.metricPattern))?.[1];
  if (s.dimension) return row.attributes[s.dimension] ?? 'unspecified';
  if (s.facet === 'task_family_or_backend') return row.attributes.tool_backend ?? row.attributes.task_family;
}

function batchTotals(samples, rows, window) {
  const stepKey = (row) => seriesKey(row, { step: row.attributes.step });
  const groups = bySeries(samples.filter((row) => Number.isFinite(row.value)), stepKey);
  return rows.flatMap((row) => {
    if (!includes(row, signals.wait) || row.attributes.outcome !== 'success' || row.attributes.step == null) return [];
    const { started_unix_ms: start, finished_unix_ms: t } = row.body;
    if (!Number.isFinite(start) || !Number.isFinite(t) || start < window.from || t >= window.to || t < start) return [];
    const records = groups.get(stepKey(row))?.filter((item) => start <= item.timestamp_ms && item.timestamp_ms <= t);
    if (!records?.length) return [];
    return [{
      t, recorded: t, step: row.attributes.step, value: records.reduce((total, item) => total + item.value, 0),
      first: records[0].timestamp_ms, last: records.at(-1).timestamp_ms,
    }];
  }).sort((a, b) => a.t - b.t);
}

function entriesForSignal(id, s, rows, window, phases) {
  const matching = rows.filter((row) => includes(row, s));
  const entries = [];
  const add = (data, group = null, groupKey = null) => entries.push({
    id: groupKey ? id + '/' + groupKey : id,
    label: s.dimension && group ? group : s.label,
    group: s.dimension ? null : group,
    unit: s.unit,
    ...(s.distribution ? { distribution: true } : {}),
    data,
    kind: s.kind ?? (['cumulative', 'cumulativeMean', 'waitTime', 'concurrency'].includes(s.transform) ? 'line' :
      ['eventRate', 'deltaRate'].includes(s.transform) ? 'bar' : 'scatter'),
  });
  if (['cumulative', 'cumulativeMean', 'engineSnapshots'].includes(s.transform)) {
    const transform = (items) => {
      if (s.transform === 'cumulativeMean') return cumulativeMean(items, window, s.countName);
      if (s.transform === 'cumulative') return cumulativeRate(items, window);
      return engineSnapshots(items, window, s);
    };
    add(transform(matching));
    const engineGroups = bySeries(matching.filter((row) => row.attributes.engine),
      (row) => stableJson([row.attributes.engine, row.attributes.engine_index]));
    for (const [key, engineRows] of [...engineGroups].slice(0, MAX_ENGINES)) {
      const { engine, engine_index } = engineRows[0].attributes;
      const label = 'Engine ' + (engine_index ?? engine.slice(0, 8));
      add(transform(engineRows), label, key);
    }
    return entries;
  }
  const nativeRate = (samples) => deltaRate(samples.map((row) => ({
    t: field(row, s.time), value: s.transform === 'eventRate' ? 1 : field(row, s.value ?? 'value'),
  })).filter((row) => Number.isFinite(row.t) && Number.isFinite(row.value)), window);
  const transform = ['eventRate', 'deltaRate'].includes(s.transform) ? nativeRate
    : (samples) => points(samples, s, window);
  if (s.transform === 'waitTime') {
    add(waitTime(phases.filter((phase) => phase.kind === 'rollout_wait'), phases, window));
  } else if (s.transform === 'concurrency') {
    add(concurrency(matching, window));
  } else if (s.transform === 'batchTotal') {
    add(batchTotals(matching, rows, window));
  } else if (s.transform === 'unavailable') {
    add([]);
  } else if (s.dimension || s.metricPattern) {
    const named = bySeries(matching, (row) => splitName(row, s));
    if (s.facet === 'eval_suite') named.delete('all');
    if (!named.size) add([]);
    for (const [name, samples] of named) {
      add(transform(samples), name, name);
    }
  } else {
    const facets = matching.filter((row) => splitName(row, s));
    add(transform(matching));
    for (const [name, samples] of bySeries(facets, (row) => splitName(row, s))) {
      add(transform(samples), name, name);
    }
  }
  return entries;
}

const phaseLabel = { rollout_wait: 'Trainer batch wait', training: 'Training', weight_sync: 'Weight sync' };

function dispatchWaitTimes(rows, window) {
  const waits = [signals.leaseWait, signals.taskWait].map((signal) =>
    rows.filter((row) => Number.isFinite(row.value) && row.value >= 0 && includes(row, signal)));
  const owners = new Set(waits.flat().map((row) => seriesKey(row, {})));
  if (owners.size !== 1) return {};
  // The dispatcher publishes these durations upon leaving each await.
  const spans = waits.map((samples) => samples.map((row) => ({
    start: row.timestamp_ms - Math.round(row.value * 1000), end: row.timestamp_ms,
  })));
  return Object.fromEntries(['leaseWaitTime', 'taskWaitTime'].map((id, index) => [id, [{
    id, label: [signals.leaseWait, signals.taskWait][index].label,
    unit: 's per 10 s', group: null, kind: 'line', data: waitTime(spans[index], spans.flat(), window),
  }]]));
}

function evaluationSummaries(rows, window) {
  const fields = new Map([
    [signals.evalGraded.filter.metric, 'graded'],
    [signals.evalAttempted.filter.metric, 'attempted'],
    [signals.evalFailed.filter.metric, 'submissionFailures'],
  ]);
  const samples = rows.filter((row) => row.name === 'training_metric_value' && fields.has(row.attributes.metric)
    && row.attributes.step != null && Number.isFinite(row.value));
  return [...bySeries(samples, (row) => seriesKey(row, { step: row.attributes.step, kind: row.attributes.payload_kind })).values()]
    .flatMap((group) => {
      const record = {};
      for (const row of group) {
        const fieldName = fields.get(row.attributes.metric);
        if (fieldName in record) return [];
        record[fieldName] = row.value;
      }
      const last = group.at(-1);
      if (record.graded == null || last.timestamp_ms < window.from || last.timestamp_ms >= window.to) return [];
      return [{ ...record, t: last.timestamp_ms, step: last.attributes.step }];
    }).sort((a, b) => a.t - b.t);
}

function attachEntryOccupancy(rows, points, window, { startMs, endMs, trainingRows }) {
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || window.from > startMs ||
      window.to <= endMs || !Number.isSafeInteger(trainingRows) || trainingRows < 1) return;
  const attempts = rows.filter((row) => row.name === 'rollout_call' && Number.isFinite(row.body.started_unix_ms));
  const exits = rows.filter((row) => row.name === 'rollout_groups' && Number.isFinite(row.value));
  if (!attempts.length || !exits.length || new Set([...attempts, ...exits].map((row) => seriesKey(row, {}))).size !== 1) return;

  // Dispatch marks a UID live at launch. The Buffer's dispositions release it.
  const changes = [
    ...attempts.map((row) => ({ t: row.body.started_unix_ms, delta: 1 })),
    ...exits.map((row) => ({ t: row.timestamp_ms, delta: -row.value })),
  ].sort((a, b) => a.t - b.t);
  const waits = points.filter((point) => point.waitStart != null).sort((a, b) => a.waitStart - b.waitStart);
  let index = 0;
  let outstanding = 0;
  for (const point of waits) {
    while (index < changes.length && changes[index].t <= point.waitStart) outstanding += changes[index++].delta;
    if (outstanding >= 0 && outstanding <= trainingRows) point.outstanding = outstanding;
  }
}

export function createMeasurements(rawRows, window, configRows = [], inferenceGpus = null, context = {}) {
  const decode = (row) => ({
    ...row, attributes: parseObject(row.attributes_json),
    body: parseObject(row.body_json), resource: parseObject(row.resource_attributes_json),
  });
  const rows = rawRows.map(decode).sort(byTime);
  const phases = rows.filter((row) => row.name === 'async_phase_window').flatMap((row) => {
    const { started_unix_ms: start, finished_unix_ms: end } = row.body;
    const kind = row.attributes.phase;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start || !phaseLabel[kind]) return [];
    return [{ id: 'phase/' + row.seq, start, end, kind, label: phaseLabel[kind], ...eventDetails(row) }];
  });
  const events = rows.flatMap((row) => {
    const kind = includes(row, signals.optimizer) ? 'step_complete'
      : includes(row, signals.sync) ? 'sync_complete' : null;
    const label = kind === 'step_complete' ? 'Optimizer completion' : 'Weight sync completed';
    return kind && row.timestamp_ms >= window.from && row.timestamp_ms < window.to
      ? [{ id: 'event/' + row.seq, t: row.timestamp_ms, kind, label, ...eventDetails(row) }] : [];
  });
  const series = {};
  for (const [id, signal] of Object.entries(signals)) {
    if (signal.transform !== 'event' && signal.transform !== 'config') {
      series[id] = entriesForSignal(id, signal, rows, window, phases);
    }
  }
  attachEntryOccupancy(rows, series.taskWait[0].data, window, context);
  if (Number.isSafeInteger(inferenceGpus) && inferenceGpus > 0) {
    for (const id of ['promptTokens', 'generation']) {
      const [total] = series[id];
      const normalized = id + 'PerGpu';
      series[normalized] = [{
        ...total, id: normalized, unit: 'tokens/s/GPU',
        data: total.data.map((point) => ({
          ...point, value: point.value === null ? null : point.value / inferenceGpus,
        })),
      }];
    }
  }
  const config = configRows.map(decode).find((row) => includes(row, signals.batchTarget));
  const batchTarget = config ? field(config, signals.batchTarget.value) : null;
  const maxStaleness = config?.body.max_staleness_steps;
  const configFields = {
    outstandingLeaseLimit: 'generation_workers', policyNodes: 'policy_nodes', policyGpusPerNode: 'policy_gpus_per_node',
    policyTp: 'policy_tp', policyPp: 'policy_pp', policyCp: 'policy_cp', policyEp: 'policy_ep',
  };
  const configuration = Object.fromEntries(Object.entries(configFields).flatMap(([key, fieldName]) => {
    const value = config?.body[fieldName];
    return Number.isSafeInteger(value) && value > 0 ? [[key, value]] : [];
  }));
  const strategy = config?.body.strategy;
  if (typeof strategy === 'string' && strategy.trim() && strategy.length <= 64) configuration.strategy = strategy.trim();
  const initial = rows.find((row) => includes(row, signals.sync) && row.attributes.reason === 'initial');
  const supplemental = [...startupPhases(context.startMs, initial && eventDetails(initial)), ...(context.driverPhases ?? [])];
  return {
    series: { ...series, ...dispatchWaitTimes(rows, window) },
    phases: [...phases, ...supplemental].sort((a, b) => a.start - b.start),
    events, evaluations: evaluationSummaries(rows, window), configuration,
    batchTarget: Number.isFinite(batchTarget) ? batchTarget : null,
    maxStaleness: Number.isFinite(maxStaleness) ? maxStaleness : null,
  };
}
