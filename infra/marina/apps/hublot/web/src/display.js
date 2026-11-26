import { cards, nodeSettings } from './contracts.js';

export const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 });
const compactNumber = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1, notation: 'compact' });

function shortValue(value) {
  return Math.abs(value) >= 1000 ? compactNumber.format(value) : number.format(value);
}

export function compact(value, unit) {
  return shortValue(value) + (unit === '%' ? '%' : ' ' + unit);
}

export function utc(t) {
  return new Date(t).toISOString().replace('T', ' ').replace('Z', ' UTC');
}

export function clock(t, exact = false) {
  return new Date(t).toISOString().slice(11, exact ? 23 : 19);
}

export function elapsed(t, origin) {
  const seconds = Math.floor(Math.abs(t - origin) / 1000);
  const sign = t < origin ? '−' : '';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  const rest = String(seconds % 60).padStart(2, '0');
  return sign + (hours ? hours + ':' + String(minutes).padStart(2, '0') : minutes) + ':' + rest;
}

export function shortName(runId) {
  return runId?.split('/').filter(Boolean).at(-1) ?? 'Execution';
}

export function cardSeries(id, series, includeContext = true) {
  const { series: direct, context = [] } = cards[id];
  const names = includeContext ? [...direct, ...context] : direct;
  return names.flatMap((name) => series[name] ?? []);
}

export function inRange(point, range) {
  return point.t >= range.from && point.t < range.to;
}

export function hasValues(series, range) {
  return series.data.some((point) => Number.isFinite(point.value) && inRange(point, range));
}

export function entryOccupancyAtLongestWait(data, range) {
  let longest;
  for (const wait of data) {
    if (inRange(wait, range) && (!longest || wait.value > longest.value)) longest = wait;
  }
  return longest?.outstanding == null ? null : longest;
}

export function canceledGroupJobCount(data, range) {
  let count = null;
  for (const job of data) {
    if (!inRange(job, range) || job.outcome == null) continue;
    count ??= 0;
    if (job.outcome === 'cancelled') count++;
  }
  return count;
}

export function cardsWithSamples(model) {
  return new Set(Object.keys(cards).filter((id) =>
    cardSeries(id, model.series).some((signal) => hasValues(signal, model.window))));
}

export function recordedNodeSettings(id, model) {
  const { fields = [] } = nodeSettings[id];
  const values = {
    ...model.run, ...model.configuration, batchTarget: model.batchTarget, maxStaleness: model.maxStaleness,
  };
  return fields.flatMap(([key, label]) => {
    const value = values[key];
    return value == null || value === '' ? [] : [{ key, label, value }];
  });
}

const durationBuckets = [
  [.01, 'Up to 10 ms'], [.1, '10–100 ms'], [1, '0.1–1 s'], [5, '1–5 s'], [15, '5–15 s'],
  [30, '15–30 s'], [60, '30–60 s'], [180, '1–3 min'], [600, '3–10 min'], [Infinity, 'Over 10 min'],
];

export function timingDistribution(data, range) {
  const values = data.filter((point) => inRange(point, range) && Number.isFinite(point.value) && point.value >= 0)
    .map((point) => point.value).sort((a, b) => a - b);
  if (!values.length) return null;
  const buckets = durationBuckets.map(([maximum, label]) => ({ maximum, label, count: 0 }));
  let index = 0;
  for (const value of values) {
    while (value > buckets[index].maximum) index++;
    buckets[index].count++;
  }
  const percentile = (fraction) => values[Math.ceil(values.length * fraction) - 1];
  return {
    count: values.length, p50: percentile(.5), p95: percentile(.95),
    buckets: buckets.slice(buckets.findIndex((bucket) => bucket.count), index + 1),
  };
}

export function sampleAt(data, t, tolerance) {
  let nearest;
  for (const point of data) {
    if ('start' in point) {
      if (point.start <= t && t < point.end) return point;
    } else if (Math.abs(point.t - t) <= tolerance &&
        (!nearest || Math.abs(point.t - t) < Math.abs(nearest.t - t))) {
      nearest = point;
    }
  }
  return nearest;
}

export function summary(id, series, range, cursor) {
  for (const signal of cardSeries(id, series, false).filter((s) => !s.group)) {
    const inWindow = signal.data.filter((point) => inRange(point, range));
    const current = cursor === null ? inWindow.findLast((point) => point.value !== null)
      : sampleAt(inWindow, cursor, (range.to - range.from) / 115);
    if (current?.value !== null && current?.value !== undefined) {
      return {
        value: compact(current.value, signal.unit), count: shortValue(current.value),
        label: signal.label.replaceAll('_', ' '), unit: signal.unit,
      };
    }
  }
  return null;
}
