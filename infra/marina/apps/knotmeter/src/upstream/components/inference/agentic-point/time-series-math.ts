// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import type { RequestChartRecord } from "@/hooks/api/use-request-chart-data";
import type { TimeSeriesPoint } from "@/hooks/api/use-trace-server-metrics";

export interface ChartSeries {
  name: string;
  data: TimeSeriesPoint[];
  rawData?: TimeSeriesPoint[];
  color: string;
  strokeWidth?: number;
  strokeOpacity?: number;
  hideFromHover?: boolean;
}

export function maxTimeSeriesValue(
  series: readonly ChartSeries[],
  initialValue = Number.NEGATIVE_INFINITY,
): number {
  let maximum = initialValue;
  for (const chartSeries of series) {
    for (const point of chartSeries.data)
      maximum = Math.max(maximum, point.value);
  }
  return maximum;
}

export type RequestMetric = "interactivity" | "ttft" | "e2e";
export type RequestPercentile = "p75" | "p90";
export type ThroughputSeriesKey = "input" | "decode";

export function toggleThroughputSeries(
  selected: ReadonlySet<ThroughputSeriesKey>,
  key: ThroughputSeriesKey,
): ReadonlySet<ThroughputSeriesKey> {
  if (selected.has(key) && selected.size === 1) return selected;
  const next = new Set(selected);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}

export function quantile(sortedAsc: number[], q: number): number {
  if (sortedAsc.length === 1) return sortedAsc[0]!;
  const pos = (sortedAsc.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedAsc[lo]!;
  return sortedAsc[lo]! + (sortedAsc[hi]! - sortedAsc[lo]!) * (pos - lo);
}

export function interpAt(data: TimeSeriesPoint[], t: number): number | null {
  if (data.length === 0) return null;
  if (t <= data[0]!.t) return data[0]!.value;
  if (t >= data.at(-1)!.t) return data.at(-1)!.value;
  let lo = 0;
  let hi = data.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (data[mid]!.t <= t) lo = mid;
    else hi = mid;
  }
  const a = data[lo]!;
  const b = data[hi]!;
  if (b.t === a.t) return a.value;
  const frac = (t - a.t) / (b.t - a.t);
  return a.value + (b.value - a.value) * frac;
}

export function rollingRequestMetric(
  requests: readonly RequestChartRecord[],
  metric: RequestMetric,
  percentile: RequestPercentile,
  windowSize = 50,
): {
  raw: TimeSeriesPoint[];
  trend: TimeSeriesPoint[];
  cumulative: TimeSeriesPoint[];
} {
  const q = percentile === "p75" ? 0.75 : 0.9;
  const samples = requests
    .filter((request) => !request.cancelled)
    .flatMap((request) => {
      const latencyMs =
        metric === "ttft"
          ? request.ttftMs
          : metric === "e2e"
            ? (request.end - request.start) / 1e6
            : request.tpotMs;
      if (latencyMs === null || !Number.isFinite(latencyMs) || latencyMs <= 0)
        return [];
      return [{ t: request.end / 1e9, latencyMs }];
    })
    .toSorted((a, b) => a.t - b.t);

  const raw = samples.map(({ t, latencyMs }) => ({
    t,
    value: metric === "interactivity" ? 1000 / latencyMs : latencyMs / 1000,
  }));
  const trend = samples.map(({ t }, i) => {
    const start = Math.max(0, i - Math.max(1, windowSize) + 1);
    const sorted = samples
      .slice(start, i + 1)
      .map((sample) => sample.latencyMs)
      .toSorted((a, b) => a - b);
    const latencyMs = quantile(sorted, q);
    return {
      t,
      value: metric === "interactivity" ? 1000 / latencyMs : latencyMs / 1000,
    };
  });

  const latencyValues = [
    ...new Set(samples.map(({ latencyMs }) => latencyMs)),
  ].toSorted((a, b) => a - b);
  const latencyIndex = new Map(
    latencyValues.map((value, index) => [value, index + 1]),
  );
  const counts = new Uint32Array(latencyValues.length + 1);
  const addLatency = (index: number) => {
    for (
      let cursor = index;
      cursor < counts.length;
      cursor += cursor & -cursor
    ) {
      counts[cursor]! += 1;
    }
  };
  const kthLatency = (zeroBasedRank: number): number => {
    let index = 0;
    let target = zeroBasedRank + 1;
    let step = 1;
    while (step * 2 < counts.length) step *= 2;
    for (; step > 0; step >>= 1) {
      const next = index + step;
      if (next < counts.length && counts[next]! < target) {
        index = next;
        target -= counts[next]!;
      }
    }
    return latencyValues[index]!;
  };
  const cumulative = samples.map(({ t, latencyMs }, index) => {
    addLatency(latencyIndex.get(latencyMs)!);
    const count = index + 1;
    const position = (count - 1) * q;
    const lowRank = Math.floor(position);
    const highRank = Math.ceil(position);
    const low = kthLatency(lowRank);
    const high = kthLatency(highRank);
    const cumulativeLatencyMs = low + (high - low) * (position - lowRank);
    return {
      t,
      value:
        metric === "interactivity"
          ? 1000 / cumulativeLatencyMs
          : cumulativeLatencyMs / 1000,
    };
  });

  return { raw, trend, cumulative };
}

export function timeRollingAverage(
  data: TimeSeriesPoint[],
  windowS: number,
): TimeSeriesPoint[] {
  if (data.length === 0 || windowS <= 0) return data;
  const areaAt = new Float64Array(data.length);
  areaAt[0] = data[0]!.value * data[0]!.t;
  for (let index = 1; index < data.length; index += 1) {
    const previous = data[index - 1]!;
    const current = data[index]!;
    areaAt[index] =
      areaAt[index - 1]! + previous.value * (current.t - previous.t);
  }
  const out: TimeSeriesPoint[] = Array.from({ length: data.length });
  let left = 0;
  for (let i = 0; i < data.length; i++) {
    const tEnd = data[i]!.t;
    const tStart = Math.max(0, tEnd - windowS);
    while (left + 1 <= i && data[left + 1]!.t <= tStart) left += 1;
    const areaAtStart =
      tStart <= data[0]!.t
        ? data[0]!.value * tStart
        : areaAt[left]! + data[left]!.value * (tStart - data[left]!.t);
    const area = areaAt[i]! - areaAtStart;
    const dur = tEnd - tStart;
    out[i] = { t: tEnd, value: dur > 0 ? area / dur : data[i]!.value };
  }
  return out;
}

export function rollingAverage(
  data: TimeSeriesPoint[],
  windowSize: number,
): TimeSeriesPoint[] {
  if (data.length === 0 || windowSize <= 1) return data;
  const half = Math.floor(windowSize / 2);
  const out: TimeSeriesPoint[] = Array.from({ length: data.length });
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - half);
    const end = Math.min(data.length, i + half + 1);
    let sum = 0;
    let n = 0;
    for (let j = start; j < end; j++) {
      sum += data[j]!.value;
      n++;
    }
    out[i] = { t: data[i]!.t, value: n > 0 ? sum / n : 0 };
  }
  return out;
}

export function rollingRatioOfSums(
  numerator: TimeSeriesPoint[],
  denominator: TimeSeriesPoint[],
  windowSize: number,
  upperBound = 1,
): TimeSeriesPoint[] {
  if (denominator.length === 0 || windowSize <= 0) return [];

  const numeratorByT = new Map<number, number>();
  for (const point of numerator) {
    if (!Number.isFinite(point.t) || !Number.isFinite(point.value)) continue;
    numeratorByT.set(
      point.t,
      (numeratorByT.get(point.t) ?? 0) + Math.max(0, point.value),
    );
  }

  const rows: { t: number; numerator: number; denominator: number }[] = [];
  for (const point of denominator) {
    if (!Number.isFinite(point.t) || !Number.isFinite(point.value)) continue;
    const previous = rows.at(-1);
    if (previous?.t === point.t) {
      previous.denominator += Math.max(0, point.value);
      continue;
    }
    rows.push({
      t: point.t,
      numerator: numeratorByT.get(point.t) ?? 0,
      denominator: Math.max(0, point.value),
    });
  }
  if (rows.length === 0) return [];

  const numeratorPrefix = new Float64Array(rows.length + 1);
  const denominatorPrefix = new Float64Array(rows.length + 1);
  for (let i = 0; i < rows.length; i++) {
    numeratorPrefix[i + 1] = numeratorPrefix[i]! + rows[i]!.numerator;
    denominatorPrefix[i + 1] = denominatorPrefix[i]! + rows[i]!.denominator;
  }

  const leftSpan = Math.floor((windowSize - 1) / 2);
  const rightSpan = windowSize - leftSpan - 1;
  const boundedUpper = Number.isFinite(upperBound)
    ? Math.max(0, upperBound)
    : Infinity;
  const out: TimeSeriesPoint[] = [];
  for (let i = 0; i < rows.length; i++) {
    const start = Math.max(0, i - leftSpan);
    const end = Math.min(rows.length, i + rightSpan + 1);
    const denominatorSum = denominatorPrefix[end]! - denominatorPrefix[start]!;
    if (denominatorSum <= 0) continue;
    const numeratorSum = numeratorPrefix[end]! - numeratorPrefix[start]!;
    out.push({
      t: rows[i]!.t,
      value: Math.min(boundedUpper, Math.max(0, numeratorSum / denominatorSum)),
    });
  }
  return out;
}

export function rollingRatioFromComponents(
  ratio: TimeSeriesPoint[],
  numerator: TimeSeriesPoint[],
  zeroRatioDenominator: TimeSeriesPoint[],
  windowSize: number,
  upperBound = 1,
): TimeSeriesPoint[] {
  const ratioByT = new Map<number, number>();
  for (const point of ratio) {
    if (
      !Number.isFinite(point.t) ||
      !Number.isFinite(point.value) ||
      point.value < 0
    )
      continue;
    ratioByT.set(point.t, point.value);
  }
  const numeratorByT = new Map<number, number>();
  for (const point of numerator) {
    if (!Number.isFinite(point.t) || !Number.isFinite(point.value)) continue;
    numeratorByT.set(
      point.t,
      (numeratorByT.get(point.t) ?? 0) + Math.max(0, point.value),
    );
  }
  const fallbackByT = new Map(
    zeroRatioDenominator.map((point) => [point.t, Math.max(0, point.value)]),
  );
  const recoveredNumerator: TimeSeriesPoint[] = [];
  const recoveredDenominator: TimeSeriesPoint[] = [];

  const timeline: number[] = [];
  let ratioIndex = 0;
  let numeratorIndex = 0;
  while (ratioIndex < ratio.length || numeratorIndex < numerator.length) {
    while (ratioIndex < ratio.length && !Number.isFinite(ratio[ratioIndex]!.t))
      ratioIndex++;
    while (
      numeratorIndex < numerator.length &&
      !Number.isFinite(numerator[numeratorIndex]!.t)
    ) {
      numeratorIndex++;
    }
    const ratioT = ratio[ratioIndex]?.t ?? Infinity;
    const numeratorT = numerator[numeratorIndex]?.t ?? Infinity;
    const t = Math.min(ratioT, numeratorT);
    if (!Number.isFinite(t)) break;
    if (timeline.at(-1) !== t) timeline.push(t);
    while (ratio[ratioIndex]?.t === t) ratioIndex++;
    while (numerator[numeratorIndex]?.t === t) numeratorIndex++;
  }

  for (const t of timeline) {
    const pointRatio = ratioByT.get(t);
    const observedNumerator = numeratorByT.get(t) ?? 0;
    if (pointRatio === undefined) {
      if (observedNumerator > 0) {
        recoveredNumerator.push({ t, value: observedNumerator });
        recoveredDenominator.push({ t, value: 0 });
      }
      continue;
    }

    const fallbackDenominator = fallbackByT.get(t) ?? 0;
    const denominator =
      pointRatio > 0 && observedNumerator > 0
        ? observedNumerator / pointRatio
        : fallbackDenominator;
    if (!Number.isFinite(denominator) || denominator <= 0) continue;

    recoveredDenominator.push({ t, value: denominator });
    recoveredNumerator.push({
      t,
      value:
        observedNumerator > 0 ? observedNumerator : pointRatio * denominator,
    });
  }

  return rollingRatioOfSums(
    recoveredNumerator,
    recoveredDenominator,
    windowSize,
    upperBound,
  );
}

export function cumulativeAverage(
  data: TimeSeriesPoint[],
  burnInS = 0,
): TimeSeriesPoint[] {
  if (data.length === 0) return data;
  const out: TimeSeriesPoint[] = [];
  const firstT = data[0]!.t;
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i]!.value;
    if (data[i]!.t - firstT >= burnInS) {
      out.push({ t: data[i]!.t, value: sum / (i + 1) });
    }
  }
  return out;
}

export function cumulativeTimeAverage(
  data: TimeSeriesPoint[],
): TimeSeriesPoint[] {
  if (data.length === 0) return [];
  const points: TimeSeriesPoint[] = [];
  for (const point of data.toSorted((a, b) => a.t - b.t)) {
    if (!Number.isFinite(point.t) || !Number.isFinite(point.value)) continue;
    const previous = points.at(-1);
    if (previous?.t === point.t) previous.value = point.value;
    else points.push({ ...point });
  }
  if (points.length === 0) return [];

  const firstT = points[0]!.t;
  let previousT = firstT;
  let previousValue = points[0]!.value;
  let area = 0;
  return points.map((point, index) => {
    if (index === 0) return { t: point.t, value: point.value };
    area += previousValue * (point.t - previousT);
    const duration = point.t - firstT;
    previousT = point.t;
    previousValue = point.value;
    return { t: point.t, value: duration > 0 ? area / duration : point.value };
  });
}

export function cumulativeCompletedRequests(
  requests: readonly RequestChartRecord[],
): TimeSeriesPoint[] {
  const completionTimes = requests
    .filter((request) => !request.cancelled)
    .map((request) => request.end / 1e9)
    .filter(Number.isFinite)
    .toSorted((a, b) => a - b);
  if (completionTimes.length === 0) return [];
  return [
    { t: 0, value: 0 },
    ...completionTimes.map((t, index) => ({ t, value: index + 1 })),
  ];
}

export function averageSequenceLengthInFlight(
  requests: readonly RequestChartRecord[],
  metric: "isl" | "osl",
): TimeSeriesPoint[] {
  const events = new Map<number, { tokenDelta: number; countDelta: number }>();
  const addEvent = (t: number, tokenDelta: number, countDelta: number) => {
    const current = events.get(t) ?? { tokenDelta: 0, countDelta: 0 };
    current.tokenDelta += tokenDelta;
    current.countDelta += countDelta;
    events.set(t, current);
  };

  for (const request of requests) {
    const tokens = request[metric];
    if (
      request.cancelled ||
      tokens === null ||
      !Number.isFinite(tokens) ||
      tokens < 0 ||
      request.end < request.start
    ) {
      continue;
    }
    addEvent(request.start / 1e9, tokens, 1);
    addEvent(request.end / 1e9, -tokens, -1);
  }

  let tokensInFlight = 0;
  let requestsInFlight = 0;
  return [...events.entries()]
    .toSorted((a, b) => a[0] - b[0])
    .map(([t, event]) => {
      tokensInFlight += event.tokenDelta;
      requestsInFlight += event.countDelta;
      return {
        t,
        value: requestsInFlight > 0 ? tokensInFlight / requestsInFlight : 0,
      };
    });
}

const CACHE_SOURCE_RE = /cache|hit|transfer|reuse/iu;

export function cumulativeUniqueInputTokens(
  promptTokensBySource: Record<string, TimeSeriesPoint[]> | undefined,
): TimeSeriesPoint[] {
  if (!promptTokensBySource) return [];
  const computedByT = new Map<number, number>();
  let sawComputed = false;
  for (const [source, series] of Object.entries(promptTokensBySource)) {
    if (CACHE_SOURCE_RE.test(source)) continue;
    sawComputed = true;
    for (const p of series)
      computedByT.set(p.t, (computedByT.get(p.t) ?? 0) + p.value);
  }
  if (!sawComputed) return [];
  const out: TimeSeriesPoint[] = [];
  let sum = 0;
  for (const t of [...computedByT.keys()].toSorted((x, y) => x - y)) {
    sum += computedByT.get(t)!;
    out.push({ t, value: sum });
  }
  return out;
}

export function inflightUniqueTokens(
  requests: readonly {
    cid: string;
    start: number;
    end: number;
    isl: number | null;
  }[],
): TimeSeriesPoint[] {
  if (requests.length === 0) return [];
  interface Event {
    tNs: number;
    kind: "start" | "end";
    cid: string;
    isl: number;
  }
  const events: Event[] = [];
  for (const r of requests) {
    const isl = r.isl ?? 0;
    if (isl <= 0 || r.end <= r.start) continue;
    events.push(
      { tNs: r.start, kind: "start", cid: r.cid, isl },
      { tNs: r.end, kind: "end", cid: r.cid, isl },
    );
  }
  if (events.length === 0) return [];
  events.sort((a, b) => a.tNs - b.tNs || (a.kind === "end" ? -1 : 1));

  const activeByCid = new Map<string, Map<number, number>>();
  let total = 0;
  const out: TimeSeriesPoint[] = [{ t: 0, value: 0 }];
  for (const e of events) {
    const tSec = e.tNs / 1e9;
    // A source conversation can have several concurrent replay requests.
    // Expiring one must retain the maximum input length of those still active.
    const lengths = activeByCid.get(e.cid) ?? new Map<number, number>();
    const maxLength = () =>
      [...lengths.keys()].reduce((max, length) => Math.max(max, length), 0);
    const previous = maxLength();
    const count = lengths.get(e.isl) ?? 0;
    if (e.kind === "start") lengths.set(e.isl, count + 1);
    else if (count > 1) lengths.set(e.isl, count - 1);
    else lengths.delete(e.isl);
    total += maxLength() - previous;
    if (lengths.size) activeByCid.set(e.cid, lengths);
    else activeByCid.delete(e.cid);
    out.push({ t: tSec, value: Math.max(0, total) });
  }
  return out;
}

export function cumulativeDifferenceMonotonic(
  a: TimeSeriesPoint[],
  b: TimeSeriesPoint[],
): TimeSeriesPoint[] {
  const aByT = new Map(a.map((p) => [p.t, p.value]));
  const bByT = new Map(b.map((p) => [p.t, p.value]));
  const allT = [...new Set([...aByT.keys(), ...bByT.keys()])].toSorted(
    (x, y) => x - y,
  );
  const out: TimeSeriesPoint[] = Array.from({ length: allT.length });
  let cumA = 0;
  let cumB = 0;
  let runningMax = 0;
  for (let i = 0; i < allT.length; i++) {
    const t = allT[i]!;
    cumA += aByT.get(t) ?? 0;
    cumB += bByT.get(t) ?? 0;
    const diff = cumA - cumB;
    if (diff > runningMax) runningMax = diff;
    out[i] = { t, value: runningMax };
  }
  return out;
}

function sumSeries(
  a: TimeSeriesPoint[],
  b: TimeSeriesPoint[],
): TimeSeriesPoint[] {
  const n = Math.min(a.length, b.length);
  const out: TimeSeriesPoint[] = Array.from({ length: n });
  for (let i = 0; i < n; i++) {
    out[i] = { t: a[i]!.t, value: a[i]!.value + b[i]!.value };
  }
  return out;
}

export function buildThroughputChartSeries(
  input: TimeSeriesPoint[],
  decode: TimeSeriesPoint[],
  selected: ReadonlySet<ThroughputSeriesKey>,
): ChartSeries[] {
  const series: ChartSeries[] = [];
  if (selected.has("input")) {
    series.push({
      name: "Input (avg n=50)",
      data: rollingAverage(input, 50),
      color: "#3b82f6",
      strokeWidth: 1.6,
    });
  }
  if (selected.has("decode")) {
    series.push({
      name: "Decode (avg n=50)",
      data: rollingAverage(decode, 50),
      color: "#f97316",
      strokeWidth: 1.6,
    });
  }
  if (selected.size === 2) {
    series.push({
      name: "Total running avg (60s burn-in)",
      data: cumulativeAverage(sumSeries(input, decode), 60),
      color: "#ef4444",
      strokeWidth: 3,
    });
  }
  return series;
}
