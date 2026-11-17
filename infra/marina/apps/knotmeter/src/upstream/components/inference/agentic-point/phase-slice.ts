// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce plus relative-nanosecond phase-boundary correction; see THIRD_PARTY.md.
import type {
  RequestChartData,
  RequestChartRecord,
} from "@/hooks/api/use-request-chart-data";
import type {
  RequestRecord,
  RequestTimeline,
} from "@/hooks/api/use-request-timeline";
import type {
  QueueDepthPoint,
  TimeSeriesPoint,
  TraceServerMetrics,
} from "@/hooks/api/use-trace-server-metrics";

export type StagePhase = "warmup" | "profiling";

export interface ServerSeriesLike {
  kvCacheUsage: TimeSeriesPoint[];
  prefixCacheHitRate: TimeSeriesPoint[];
  queueDepth: QueueDepthPoint[];
  promptTokensBySource: Record<string, TimeSeriesPoint[]>;
  prefillTps: TimeSeriesPoint[];
  decodeTps: TimeSeriesPoint[];
  prefixCacheHitsTps: TimeSeriesPoint[];
  hostKvCacheUsage: TimeSeriesPoint[];
  kvCacheUsageByEngine: { engineLabel: string; points: TimeSeriesPoint[] }[];
}

type RequestPhaseData = Pick<
  RequestChartData,
  "startNs" | "durationS" | "requests"
>;

export function timelineHasWarmup(
  timeline: RequestPhaseData | null | undefined,
): boolean {
  return Boolean(timeline?.requests.some((r) => r.phase !== "profiling"));
}

function phaseBoundaryOffsetNs(
  timeline: RequestPhaseData | null | undefined,
): number | null {
  if (!timeline) return null;
  let hasWarmup = false;
  let minProfilingStart: number | null = null;
  for (const r of timeline.requests) {
    if (r.phase === "profiling") {
      if (minProfilingStart === null || r.start < minProfilingStart)
        minProfilingStart = r.start;
    } else {
      hasWarmup = true;
    }
  }
  if (!hasWarmup || minProfilingStart === null) return null;
  return minProfilingStart;
}

export function phaseBoundaryNs(
  timeline: RequestPhaseData | null | undefined,
): number | null {
  const offset = phaseBoundaryOffsetNs(timeline);
  return offset === null || !timeline ? null : timeline.startNs + offset;
}

export function phaseBoundarySec(
  serverMetrics: Pick<TraceServerMetrics, "startNs"> | null | undefined,
  timeline: RequestPhaseData | null | undefined,
): number | null {
  if (!serverMetrics) return null;
  const offset = phaseBoundaryOffsetNs(timeline);
  if (offset === null || !timeline) return null;
  return Math.max(0, (timeline.startNs - serverMetrics.startNs + offset) / 1e9);
}

export interface PhaseSlicedSeries<S> {
  series: S;
  durationS: number;
}

export function sliceServerSeriesByPhase<S extends ServerSeriesLike>(
  series: S,
  phase: StagePhase,
  boundarySec: number | null,
  fullDurationS: number,
): PhaseSlicedSeries<S> {
  if (boundarySec === null) return { series, durationS: fullDurationS };
  const b = boundarySec;
  const keep =
    phase === "warmup" ? (t: number) => t < b : (t: number) => t >= b;
  const rebase =
    phase === "profiling" ? (t: number) => t - b : (t: number) => t;

  const sliceTs = (pts: TimeSeriesPoint[]): TimeSeriesPoint[] =>
    pts.filter((p) => keep(p.t)).map((p) => ({ ...p, t: rebase(p.t) }));
  const sliceQd = (pts: QueueDepthPoint[]): QueueDepthPoint[] =>
    pts.filter((p) => keep(p.t)).map((p) => ({ ...p, t: rebase(p.t) }));
  const sliceRecord = (
    rec: Record<string, TimeSeriesPoint[]>,
  ): Record<string, TimeSeriesPoint[]> => {
    const out: Record<string, TimeSeriesPoint[]> = {};
    for (const [k, v] of Object.entries(rec)) out[k] = sliceTs(v);
    return out;
  };

  const slicedFields: ServerSeriesLike = {
    kvCacheUsage: sliceTs(series.kvCacheUsage),
    prefixCacheHitRate: sliceTs(series.prefixCacheHitRate),
    queueDepth: sliceQd(series.queueDepth),
    promptTokensBySource: sliceRecord(series.promptTokensBySource),
    prefillTps: sliceTs(series.prefillTps),
    decodeTps: sliceTs(series.decodeTps),
    prefixCacheHitsTps: sliceTs(series.prefixCacheHitsTps),
    hostKvCacheUsage: sliceTs(series.hostKvCacheUsage),
    kvCacheUsageByEngine: series.kvCacheUsageByEngine.map((e) => ({
      engineLabel: e.engineLabel,
      points: sliceTs(e.points),
    })),
  };

  const durationS = phase === "warmup" ? b : Math.max(1, fullDurationS - b);
  return { series: { ...series, ...slicedFields } as S, durationS };
}

export function requestsForPhase(
  requests: RequestRecord[],
  phase: StagePhase,
): RequestRecord[] {
  return phase === "warmup"
    ? requests.filter((r) => r.phase !== "profiling")
    : requests.filter((r) => r.phase === "profiling");
}

export function sliceTimelineByPhase(
  timeline: RequestTimeline,
  phase: StagePhase,
): RequestTimeline {
  const boundaryOff = phaseBoundaryOffsetNs(timeline);
  if (boundaryOff === null) return timeline;
  const inPhase = (r: RequestRecord) =>
    phase === "warmup" ? r.start < boundaryOff : r.start >= boundaryOff;
  const shift = phase === "profiling" ? boundaryOff : 0;
  const requests = timeline.requests.filter(inPhase).map((r) => ({
    ...r,
    credit: r.credit - shift,
    start: r.start - shift,
    ack: r.ack === null ? null : r.ack - shift,
    end: r.end - shift,
  }));
  const durationS =
    phase === "warmup"
      ? boundaryOff / 1e9
      : Math.max(1, timeline.durationS - boundaryOff / 1e9);
  return {
    ...timeline,
    startNs: timeline.startNs + shift,
    requests,
    durationS,
  };
}

export function sliceRequestChartDataByPhase(
  data: RequestChartData,
  phase: StagePhase,
): RequestChartData {
  const boundaryOff = phaseBoundaryOffsetNs(data);
  if (boundaryOff === null) return data;
  const inPhase = (request: RequestChartRecord) =>
    phase === "warmup"
      ? request.start < boundaryOff
      : request.start >= boundaryOff;
  const shift = phase === "profiling" ? boundaryOff : 0;
  const requests = data.requests.filter(inPhase).map((request) => ({
    ...request,
    start: request.start - shift,
    end: request.end - shift,
  }));
  const durationS =
    phase === "warmup"
      ? boundaryOff / 1e9
      : Math.max(1, data.durationS - boundaryOff / 1e9);
  return { ...data, startNs: data.startNs + shift, requests, durationS };
}
