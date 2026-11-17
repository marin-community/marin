import type { RequestTimeline } from "@/hooks/api/use-request-timeline";
import type { RequestChartData } from "@/hooks/api/use-request-chart-data";
import type {
  MetricSourceSeries,
  TraceServerMetrics,
} from "@/hooks/api/use-trace-server-metrics";
import type { RequestRecord, Run } from "./types";

export interface RequestResponse {
  origin_ns: string;
  requests: RequestRecord[];
}
export type ServerMetrics = Omit<
  TraceServerMetrics,
  "meta" | "metricSources" | "kvCachePoolTokens"
> & { metricSources: MetricSourceSeries[] };
export interface RunData extends RequestResponse {
  run: Run;
  series: ServerMetrics | null;
}

export function requestTimeline(response: RequestResponse): RequestTimeline {
  const duration = response.requests.reduce(
    (end, request) => Math.max(end, request.end_ns),
    0,
  );
  return {
    version: 6,
    startNs: Number(response.origin_ns),
    endNs: Number(response.origin_ns) + duration,
    durationS: duration / 1e9,
    requests: response.requests.map((r) => ({
      cid: r.conversation_id,
      ri: r.replay_index ?? undefined,
      ti: r.turn,
      srcTrace: r.corpus_id,
      srcOuter: r.source_outer_idx,
      srcInner: r.source_inner_idx ?? undefined,
      srcKind: r.source_kind,
      wid: r.worker_id,
      ad: r.agent_depth,
      phase: r.phase,
      credit: r.credit_ns,
      start: r.start_ns,
      ack: r.ack_ns,
      end: r.end_ns,
      ttftMs: r.ttft_ms,
      tpotMs: r.tpot_ms,
      isl: r.input_tokens,
      osl: r.output_tokens,
      cancelled: r.cancelled,
    })),
  };
}

export function requestChartData(timeline: RequestTimeline): RequestChartData {
  return { ...timeline, version: 1, timelineVersion: timeline.version };
}
