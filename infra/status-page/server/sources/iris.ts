// Iris controller reachability via the /health endpoint.
//
// All Connect RPC methods on the controller support JSON natively
// (POST + Content-Type: application/json), which is how the workers
// and jobs sources call ExecuteRawQuery. This source sticks with the
// simpler /health GET because it only needs a binary reachable/latency
// signal, not structured data from the controller's SQLite.
//
// `pingIris()` does one fetch and returns its outcome. `irisStatus()`
// assembles the API response by combining the most recent ping with
// percentiles computed over a rolling window of successful samples
// kept by the caller (see server/main.ts).

import { getControllerUrl } from "./discovery.js";

const CLUSTER = process.env.CLUSTER_NAME ?? "marin";

export interface IrisPingSample {
  t: number; // epoch millis
  latencyMs: number;
}

export interface PingPercentiles {
  p50: number;
  p90: number;
  p99: number;
}

export interface IrisPingResult {
  fetchedAt: string;
  reachable: boolean;
  latencyMs: number | null;
  controllerUrl: string | null;
  error?: string;
  raw?: unknown;
}

export interface IrisStatus {
  cluster: string;
  reachable: boolean;
  latencyMs: number | null;
  pingPercentiles: PingPercentiles | null;
  pingSampleCount: number;
  pingSpanMs: number;
  pingWindowMs: number;
  controllerUrl: string | null;
  fetchedAt: string;
  error?: string;
  raw?: unknown;
}

export async function pingIris(): Promise<IrisPingResult> {
  const fetchedAt = new Date().toISOString();

  let base: string;
  try {
    base = await getControllerUrl();
  } catch (err) {
    return {
      fetchedAt,
      reachable: false,
      latencyMs: null,
      controllerUrl: null,
      error: `discovery failed: ${(err as Error).message}`,
    };
  }

  const start = performance.now();
  try {
    const res = await fetch(`${base}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    const latencyMs = Math.round(performance.now() - start);
    if (!res.ok) {
      return {
        fetchedAt,
        reachable: false,
        latencyMs,
        controllerUrl: base,
        error: `controller /health returned ${res.status}`,
      };
    }
    const raw = await res.json().catch(() => ({}));
    return {
      fetchedAt,
      reachable: true,
      latencyMs,
      controllerUrl: base,
      raw,
    };
  } catch (err) {
    return {
      fetchedAt,
      reachable: false,
      latencyMs: null,
      controllerUrl: base,
      error: `controller fetch failed: ${(err as Error).message}`,
    };
  }
}

// Linear-interpolated percentile over a sorted ascending array. Returns
// null when there are no samples so the API can distinguish "no data
// yet" from a real zero-millisecond reading.
function percentile(sorted: number[], p: number): number {
  const idx = ((p / 100) * (sorted.length - 1));
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export function computePercentiles(samples: IrisPingSample[]): PingPercentiles | null {
  if (samples.length === 0) return null;
  const sorted = samples.map((s) => s.latencyMs).sort((a, b) => a - b);
  return {
    p50: Math.round(percentile(sorted, 50)),
    p90: Math.round(percentile(sorted, 90)),
    p99: Math.round(percentile(sorted, 99)),
  };
}

function sampleSpanMs(samples: IrisPingSample[]): number {
  if (samples.length < 2) return 0;
  const first = samples[0];
  const last = samples[samples.length - 1];
  return Math.max(0, last.t - first.t);
}

export function irisStatus(
  last: IrisPingResult | null,
  samples: IrisPingSample[],
  windowMs: number,
): IrisStatus {
  const fetchedAt = last?.fetchedAt ?? new Date().toISOString();
  return {
    cluster: CLUSTER,
    reachable: last?.reachable ?? false,
    latencyMs: last?.latencyMs ?? null,
    pingPercentiles: computePercentiles(samples),
    pingSampleCount: samples.length,
    pingSpanMs: sampleSpanMs(samples),
    pingWindowMs: windowMs,
    controllerUrl: last?.controllerUrl ?? null,
    fetchedAt,
    error: last?.error,
    raw: last?.raw,
  };
}
