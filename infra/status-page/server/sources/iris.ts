// Iris controller reachability.
//
// In v1, the controller's only easily-callable JSON endpoint is /health
// (see lib/iris/src/iris/cluster/controller/dashboard.py:396). Everything
// else is HTML dashboard or Connect RPC. This source reports reachability
// + round-trip latency and leaves room in the response shape for richer
// data later (which is increasingly supplied by the workers and jobs
// sources reading the controller's SQLite directly via ExecuteRawQuery).

import { getControllerUrl } from "./discovery.js";

const CLUSTER = process.env.CLUSTER_NAME ?? "marin";

export interface IrisStatus {
  cluster: string;
  reachable: boolean;
  latencyMs: number | null;
  controllerUrl: string | null;
  fetchedAt: string;
  error?: string;
  raw?: unknown;
}

export async function irisStatus(): Promise<IrisStatus> {
  const fetchedAt = new Date().toISOString();

  let base: string;
  try {
    base = await getControllerUrl();
  } catch (err) {
    return {
      cluster: CLUSTER,
      reachable: false,
      latencyMs: null,
      controllerUrl: null,
      fetchedAt,
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
        cluster: CLUSTER,
        reachable: false,
        latencyMs,
        controllerUrl: base,
        fetchedAt,
        error: `controller /health returned ${res.status}`,
      };
    }
    const raw = await res.json().catch(() => ({}));
    return {
      cluster: CLUSTER,
      reachable: true,
      latencyMs,
      controllerUrl: base,
      fetchedAt,
      raw,
    };
  } catch (err) {
    return {
      cluster: CLUSTER,
      reachable: false,
      latencyMs: null,
      controllerUrl: base,
      fetchedAt,
      error: `controller fetch failed: ${(err as Error).message}`,
    };
  }
}
