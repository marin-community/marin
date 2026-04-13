// Iris controller reachability via the /health endpoint.
//
// All Connect RPC methods on the controller support JSON natively
// (POST + Content-Type: application/json), which is how the workers
// and jobs sources call ExecuteRawQuery. This source sticks with the
// simpler /health GET because it only needs a binary reachable/latency
// signal, not structured data from the controller's SQLite.

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
