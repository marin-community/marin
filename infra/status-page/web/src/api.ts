// Typed fetch wrappers for the server's JSON endpoints.
//
// The types here intentionally mirror the server-side shapes defined in
// server/sources/*.ts. Keeping them duplicated (rather than importing
// across the server/web boundary) avoids tangling tsconfigs and lets the
// web bundle stay independent of node types.

export interface FerryRun {
  id: number;
  conclusion: string | null;
  status: string;
  sha: string;
  shaShort: string;
  startedAt: string;
  durationSeconds: number | null;
  url: string;
  event: string;
  actor: string;
}

export interface FerryWorkflowStatus {
  name: string;
  file: string;
  latest: FerryRun | null;
  history: FerryRun[];
  successRate: number | null;
  fetchedAt: string;
  error?: string;
}

export interface FerryResponse {
  workflows: FerryWorkflowStatus[];
}

export interface OrchStatus {
  cluster: string;
  reachable: boolean;
  latencyMs: number | null;
  controllerUrl: string | null;
  fetchedAt: string;
  error?: string;
  raw?: unknown;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} returned ${res.status}`);
  }
  return (await res.json()) as T;
}

export const fetchFerry = () => getJson<FerryResponse>("/api/ferry");
export const fetchOrch = () => getJson<OrchStatus>("/api/orch");
