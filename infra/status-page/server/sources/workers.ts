// Iris worker counts via the ListWorkers Connect RPC.
//
// Worker liveness used to live in SQLite columns (`healthy`, `active`,
// `last_heartbeat_ms`), which let us aggregate everything in two raw-SQL
// queries. After PR #5559 ("in-memory worker liveness") those columns
// were dropped — health now lives only in WorkerHealthTracker — so this
// source pages through ListWorkers and aggregates client-side.
//
// One snapshot returns:
//   - healthy worker count + total CPU millicores / memory bytes / TPU
//     chips across all healthy workers;
//   - per-region healthy breakdown (region read from
//     metadata.attributes.region).
//
// Note: previous code reported "currently unallocated" CPU/memory by
// subtracting committed columns. The proto returned by ListWorkers
// doesn't carry committed_* (those still live in the workers SQL table
// but can't be joined to in-memory health from a single RPC), so we
// report total resources of healthy workers instead. Iris schedules at
// whole-VM granularity anyway, which made the "available" number a
// thin proxy for "idle VMs × resources" on a busy cluster.
//
// History lives in a separate ring buffer (server/history.ts); this
// file only ever returns the current snapshot.

import { getControllerUrl } from "./discovery.js";

const PAGE_LIMIT = 1000; // Matches MAX_LIST_WORKERS_LIMIT on the controller.
const SNAPSHOT_DEADLINE_MS = 20_000;
const RPC_TIMEOUT_MS = 10_000;

// Subset of iris.cluster.Controller.WorkerHealthStatus we read. Connect
// returns proto3 JSON with camelCase field names; only the fields we
// actually consume are typed here.
interface WorkerHealthStatusJson {
  workerId?: string;
  healthy?: boolean;
  metadata?: WorkerMetadataJson;
}

interface WorkerMetadataJson {
  cpuCount?: number;
  memoryBytes?: string | number; // int64 → string in proto3 JSON
  device?: DeviceConfigJson;
  attributes?: Record<string, AttributeValueJson>;
}

interface DeviceConfigJson {
  tpu?: { count?: number };
  gpu?: { count?: number };
}

interface AttributeValueJson {
  stringValue?: string;
  intValue?: string | number;
  floatValue?: number;
}

interface ListWorkersResponseJson {
  workers?: WorkerHealthStatusJson[];
  totalCount?: number;
  hasMore?: boolean;
}

export interface WorkerResourceTotals {
  cpuTotalMillicores: number;
  memoryTotalBytes: number;
  // "chips" = total TPU chips across all healthy workers.
  chipsTotal: number;
}

export interface WorkerRegionCount {
  region: string;
  healthy: number;
}

export interface WorkersSnapshot {
  healthy: number;
  resources: WorkerResourceTotals;
  byRegion: WorkerRegionCount[];
  fetchedAt: string;
  error?: string;
}

export interface WorkerSample {
  t: number; // epoch millis
  // Per-region healthy worker count at the sample time. Flat map keyed
  // by region name so the frontend can feed recharts directly (each
  // region becomes a <Line dataKey={region} />).
  regions: Record<string, number>;
}

function emptyResources(): WorkerResourceTotals {
  return {
    cpuTotalMillicores: 0,
    memoryTotalBytes: 0,
    chipsTotal: 0,
  };
}

async function listWorkersPage(
  base: string,
  offset: number,
): Promise<ListWorkersResponseJson> {
  const ac = new AbortController();
  const timer = setTimeout(
    () => ac.abort(new Error(`ListWorkers timed out after ${RPC_TIMEOUT_MS}ms`)),
    RPC_TIMEOUT_MS,
  );
  try {
    const res = await fetch(
      `${base}/iris.cluster.ControllerService/ListWorkers`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: { offset, limit: PAGE_LIMIT } }),
        signal: ac.signal,
      },
    );
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`ListWorkers ${res.status}: ${body.slice(0, 300)}`);
    }
    return (await res.json()) as ListWorkersResponseJson;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchAllWorkers(base: string): Promise<WorkerHealthStatusJson[]> {
  const all: WorkerHealthStatusJson[] = [];
  let offset = 0;
  // Bound the loop so a misbehaving controller can't keep us iterating
  // forever; total_count > offset+limit drives the next page, and the
  // SNAPSHOT_DEADLINE_MS race in workerSnapshot is the outer safety net.
  for (let pages = 0; pages < 50; pages += 1) {
    const page = await listWorkersPage(base, offset);
    const workers = page.workers ?? [];
    all.push(...workers);
    if (!page.hasMore || workers.length === 0) {
      break;
    }
    offset += workers.length;
  }
  return all;
}

function regionOf(worker: WorkerHealthStatusJson): string {
  const value = worker.metadata?.attributes?.region;
  if (value && typeof value.stringValue === "string" && value.stringValue) {
    return value.stringValue;
  }
  return "unknown";
}

function chipsOf(worker: WorkerHealthStatusJson): number {
  return Number(worker.metadata?.device?.tpu?.count ?? 0);
}

function aggregate(workers: WorkerHealthStatusJson[]): {
  healthy: number;
  resources: WorkerResourceTotals;
  byRegion: WorkerRegionCount[];
} {
  let healthy = 0;
  let cpuTotalMillicores = 0;
  let memoryTotalBytes = 0;
  let chipsTotal = 0;
  const regionCounts = new Map<string, number>();

  for (const w of workers) {
    if (!w.healthy) continue;
    healthy += 1;
    cpuTotalMillicores += Number(w.metadata?.cpuCount ?? 0) * 1000;
    memoryTotalBytes += Number(w.metadata?.memoryBytes ?? 0);
    chipsTotal += chipsOf(w);
    const region = regionOf(w);
    regionCounts.set(region, (regionCounts.get(region) ?? 0) + 1);
  }

  const byRegion = Array.from(regionCounts.entries())
    .map(([region, count]) => ({ region, healthy: count }))
    .sort((a, b) => b.healthy - a.healthy);

  return {
    healthy,
    resources: { cpuTotalMillicores, memoryTotalBytes, chipsTotal },
    byRegion,
  };
}

export async function workerSnapshot(): Promise<WorkersSnapshot> {
  const fetchedAt = new Date().toISOString();
  try {
    const base = await getControllerUrl();
    const workers = await Promise.race([
      fetchAllWorkers(base),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error("workerSnapshot deadline exceeded")),
          SNAPSHOT_DEADLINE_MS,
        ),
      ),
    ]);
    const { healthy, resources, byRegion } = aggregate(workers);
    return { healthy, resources, byRegion, fetchedAt };
  } catch (err) {
    return {
      healthy: 0,
      resources: emptyResources(),
      byRegion: [],
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
