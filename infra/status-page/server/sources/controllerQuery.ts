// Helper for calling the iris controller's raw-SQL Connect RPC.
//
// The controller exposes a SQLite read endpoint at
// /iris.cluster.ControllerService/ExecuteRawQuery. The RPC normally
// requires the "admin" role, but the marin cluster runs in null-auth mode
// (see lib/iris/src/iris/cluster/controller/auth.py) which routes every
// caller through NullAuthInterceptor → admin, so no bearer token is
// required. If auth is ever enabled we'll need to add one.
//
// Wire shape (per lib/iris/src/iris/rpc/query.proto):
//   request  { sql: string }
//   response { columns: [{ name, type }], rows: string[] }
// Each row is a JSON-encoded array of cell values, so we parse once at
// the boundary and hand callers plain JS arrays.

import { getControllerUrl } from "./discovery.js";

export interface ColumnMeta {
  name: string;
  type: string;
}

export interface RawQueryResult {
  columns: ColumnMeta[];
  rows: unknown[][];
}

interface RawQueryWireResponse {
  columns: ColumnMeta[];
  rows: string[];
}

export async function executeRawQuery(sql: string): Promise<RawQueryResult> {
  const base = await getControllerUrl();
  const res = await fetch(`${base}/iris.cluster.ControllerService/ExecuteRawQuery`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ sql }),
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`ExecuteRawQuery ${res.status}: ${body.slice(0, 300)}`);
  }
  const body = (await res.json()) as RawQueryWireResponse;
  const rows = body.rows.map((r) => JSON.parse(r) as unknown[]);
  return { columns: body.columns, rows };
}
