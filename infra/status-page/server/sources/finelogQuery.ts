// Helper for running a read-only SQL query against the finelog log-server.
//
// Finelog's StatsService exposes a DuckDB-backed query endpoint at
// /finelog.stats.StatsService/Query. The server speaks the Connect protocol
// over plain HTTP, so — exactly like the iris controller's ExecuteRawQuery
// (see controllerQuery.ts) — a JSON POST works without a generated client:
//
//   request  { sql: string }
//   response { arrowIpc: <base64 bytes>, rowCount: <int64 as string> }
//
// The result table is an Arrow IPC stream (proto3 JSON encodes the `bytes`
// field as base64). We decode it with apache-arrow and hand callers plain
// JS row objects, coercing Arrow's BigInt cells to numbers so the rows
// survive JSON serialization back out of the dashboard API.

import { tableFromIPC } from "apache-arrow";
import { activeFinelogUrl } from "./serviceHealth.js";

const QUERY_PATH = "/finelog.stats.StatsService/Query";
const QUERY_TIMEOUT_MS = 10_000;

interface QueryWireResponse {
  // proto3 JSON lower-camel-cases field names; accept the proto names too in
  // case a future codec emits them verbatim.
  arrowIpc?: string;
  arrow_ipc?: string;
}

// Format a Date as the tz-naive UTC literal DuckDB compares against finelog's
// timestamp('ms') columns (which the probes stamp in UTC). Mirrors the
// `TIMESTAMP 'YYYY-MM-DD HH:MM:SS'` literal infra/probes/src/provisioning.py
// builds, sidestepping tz-aware-vs-naive comparison errors from now().
export function sqlTimestampUtc(at: Date): string {
  return at.toISOString().slice(0, 19).replace("T", " ");
}

export async function queryFinelog(sql: string): Promise<Record<string, unknown>[]> {
  const base = await activeFinelogUrl();
  // Hold a strong reference to the controller so the abort timer is not GC'd
  // before it fires (same undici quirk controllerQuery.ts guards against).
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(new Error("finelog query timed out after 10s")), QUERY_TIMEOUT_MS);
  try {
    const res = await fetch(`${base}${QUERY_PATH}`, {
      method: "POST",
      // connectrpc (the Rust server crate) follows the Connect spec, which
      // requires this header on unary JSON calls to distinguish them from a
      // stray POST; the iris controller's Python server is laxer but it's
      // standard to send it.
      headers: { "content-type": "application/json", "connect-protocol-version": "1" },
      body: JSON.stringify({ sql }),
      signal: ac.signal,
    });
    if (!res.ok) {
      // Connect reports failures as non-2xx with a JSON { code, message } body.
      const body = await res.text().catch(() => "");
      throw new Error(`finelog Query ${res.status}: ${body.slice(0, 300)}`);
    }
    const body = (await res.json()) as QueryWireResponse;
    const arrowB64 = body.arrowIpc ?? body.arrow_ipc;
    if (!arrowB64) return [];
    return decodeArrowRows(Buffer.from(arrowB64, "base64"));
  } finally {
    clearTimeout(timer);
  }
}

function decodeArrowRows(ipc: Uint8Array): Record<string, unknown>[] {
  const table = tableFromIPC(ipc);
  const fields = table.schema.fields.map((f) => f.name);
  const rows: Record<string, unknown>[] = [];
  for (const row of table) {
    const obj: Record<string, unknown> = {};
    for (const name of fields) {
      const value = (row as Record<string, unknown>)[name];
      // Arrow returns int64/timestamp cells as BigInt, which JSON.stringify
      // rejects; downgrade to number (safe for our ms timestamps and counts).
      obj[name] = typeof value === "bigint" ? Number(value) : value;
    }
    rows.push(obj);
  }
  return rows;
}
