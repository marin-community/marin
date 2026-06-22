---
name: offline-checkpoint-analysis
description: Run expensive controller queries safely — verify cheap against the live DB, then scan an existing GCS checkpoint or the parquet logs offline.
---

# Runbook: Run expensive controller queries safely (offline checkpoint + parquet log queries)

**When you're here:** You need a query the live controller can't safely serve —
a full-table scan, a cross-table invariant check, or a grep through the
controller's log history. The trap is reaching for `iris cluster controller
checkpoint` or a heavy `iris query` and stalling a controller whose scheduling
loop may already be dead.

**TL;DR:**
- **Verify the hypothesis cheaply against the *live* DB first** with a narrow
  `iris query` (state filter + LIMIT). Most hypotheses die here in one shot.
- **For a slow scan, copy the most recent EXISTING checkpoint from GCS** (`gcloud
  storage cp` + `zstd -df`) and run `sqlite3` against it locally. Do **not**
  trigger a new `iris cluster controller checkpoint` to get fresher state — that
  stalls the controller, and on a wedged one it can hang.
- **For controller *log* history** (tracebacks, audit events at volume), query
  the `logs_*.parquet` shards in GCS directly with duckdb + gcsfs over ADC. No
  local download.
- The offline snapshot may **pre-date** the incident. Cross-check any conclusion
  against the live DB before you act on it.

## Before you touch anything

- **Never run expensive queries against the live controller DB.** A full scan or
  unindexed cross-table join blocks the controller's single SQLite connection and
  stalls scheduling (lib/iris/OPS.md:151). Cheap, bounded `iris query` calls
  (state filter, `LIMIT`) are fine and are the *first* step, not a banned one.
- **Never trigger a new checkpoint on a controller you suspect is wedged.**
  `iris cluster controller checkpoint` briefly stalls the controller; if the
  scheduling loop has already crashed, the checkpoint can hang and buys you
  nothing — checkpoints already land in GCS roughly hourly. Copy the existing
  one instead.
- **Read-only, always.** Even on an offline copy: **never modify the controller
  DB** without explicit user approval (lib/iris/OPS.md:113). The offline file is
  for inspection; fixes go through migrations, not `UPDATE`.
- **Snapshot staleness is the standing gotcha.** A copied checkpoint or a stale
  `/tmp/iris-debug/controller.sqlite3` may be older than the event you're
  chasing. Treat offline counts (e.g. "0 split jobs", "row not present") as
  *suggestive*, not authoritative, and confirm against the live DB.
- **Cross-region cost.** Copy from the cluster's own region bucket
  (`gs://marin-us-central2/...` for marin). Don't pull checkpoints or log shards
  across regions.

## Steps

### 1. Verify cheap against the live DB first

State your hypothesis as a bounded query and run it against the live controller.
The connection selector (`--cluster`/`--config`) is a global flag before the
subcommand (lib/iris/OPS.md:11). Query catalog, state codes, and the active-state
sharp edges live in lib/iris/OPS.md:104 ("SQL Queries") — don't restate them.

```bash
iris query "SELECT job_id, error FROM jobs WHERE state=5 ORDER BY submitted_at_ms DESC LIMIT 10"
```

Keep it bounded: a state filter and a `LIMIT` keep the scan off the hot path.
If this answers the question — and it usually does — you're done. Only escalate
to an offline scan when the query you actually need is genuinely expensive
(full-table, unindexed join, `max()` over `task_attempts` per task).

### 2. Copy the most recent existing GCS checkpoint (for slow DB scans)

List the checkpoints already in GCS and copy the newest. Timestamps are
epoch-ms; the bucket/cluster path for marin is
`gs://marin-us-central2/iris/marin/state/controller-state/`.

```bash
# Newest checkpoints (epoch-ms timestamps)
gcloud storage ls gs://marin-us-central2/iris/marin/state/controller-state/

# Copy the most recent and decompress
gcloud storage cp gs://marin-us-central2/iris/marin/state/controller-state/<epoch-ms>/controller.sqlite3.zst /tmp/
zstd -df /tmp/controller.sqlite3.zst

sqlite3 /tmp/controller.sqlite3 "SELECT task_id, max(attempt_id) FROM task_attempts GROUP BY task_id HAVING max(attempt_id) > 0"
```

`gcloud storage cp`, **not** `gsutil` (deprecated at this org). Note the
checkpoint's timestamp against your incident window — if it pre-dates the event,
the row you want may simply not be in it yet. A stale snapshot has quietly
refuted real-looking hypotheses before; that is a staleness artifact, not a
finding.

### 3. Query the parquet log shards directly (for log history)

For controller log history — tracebacks, `ManagedThread.*crashed`, audit events
at volume — the source is the `logs_*.parquet` shards in GCS, schema
`seq, key, source, data, epoch_ms, level`. Filter `key='/system/controller'`
for controller-only lines. Query them in place with pyarrow.dataset + gcsfs +
duckdb over ADC — no local download:

```python
import duckdb, gcsfs, pyarrow.dataset as ds

fs = gcsfs.GCSFileSystem(project='hai-gcp-models')
files = sorted(fs.glob('gs://marin-us-central2/iris/marin/state/logs/logs_*.parquet'))[-20:]
con = duckdb.connect()
con.register('logs', ds.dataset(files, format='parquet', filesystem=fs))

con.execute("""
    SELECT epoch_ms, substr(data, 1, 200)
    FROM logs
    WHERE key='/system/controller' AND data LIKE '%Traceback%'
    ORDER BY epoch_ms DESC LIMIT 50
""").fetchall()
```

Auth is ADC (`gcloud auth application-default login`). **Do not** reach for
duckdb's native `gs://` httpfs extension — it wants HMAC keys and rejects ADC;
the `CREATE SECRET` path is a 10-minute dead end. The `pyarrow.dataset` +
`con.register` path is the one that works. For low-volume audit events,
`iris process logs --since 24h | grep 'event=...'` is simpler — see
lib/iris/OPS.md "Audit events".

## Verify

- **Re-confirm any actionable conclusion against the live DB.** If the offline
  scan says a poisoned/orphan row exists (or doesn't), re-run the narrowest form
  of that query via `iris query` before acting. The snapshot may pre-date the
  state you care about.
- **Confirm you took no checkpoint on a wedged controller.** If you ran step 2
  by copying an existing file, the controller never paused. If scheduling was
  frozen and you reflexively ran `iris cluster controller checkpoint`, check
  whether it returned — a hang there confirms the loop is dead, not slow.
- **Confirm read-only.** The offline file and the live DB are both unchanged; any
  repair lands as a migration on restart, not an `UPDATE` here.

## Why this happens

The controller serves its SQLite DB over RPC on a single connection that the
scheduling loop also uses. An expensive scan blocks that connection, so a "just
checking" query becomes a scheduling stall. Taking a fresh checkpoint has the
same failure mode plus a worse one: when the scheduling loop has *already*
crashed (a `ManagedThread` dies silently, logs `<name> crashed`, and is never
respawned), the checkpoint can hang and you learn nothing the hourly GCS
checkpoint wouldn't have told you. Copying the existing checkpoint sidesteps both: zero controller
impact, and the data is at most ~an hour stale — which is exactly why the
verify-against-live step exists.

The parquet detour matters because the controller's log *history* isn't in the
SQLite DB at all; it's in the finelog parquet shards. Grepping them in place is
~10× faster than downloading a handful of shards and avoids the temptation to
scan too few.

## See also

- lib/iris/OPS.md:104 "SQL Queries" — the query catalog, state codes, active-state
  and committed-resource sharp edges, and the audit-event grep. This runbook owns
  the *offline procedure*; that section owns the *query reference*.
