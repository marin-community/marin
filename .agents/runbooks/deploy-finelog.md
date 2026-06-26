---
name: deploy-finelog
description: Ship a new finelog server image with health-gated auto-rollback, roll back to a prior digest, or query evicted-segment parquet from GCS.
---

# Runbook: Roll out / roll back a finelog server, and query archived parquet

**When you're here:** You need to push a new finelog (log/stats server) image to
a cluster's deployment, undo a bad push, or read log/stats data that the live
server has already evicted to GCS. Each iris cluster names its finelog via
`log_server_config` in its cluster config; that name resolves to
`lib/finelog/config/<name>.yaml` (e.g. `marin` → `lib/finelog/config/marin.yaml`).

**TL;DR:**
- **Risky push → use safe_deploy.** `uv run python lib/finelog/scripts/safe_deploy.py rollout <name>`
  captures the running digest, refuses to deploy a mutable tag, polls `/health`,
  and auto-rolls-back on failure. Plain `finelog deploy restart` / `iris cluster
  log-server restart` does **none** of that.
- **Undo a bad push →** `safe_deploy.py rollback <name>` restores the last
  captured digest (or `--to <image>` for an explicit one).
- **Data missing from `FetchLogs` →** it was evicted to parquet. Query it with
  `finelog gcs-query <name> "<sql>"`, and **bound egress** with
  `--created-since-ms` / `--created-until-ms`.

## Before you touch anything

- **`finelog deploy restart` and `iris cluster log-server restart` have NO
  rollback.** They rebuild and replace the container in place; if the new image
  crash-loops, the deployment is just down. For any push you're not certain of,
  use `safe_deploy.py rollout` instead — it captures the previous digest and
  auto-reverts. (Both restart paths default to `--build`; pass `--no-build` to
  reuse the registry's existing tag.)
- **`gcs-query` reads parquet straight from `remote_log_dir`** (a GCS bucket,
  e.g. `gs://marin-us-central2/finelog/marin` in `lib/finelog/config/marin.yaml`).
  Scanning every segment fetches every parquet footer — that is real egress and
  real cost. **Always bound it** with `--created-since-ms` / `--created-until-ms`
  so DuckDB skips clearly-out-of-range segments before opening them, and keep a
  `LIMIT` in the SQL. The default `--max-rows=100000` is a guardrail, not a budget.
- **safe_deploy is GCP-only** (`_require_gcp` raises otherwise) and operates on
  the GCE VM behind the config — confirm you're pointed at the right `<name>`.
- **`down` is destructive**; on k8s `-y` also deletes the PVC (the archived
  parquet). Not part of a rollout — don't reach for it to "reset."

## Steps

### Roll out a new image (health-gated, auto-rollback)

```bash
uv run python lib/finelog/scripts/safe_deploy.py rollout <name>
```

What it does, in order (`safe_deploy.py:141-216`): builds & pushes `cfg.image`
(skip with `--no-build`; `--fast` uses the Rust `fast` cargo profile for
dev/test clusters), captures the **currently-running** container's pinned
`@sha256:` digest, resolves `cfg.image` to a content digest and **refuses to
deploy if it can't pin one** (no mutable tags), records both to
`~/.cache/finelog/deploy-state/<name>.json`, re-bootstraps with the new digest,
then polls `/health`. If health fails it re-bootstraps the captured old digest
and exits non-zero. If there was no running container to capture, auto-rollback
is disabled and a health failure leaves you to run `rollback` manually.

For a routine, low-risk refresh you can use the plain path instead — but it has
no safety net:

```bash
iris --config=<cluster-config> cluster log-server restart   # wraps finelog deploy restart
# or, addressing the finelog config directly:
cd lib/finelog && uv run finelog deploy restart <name>
```

### Roll back

```bash
uv run python lib/finelog/scripts/safe_deploy.py rollback <name>
# or to an explicit image/digest:
uv run python lib/finelog/scripts/safe_deploy.py rollback <name> --to ghcr.io/...@sha256:...
```

With no `--to`, it restores `previous_digest` from the deploy-state file
(`safe_deploy.py:219-258`); it re-bootstraps and verifies `/health`, so a
rollback that can't come up healthy fails loudly rather than silently.

Check what's deployed and what the last rollout captured:

```bash
uv run python lib/finelog/scripts/safe_deploy.py status <name>
```

### Query evicted-segment parquet

When `FetchLogs` (the dashboard or `finelog query`) returns empty for an old
time range, the live deque has already evicted those segments to `REMOTE`; the
parquet still lives under `remote_log_dir`. Query it directly:

```bash
cd lib/finelog
uv run finelog gcs-query <name> \
  'select ts, key, value from log where key like '"'"'/user/job/%'"'"' order by ts desc limit 500' \
  --created-since-ms <epoch_ms_lower> --created-until-ms <epoch_ms_upper>
```

Each namespace directory under `remote_log_dir` is registered as a DuckDB view
named after the namespace; reference it in the `FROM` clause, double-quoting
dotted names (`from "iris.worker"`). Restrict the view set with
`--namespace <ns>` (repeatable). Pad `--created-until-ms` above your `ts` upper
bound by a few hours to cover L0→L1 compaction lag (`cli.py:443-451`). This is
distinct from `finelog query`, which hits the **live** server over a tunnel (see
lib/iris/OPS.md:183) — `gcs-query` never touches the server.

## Verify

- **Rollout/rollback:** `safe_deploy.py status <name>` should show `running
  digest:` equal to the digest you intended (`current_digest` in state). The
  rollout already gated on `/health`; if it printed `OK — <name> healthy on
  <digest>` the server answered `/health`. Cross-check via `iris cluster
  log-server status` / `finelog deploy status <name>`.
- **Live data flowing again:** run a `finelog query <name> "..."` against a
  recent namespace (lib/iris/OPS.md:183) and confirm fresh rows, not just that
  the container is up.
- **gcs-query:** confirm the row count is non-zero and within your time bounds;
  if you hit the `--max-rows` cap, tighten the `WHERE`/`LIMIT` rather than just
  raising the cap (that's more egress).

## Why this happens / Notes

finelog is decoupled from iris: it has no `iris.*` imports and treats log keys as
opaque strings (`lib/finelog/AGENTS.md` "Boundaries"). So iris can't ship finelog
inline — the deployment is a separate GCE container, and `iris cluster
log-server <verb>` is a thin shim that invokes the matching `finelog deploy`
command for the cluster's `log_server_config` (`lib/iris/cli/cluster.py:447-518`).

The split between rollout and the namespace schema matters: **iris owns the
stats namespaces** (`iris.worker`, `iris.task`, `iris.profile`) and their
retention; finelog only stores opaque rows. When you query archived data you are
reading iris-defined schemas out of a finelog-owned bucket — get the column/key
shapes from lib/iris/OPS.md:161 "Stats Namespaces", not from finelog.

`safe_deploy` exists because the in-place `restart` path has no health gate: a
bad image just takes finelog down with no captured prior digest to revert to.
The state file (`safe_deploy.py:34-58`) is what makes `rollback` possible after
the fact — without a prior successful `rollout`, there's no `previous_digest` and
you must pass `--to` explicitly. finelog has no `OPS.md` today; this runbook plus
the deploy CLI docstrings are the operational reference.

## See also

- `lib/finelog/scripts/safe_deploy.py` — the rollout/rollback/status authority (owns the flags).
- lib/iris/OPS.md "Stats Namespaces" — the `iris.*` schemas you read when querying archived parquet (iris owns them).
