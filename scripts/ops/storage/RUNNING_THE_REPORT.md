# Running the Storage Report

How to run the end-to-end Marin storage report — the "one-command" orchestrator
in `build_report.py`. It scans every `marin-*` GCS bucket, deduplicates, builds a
markdown rollup, and publishes it as a public GitHub gist.

> Source of truth: `scripts/ops/storage/build_report.py`. This guide just
> documents how to drive it. (The `README.md` in this directory describes the
> larger *purge* workflow — prep/triage/compute/cleanup — which is a different,
> partially-future toolset. For "just give me the storage report," use
> `build_report.py`.)

## What it does

One command on your laptop submits three Iris jobs to the `marin` cluster, then
finishes locally:

| Stage | Runs on | What | Output |
|-------|---------|------|--------|
| 1. Scan   | Iris (coordinator + N worker replicas) | Walk every prefix in all 6 `marin-*` buckets | `STAGING/objects_*.parquet` (~100 MB segments) |
| 2. Dedup  | Iris (Zephyr `group_by`) | Collapse to one row per `(bucket, name)` — fixes ~1.8× inflation from overlapping scans / RPC retries | `STAGING/deduped/objects-*.parquet` |
| 3. Report | Iris coordinator | DuckDB rollup → markdown | `STAGING/report.md` |
| 4. Gist   | **Local laptop** | Fetch `report.md`, `gh gist create --public` | gist URL (printed) |

Default staging dir: `gs://marin-us-central2/tmp/storage-scan`.

The report contains: Overview, By Bucket, By Storage Class, Top 1/2/N-level
directory prefixes, Age Distribution, and a Monthly Creation Trend. Cost columns
use GCS list prices with a 30% CUD discount (see `constants.py`).

## Prerequisites

1. **Repo on `upstream/main`** (this checkout). The tooling is *not* on
   `origin/main`.
2. **`gh` authenticated** as the account that should own the public gist.
   - Check: `gh auth status` — needs the `gist` scope. ✅ (currently
     `XenonMolecule`, scope includes `gist`)
3. **Reachable `marin` Iris cluster.** The orchestrator opens a controller
   tunnel using the same machinery as `iris --cluster=marin ...`. Config lives
   at `lib/iris/config/marin.yaml`. Sanity-check the controller is up:
   ```bash
   uv run iris --cluster=marin status      # or: ... job list
   ```
4. **GCS access via ADC** so the local stage 4 can read `report.md` back:
   ```bash
   gcloud auth application-default login   # if not already set up
   ```
5. **Python deps** are resolved by `uv run` automatically (the script is a
   `uv run` shebang). The `marin` cluster image already has `iris`/`zephyr`.

## Run it

```bash
# Full pipeline (scan → dedup → report → public gist)
./scripts/ops/storage/build_report.py

# Common knobs
./scripts/ops/storage/build_report.py --workers 64      # fewer scan replicas (default 128)
./scripts/ops/storage/build_report.py --cluster marin   # default
./scripts/ops/storage/build_report.py --staging-dir gs://marin-us-central2/tmp/storage-scan
```

### Re-running cheaply (skip flags)

Each stage's output persists in the staging dir, so you can resume:

```bash
# Reuse an existing scan, just redo dedup+report+gist
./scripts/ops/storage/build_report.py --skip-scan

# Reuse scan+dedup, only rebuild the report and re-push the gist
./scripts/ops/storage/build_report.py --skip-scan --skip-dedup

# Already have report.md — just re-push the gist
./scripts/ops/storage/build_report.py --skip-scan --skip-dedup --skip-report
```

## Operational notes

- **Stragglers (per Russell Power):** the scan splits ~50k tasks unevenly, so a
  few checkpoint-heavy prefixes lag. It's normal to kill the scan once <100
  tasks remain rather than wait for full drain. Watch progress via
  `uv run iris --cluster=marin job list` / the job's logs.
- **This is a heavy, shared-cluster job.** Stage 1 launches up to `--workers`
  (default 128) replicas against the production `marin` cluster and lists every
  object in all 6 buckets. Coordinate before running at full width.
- **Stage 4 publishes a PUBLIC gist** under your `gh` account. Anyone with the
  URL can read it (and it may be indexed/cached even after deletion). It
  contains bucket names, directory prefixes, sizes, and cost estimates — no
  object contents, but it is internal infra detail. Use `--skip-report` style
  flags to inspect `report.md` first if you want to review before publishing:
  ```bash
  gcloud storage cat gs://marin-us-central2/tmp/storage-scan/report.md | less
  ```
- **Cost discipline:** the scan only *lists* object metadata (cheap), it does
  not read object bodies or move data across regions. Staging parquet is written
  to `marin-us-central2`.

## Quick prereq check

```bash
gh auth status                          # gist scope present?
uv run iris --cluster=marin status      # controller reachable?
gcloud auth application-default print-access-token >/dev/null && echo "ADC ok"
```

## Week-over-week changes

Each run archives a compact per-`(bucket, dir_prefix)` snapshot (only prefixes
≥ 1 GiB, so it's ~1 MiB) to a stable **history dir** (`--history-dir`, default
`gs://marin-us-central2/storage-report-history`). The next run loads the most
recent prior snapshot and inserts a **"Week-over-Week Changes"** section into
`report.md`, flagging prefixes whose size moved by ≥ 100 GiB
(`--change-threshold-gib`), classified grew / shrank / new / gone. The first
run shows "baseline established" since there's nothing to diff against.

The snapshot date is UTC, and a run never diffs against a snapshot from the
same UTC date (so a same-day re-run doesn't diff against itself) — two runs you
want to compare must fall on different UTC dates.

## Flags added for automation

| Flag | Purpose |
|------|---------|
| `--gist {public,secret,none}` | Stage 4 publish mode. `none` = skip (an outer digest publishes). Default `public`. |
| `--history-dir gs://…` | Where dated snapshots live for the diff. |
| `--run-id <id>` | Suffix for Iris job names (default = UTC date) so re-runs / weekly runs don't collide with prior same-named jobs. |
| `--change-threshold-gib N` | Flag prefixes whose size moved ≥ N GiB (default 100). |

## Automation (weekly)

There is **no cluster-native cron** — the schedule lives in GitHub Actions and
the heavy work runs as Iris jobs submitted from the runner (same mechanism as
`marin-canary-ferry`).

- `.github/workflows/ops-storage-report.yaml` — weekly cron (Mon 14:00 UTC) +
  manual `workflow_dispatch`. Authenticates to GCP (`IRIS_CI_GCP_SA_KEY`),
  installs the SSH key (`IRIS_CI_GCP_SSH_KEY`) for the controller tunnel, then
  runs the digest.
- `storage_report_digest.py` — runs the pipeline (`build_report --gist none`),
  fetches `report.md` via `gcloud` (avoids the local Python TLS issue),
  publishes a **secret** gist, and posts a summary (headline + biggest changes)
  to Discord (`internal-discuss` webhook). Test it without side effects:
  ```bash
  uv run python scripts/ops/storage/storage_report_digest.py --skip-pipeline --dry-run \
    --staging-dir gs://marin-us-central2/tmp/storage-report
  ```

Because the Actions secrets live in `marin-community/marin`, the workflow must
be merged upstream to run on the schedule; `workflow_dispatch` can trigger an
end-to-end test once merged.

**Discord:** posting needs only the `internal-discuss` **webhook** (no bot /
Discord permissions). CI has it as a secret; locally set
`DISCORD_WEBHOOK_INTERNAL_DISCUSS` or have `gcloud secrets` read access to
`marin-discord-webhook-internal-discuss`.
